"""
The analyzer IOC.

A plain caproto IOC, not an areaDetector: setting PVs, a few control PVs,
the image and its fixed statistics, and optional one-file-per-acquisition
HDF5 saving. The same PVGroup serves the real analyzer (``peak-ioc``) and the
simulator (``peak-sim-ioc``); only the backend differs.

Control flow, following the HAXPES ophyd device:

- ``Measure`` = Start powers up the analyzer, moves it to the low end of the
  region with the live controls, waits ``PrepositionWait`` seconds, and sets
  up the spectrum from the setting PVs. With ``Preposition`` Off it only sets
  up the spectrum, for clients that power up and preposition through
  ``Active`` and the ``Live*`` PVs themselves. Stop finishes the measurement
  and powers the analyzer down.
- ``Acquire`` = Acquire takes one spectrum and completes the put when the
  data PVs are written. It starts a measurement first if there is none, or
  restarts it if the settings have changed. With ``Accumulate`` off the
  spectrum is cleared after each acquisition, so each one stands alone.
- ``Active`` and the ``Live*`` PVs drive the powered-up analyzer directly.
"""

import asyncio
import logging
from dataclasses import fields, replace

import numpy as np
from caproto import ChannelType
from caproto.server import PVGroup, ioc_arg_parser, pvproperty, run

from .backend import SCAN_MODES, Region
from .writer import make_filename, write_spectrum

log = logging.getLogger(__name__)

# Largest image, in pixels (y slices x energy channels), and largest axis.
MAX_IMAGE = 4 * 1024 * 1024
MAX_AXIS = 65536
OFF_ON = ["Off", "On"]


def _string(value="", **kwargs):
    """A short string PV (DBR_STRING, at most 40 characters)."""
    return pvproperty(value=value, dtype=ChannelType.STRING, **kwargs)


def _long_string(value="", max_length=256, **kwargs):
    """
    A string PV served as a char waveform, like areaDetector's FilePath.
    Clients read and write it as a char array (ophyd: ``string=True``);
    reporting it as DBR_STRING would cut it to 40 characters.
    """
    return pvproperty(value=value, dtype=ChannelType.CHAR, max_length=max_length,
                      string_encoding="utf-8", **kwargs)


def _enum(value, enum_strings, **kwargs):
    return pvproperty(value=value, enum_strings=enum_strings,
                      dtype=ChannelType.ENUM, **kwargs)


def _array(max_length, **kwargs):
    return pvproperty(value=[0.0], max_length=max_length, read_only=True,
                      **kwargs)


def y_center_of_mass(spectrum):
    """The intensity-weighted mean of the y axis, or 50 with no counts."""
    profile = spectrum.data.sum(axis=1)
    try:
        return float(np.average(spectrum.y_axis.array(), weights=profile))
    except ZeroDivisionError:
        # As in the HAXPES device: off the y axis, e.g. sample too far away.
        return 50.0


class PeakIOC(PVGroup):
    # Settings, applied at the next measurement start.
    scan_mode = _enum("swept", SCAN_MODES, name="ScanMode")
    pass_energy = pvproperty(value=50.0, name="PassEnergy", precision=2, units="eV")
    lens_mode = _string("Angular", name="LensMode")
    acq_mode = _string("Image", name="AcqMode")
    dwell_time = pvproperty(value=0.1, name="DwellTime", precision=3, units="s")
    exposure_time = pvproperty(value=1.0, name="ExposureTime", precision=3,
                               units="s")
    energy_center = pvproperty(value=2000.0, name="EnergyCenter", precision=3,
                               units="eV")
    energy_width = pvproperty(value=15.0, name="EnergyWidth", precision=3,
                              units="eV")
    energy_step = pvproperty(value=0.1, name="EnergyStep", precision=4,
                             units="eV")
    region_name = _string("Region", name="RegionName")
    description = _long_string("Description", name="Description")

    # Readbacks of the settings, as validated by the analyzer.
    scan_mode_rbv = _enum("swept", SCAN_MODES, name="ScanMode_RBV",
                          read_only=True)
    pass_energy_rbv = pvproperty(value=50.0, name="PassEnergy_RBV", precision=2,
                                 units="eV", read_only=True)
    lens_mode_rbv = _string("Angular", name="LensMode_RBV", read_only=True)
    acq_mode_rbv = _string("Image", name="AcqMode_RBV", read_only=True)
    dwell_time_rbv = pvproperty(value=0.1, name="DwellTime_RBV", precision=3,
                                units="s", read_only=True)
    exposure_time_rbv = pvproperty(value=1.0, name="ExposureTime_RBV",
                                   precision=3, units="s", read_only=True)
    energy_center_rbv = pvproperty(value=2000.0, name="EnergyCenter_RBV",
                                   precision=3, units="eV", read_only=True)
    energy_width_rbv = pvproperty(value=15.0, name="EnergyWidth_RBV",
                                  precision=3, units="eV", read_only=True)
    energy_step_rbv = pvproperty(value=0.1, name="EnergyStep_RBV", precision=4,
                                 units="eV", read_only=True)
    region_name_rbv = _string("Region", name="RegionName_RBV", read_only=True)
    description_rbv = _long_string("Description", name="Description_RBV",
                              read_only=True)

    # Control.
    active = _enum("Off", OFF_ON, name="Active")
    measure = _enum("Stop", ["Stop", "Start"], name="Measure")
    acquire = _enum("Done", ["Done", "Acquire"], name="Acquire")
    accumulate = _enum("Off", OFF_ON, name="Accumulate")
    read_settings = pvproperty(value=0, name="ReadSettings")
    preposition = _enum("On", OFF_ON, name="Preposition")
    preposition_wait = pvproperty(value=8.0, name="PrepositionWait", precision=1,
                                  units="s")
    live_energy = pvproperty(value=0.0, name="LiveEnergy", precision=3,
                             units="eV")
    live_pass_energy = pvproperty(value=50.0, name="LivePassEnergy",
                                  precision=2, units="eV")
    live_lens_mode = _string("Angular", name="LiveLensMode")
    live_dwell_time = pvproperty(value=0.1, name="LiveDwellTime", precision=3,
                                 units="s")

    # Status.
    analyzer_state = _string("Disconnected", name="State_RBV", read_only=True)
    acquiring = _enum("Off", OFF_ON, name="Acquiring_RBV", read_only=True)
    measuring = _enum("Off", OFF_ON, name="Measuring_RBV", read_only=True)
    status_message = _long_string("", name="StatusMessage_RBV", read_only=True)
    array_counter = pvproperty(value=0, name="ArrayCounter_RBV", read_only=True)

    # Data.
    image = _array(MAX_IMAGE, name="Image")
    size_x = pvproperty(value=0, name="SizeX_RBV", read_only=True)
    size_y = pvproperty(value=0, name="SizeY_RBV", read_only=True)
    x_axis = _array(MAX_AXIS, name="XAxis")
    y_axis = _array(MAX_AXIS, name="YAxis")
    edc = _array(MAX_AXIS, name="EDC")
    y_profile = _array(MAX_AXIS, name="YProfile")
    total_counts = pvproperty(value=0.0, name="TotalCounts", read_only=True)
    y_com = pvproperty(value=0.0, name="YCoM", precision=3, read_only=True)
    opt_par = pvproperty(value=0.5, name="OptPar", precision=3)
    opt_val = pvproperty(value=0.0, name="OptVal", precision=3, read_only=True)

    # File saving, one HDF5 file per acquisition.
    auto_save = _enum("Off", OFF_ON, name="AutoSave")
    file_path = _long_string("/tmp/", name="FilePath")
    file_name = _long_string("peak", name="FileName")
    file_template = _long_string("%s%s_%04d.h5", name="FileTemplate")
    file_number = pvproperty(value=0, name="FileNumber")
    auto_increment = _enum("On", OFF_ON, name="AutoIncrement")
    full_file_name = _long_string("", max_length=1024, name="FullFileName_RBV",
                             read_only=True)

    # Setting PV attribute -> Region field. The readback is the same
    # attribute with "_rbv".
    SETTINGS = [f.name for f in fields(Region)]

    def __init__(self, *args, backend, preposition_wait=8.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.initial_preposition_wait = preposition_wait
        self.lock = asyncio.Lock()
        self.configured = None

    async def call(self, func, *args, **kwargs):
        """Run a blocking backend call in a worker thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    def region(self):
        return Region(**{name: getattr(self, name).value
                         for name in self.SETTINGS})

    async def report(self, message):
        log.info(message)
        await self.status_message.write(message[:255])

    async def refresh_state(self):
        await self.analyzer_state.write(await self.call(self.backend.get_state))

    async def read_back(self, setpoints=False):
        """Copy the analyzer's settings to the readbacks (and setpoints)."""
        region = await self.call(self.backend.read_region)
        for name in self.SETTINGS:
            value = getattr(region, name)
            if value is None:
                continue
            await getattr(self, name + "_rbv").write(value)
            if setpoints:
                await getattr(self, name).write(value)
        x_axis, y_axis = await self.call(self.backend.current_axes)
        await self.x_axis.write(x_axis.array())
        await self.y_axis.write(y_axis.array())

    @analyzer_state.startup
    async def analyzer_state(self, instance, async_lib):
        """Connect, load the analyzer's settings, then poll its state."""
        await self.preposition_wait.write(self.initial_preposition_wait)
        try:
            await self.call(self.backend.connect)
            await self.read_back(setpoints=True)
            await self.report("Connected")
        except Exception as e:
            log.exception("Could not connect to the analyzer")
            await self.report(f"Connect failed: {e}")
            return
        while True:
            if not self.lock.locked():
                try:
                    await self.refresh_state()
                except Exception:
                    log.exception("Could not read the analyzer state")
            await asyncio.sleep(1.0)

    async def _set_active(self, on):
        if on:
            await self.call(self.backend.activate)
        else:
            await self.call(self.backend.deactivate)
        await self.active.write("On" if on else "Off", verify_value=False)

    async def _start_measurement(self, region, preposition=True):
        """Power up, preposition and set up the spectrum. Holds the lock."""
        if preposition:
            await self._set_active(True)
            if region.scan_mode == "swept":
                start = region.energy_center - region.energy_width / 2
            else:
                start = region.energy_center
            await self.call(self.backend.set_live, pass_energy=region.pass_energy,
                            lens_mode=region.lens_mode, kinetic_energy=start)
            await asyncio.sleep(self.preposition_wait.value)
        await self.call(self.backend.start_measurement, region)
        self.configured = replace(region)
        await self.measure.write("Start", verify_value=False)
        await self.measuring.write("On")
        await self.read_back()
        await self.refresh_state()

    async def _finish_measurement(self):
        await self.call(self.backend.finish_measurement)
        self.configured = None
        await self.measure.write("Stop", verify_value=False)
        await self.measuring.write("Off")
        await self._set_active(False)
        await self.refresh_state()

    @active.putter
    async def active(self, instance, value):
        async with self.lock:
            await self._set_active(value == "On")
        return value

    @measure.putter
    async def measure(self, instance, value):
        async with self.lock:
            try:
                if value == "Start":
                    await self.report("Starting measurement")
                    await self._start_measurement(
                        self.region(), preposition=self.preposition.value == "On")
                    await self.report("Measuring")
                else:
                    await self._finish_measurement()
                    await self.report("Measurement finished")
            except Exception as e:
                log.exception("Measure %s failed", value)
                await self.report(f"Measure {value} failed: {e}")
                raise
        return value

    @acquire.putter
    async def acquire(self, instance, value):
        if value != "Acquire":
            return "Done"
        async with self.lock:
            try:
                region = self.region()
                if self.configured is None:
                    await self.report("Starting measurement")
                    await self._start_measurement(
                        region, preposition=self.preposition.value == "On")
                elif region != self.configured:
                    await self.report("Settings changed, restarting measurement")
                    await self.call(self.backend.finish_measurement)
                    await self._start_measurement(region, preposition=False)
                await self.acquiring.write("On")
                await self.analyzer_state.write("Acquiring")
                spectrum = await self.call(self.backend.acquire)
                if self.accumulate.value == "Off":
                    await self.call(self.backend.clear_spectrum)
                await self.publish(spectrum)
                if self.auto_save.value == "On":
                    await self.save(spectrum)
                await self.report("Acquired")
            except Exception as e:
                log.exception("Acquire failed")
                await self.report(f"Acquire failed: {e}")
                raise
            finally:
                await self.acquiring.write("Off")
                await self.refresh_state()
        return "Done"

    async def publish(self, spectrum):
        data = np.asarray(spectrum.data, dtype=np.float64)
        if data.size > MAX_IMAGE:
            raise ValueError(f"Image of {data.size} pixels exceeds {MAX_IMAGE}")
        size_y, size_x = data.shape
        y_com = y_center_of_mass(spectrum)
        total = float(data.sum())
        await self.size_x.write(size_x)
        await self.size_y.write(size_y)
        await self.x_axis.write(spectrum.x_axis.array())
        await self.y_axis.write(spectrum.y_axis.array())
        await self.edc.write(data.sum(axis=0))
        await self.y_profile.write(data.sum(axis=1))
        await self.total_counts.write(total)
        await self.y_com.write(y_com)
        await self.opt_val.write(total - self.opt_par.value * abs(y_com))
        await self.image.write(data.ravel())
        await self.array_counter.write(self.array_counter.value + 1)

    async def save(self, spectrum):
        filename = make_filename(self.file_template.value, self.file_path.value,
                                 self.file_name.value, self.file_number.value)
        stats = {"total_counts": self.total_counts.value,
                 "y_com": self.y_com.value,
                 "array_counter": self.array_counter.value}
        await self.call(write_spectrum, filename, spectrum,
                        self.configured, stats)
        await self.full_file_name.write(filename)
        if self.auto_increment.value == "On":
            await self.file_number.write(self.file_number.value + 1)

    @read_settings.putter
    async def read_settings(self, instance, value):
        async with self.lock:
            await self.read_back(setpoints=True)
        return 0

    async def _live(self, **kwargs):
        if self.active.value != "On":
            raise RuntimeError("Set Active to On before using live controls")
        async with self.lock:
            await self.call(self.backend.set_live, **kwargs)

    @live_energy.putter
    async def live_energy(self, instance, value):
        await self._live(kinetic_energy=value)
        return value

    @live_pass_energy.putter
    async def live_pass_energy(self, instance, value):
        await self._live(pass_energy=value)
        await self.pass_energy.write(value)
        return value

    @live_lens_mode.putter
    async def live_lens_mode(self, instance, value):
        await self._live(lens_mode=value)
        await self.lens_mode.write(value)
        return value

    @live_dwell_time.putter
    async def live_dwell_time(self, instance, value):
        await self._live(dwell_time=value)
        await self.dwell_time.write(value)
        return value


def _run(make_backend, description, default_prefix, preposition_wait):
    ioc_options, run_options = ioc_arg_parser(
        default_prefix=default_prefix, desc=description,
        supported_async_libs=("asyncio",),
    )
    logging.basicConfig(level=logging.INFO)
    ioc = PeakIOC(backend=make_backend(), preposition_wait=preposition_wait,
                  **ioc_options)
    run(ioc.pvdb, **run_options)


def _parse_own_args(add_arguments):
    """Take this script's own options off sys.argv, leaving caproto's."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    args, sys.argv[1:] = parser.parse_known_args()
    return args


def main():
    """Run the IOC against a real PEAK server."""
    import os

    from .peak_api import DEFAULT_URL, PeakBackend

    args = _parse_own_args(lambda p: p.add_argument(
        "--url", default=os.environ.get("PEAK_URL", DEFAULT_URL)))
    _run(lambda: PeakBackend(args.url),
         f"PEAK analyzer IOC. Server URL from --url or PEAK_URL "
         f"(default {DEFAULT_URL})", "PEAK:", preposition_wait=8.0)


def sim_main():
    """Run the IOC against the simulated analyzer."""
    from .sim import SimBackend

    args = _parse_own_args(lambda p: p.add_argument(
        "--time-scale", type=float, default=1.0))
    _run(lambda: SimBackend(time_scale=args.time_scale),
         "Simulated PEAK analyzer IOC. --time-scale multiplies the "
         "acquisition time (0 for none)", "PEAKSIM:", preposition_wait=0.0)
