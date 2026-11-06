"""
A real analyzer, driven through the vendor ``peak`` client.

The calls are taken from ``haxpes.devices.peak_analyzer.PeakAnalyzer``; this
module has not yet been run against a PEAK server.
"""

import logging
from time import sleep

from .backend import Axis, Backend, Region, Spectrum

log = logging.getLogger(__name__)

DEFAULT_URL = "http://xf07id-ws29-713.nsls2.bnl.gov:8087"


def _axis(peak_axis):
    return Axis(peak_axis.minimum, peak_axis.maximum, peak_axis.count)


class PeakBackend(Backend):
    def __init__(self, url=DEFAULT_URL, activate_wait=2.0):
        # Imported here so the simulator runs without the vendor package.
        import peak
        from peak.types.peak_acquisition_mode import PeakAcquisitionMode
        from peak.types.peak_axis import PeakAxis
        from peak.types.peak_axis_mode import PeakAxisMode
        from peak.types.peak_energy_mode import PeakEnergyMode

        self._peak = peak
        self._PeakAcquisitionMode = PeakAcquisitionMode
        self._PeakAxis = PeakAxis
        self._PeakAxisMode = PeakAxisMode
        self._PeakEnergyMode = PeakEnergyMode
        self.url = url
        self.activate_wait = activate_wait
        self._peakclient = None
        self._acqclient = None
        self._anclient = None
        self._anclient_connected = False

    def connect(self):
        self._peakclient = self._peak.PeakClient(self.url)
        self._peakclient.connect()
        self._acqclient = self._peak.AcquireSpectrumClient(self._peakclient)
        self._acqclient.connect()
        self._anclient = self._peak.AnalyserControlClient(self._peakclient)

    def get_state(self):
        return str(self._acqclient.get_state())

    def read_region(self):
        specdef = self._acqclient.spectrum_definition
        x_axis = self._acqclient.current_x_axis
        if self._PeakAxis.X in specdef.sweep_axes:
            scan_mode = "swept"
        else:
            scan_mode = "fixed"
        return Region(
            scan_mode=scan_mode,
            pass_energy=specdef.pass_energy,
            lens_mode=specdef.lens_mode_name,
            acq_mode=specdef.acquisition_mode.value,
            dwell_time=specdef.dwell_time,
            exposure_time=specdef.acquisition_time,
            energy_center=x_axis.center,
            energy_width=x_axis.width,
            energy_step=x_axis.delta,
            region_name=specdef.name,
            # The HAXPES device never read the description back; None
            # leaves the IOC's value alone.
            description=None,
        )

    def current_axes(self):
        return (_axis(self._acqclient.current_x_axis),
                _axis(self._acqclient.current_y_axis))

    def set_scan_mode(self, scan_mode):
        mode = "Sweep" if scan_mode == "swept" else "Fixed"
        self._acqclient.set_x_axis_mode(self._PeakAxisMode(mode))

    def activate(self):
        if not self._anclient_connected:
            self._anclient.connect()
            self._anclient_connected = True
        if self._acqclient.get_state() == "Ready":
            self._peakclient.activate_servers()
            sleep(self.activate_wait)
        else:
            log.info("Analyzer already active")

    def deactivate(self):
        self._peakclient.deactivate_servers()

    def set_live(self, pass_energy=None, lens_mode=None, kinetic_energy=None,
                 dwell_time=None):
        if pass_energy is not None:
            self._anclient.set_pass_energy(pass_energy)
        if lens_mode is not None:
            self._anclient.set_lens_mode(lens_mode)
        if dwell_time is not None:
            self._anclient.set_dwell_time(dwell_time)
        if kinetic_energy is not None:
            self._anclient.set_analyser_energy(
                kinetic_energy, self._PeakEnergyMode("Kinetic")
            )

    def start_measurement(self, region):
        acq = self._acqclient
        self.set_scan_mode(region.scan_mode)
        acq.start_measurement()
        acq.set_pass_energy(region.pass_energy)
        acq.set_lens_mode(region.lens_mode)
        acq.set_acquisition_mode(self._PeakAcquisitionMode(region.acq_mode))
        acq.set_dwell_time(region.dwell_time)
        acq.set_name(region.region_name)
        acq.set_description(region.description)
        if region.scan_mode == "swept":
            acq.set_x_axis_delta(region.energy_step)
            acq.set_x_axis_center(region.energy_center)
            acq.set_x_axis_width(region.energy_width)
        else:
            acq.set_x_axis_center(region.energy_center)
            acq.set_acquisition_time(region.exposure_time)
        acq.setup_spectrum()

    def acquire(self):
        self._acqclient.acquire_spectrum()
        specdat = self._acqclient.get_spectrum()
        return Spectrum(
            data=specdat.data,
            x_axis=_axis(specdat.x_axis),
            y_axis=_axis(specdat.y_axis),
        )

    def clear_spectrum(self):
        self._acqclient.clear_spectrum()

    def finish_measurement(self):
        state = self._acqclient.get_state()
        while state == "Acquiring":
            sleep(0.5)
            state = self._acqclient.get_state()
        if state == "Measuring":
            self._acqclient.finish_measurement()
