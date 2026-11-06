"""
A simulated analyzer.

The states, axes and spectrum shapes follow what the HAXPES ophyd device
saw from the PEAK client; the numbers are made up. The spectrum is a few
Gaussian core levels on a step background, in kinetic energy, spread over
a Gaussian profile along the detector's y (angle) axis, with Poisson noise.
"""

import threading
import time
from dataclasses import replace

import numpy as np

from .backend import Axis, Backend, Region, Spectrum

# (kinetic energy, FWHM, relative intensity), placed around the default
# region centre so a default scan shows something.
DEFAULT_PEAKS = [(1996.0, 0.8, 0.6), (1999.5, 0.6, 1.0), (2003.0, 1.0, 0.4)]


class SimBackend(Backend):
    def __init__(self, peaks=None, x_channels=500, y_slices=200,
                 y_range=(-15.0, 15.0), beam_y=0.0, beam_width=4.0,
                 count_rate=2000.0, time_scale=1.0, seed=None):
        """
        ``x_channels`` is the fixed-mode energy channel count; ``y_slices``
        and ``y_range`` set the y axis. ``count_rate`` is the peak counts per
        second per channel at 50 eV pass energy. Acquisitions sleep for their
        estimated time multiplied by ``time_scale`` (0 for no waiting).
        """
        self.peaks = DEFAULT_PEAKS if peaks is None else peaks
        self.x_channels = x_channels
        self.y_axis = Axis(y_range[0], y_range[1], y_slices)
        self.beam_y = beam_y
        self.beam_width = beam_width
        self.count_rate = count_rate
        self.time_scale = time_scale
        self.rng = np.random.default_rng(seed)
        self.region = Region()
        self.connected = False
        self.active = False
        self.measuring = False
        self.acquiring = False
        self.live = {}
        self._accumulated = None
        self._abort = threading.Event()

    def connect(self):
        self.connected = True

    def get_state(self):
        if self.acquiring:
            return "Acquiring"
        if self.measuring:
            return "Measuring"
        return "Ready"

    def read_region(self):
        return replace(self.region)

    def x_axis(self, region=None):
        region = self.region if region is None else region
        if region.scan_mode == "fixed":
            # The detector sees roughly 8% of the pass energy at once.
            half = 0.04 * region.pass_energy
            return Axis(region.energy_center - half,
                        region.energy_center + half, self.x_channels)
        step = region.energy_step
        count = int(round(region.energy_width / step)) + 1
        low = region.energy_center - region.energy_width / 2
        return Axis(low, low + (count - 1) * step, count)

    def current_axes(self):
        return self.x_axis(), self.y_axis

    def set_scan_mode(self, scan_mode):
        self.region.scan_mode = scan_mode

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def set_live(self, pass_energy=None, lens_mode=None, kinetic_energy=None,
                 dwell_time=None):
        if not self.active:
            raise RuntimeError("Analyzer is not active")
        values = dict(pass_energy=pass_energy, lens_mode=lens_mode,
                      kinetic_energy=kinetic_energy, dwell_time=dwell_time)
        self.live.update({k: v for k, v in values.items() if v is not None})

    def start_measurement(self, region):
        if region.energy_step <= 0 and region.scan_mode == "swept":
            raise ValueError("Energy step must be positive")
        self.region = replace(region)
        self.measuring = True
        self._accumulated = None

    def acquisition_time(self):
        if self.region.scan_mode == "fixed":
            return self.region.exposure_time
        return self.x_axis().count * self.region.dwell_time

    def acquire(self):
        if not self.measuring:
            raise RuntimeError("No measurement in progress")
        self.acquiring = True
        try:
            self._abort.clear()
            if self._abort.wait(self.acquisition_time() * self.time_scale):
                raise RuntimeError("Acquisition aborted")
            data = self._make_data()
            if self._accumulated is not None:
                data = data + self._accumulated
            self._accumulated = data
            return Spectrum(data=data, x_axis=self.x_axis(),
                            y_axis=self.y_axis, metadata={"simulated": True})
        finally:
            self.acquiring = False

    def abort(self):
        self._abort.set()

    def clear_spectrum(self):
        self._accumulated = None

    def finish_measurement(self):
        self.measuring = False

    def _make_data(self):
        region = self.region
        energy = self.x_axis().array()
        # Resolution grows with pass energy, added in quadrature to the width.
        resolution = 0.002 * region.pass_energy
        edc = np.zeros_like(energy)
        background = np.zeros_like(energy)
        for center, fwhm, height in self.peaks:
            sigma = np.hypot(fwhm, resolution) / 2.3548
            edc += height * np.exp(-0.5 * ((energy - center) / sigma) ** 2)
            # Inelastic step: more background below each peak's kinetic energy.
            background += 0.15 * height / (1 + np.exp((energy - center) / sigma))
        edc += background + 0.02

        y = self.y_axis.array()
        profile = np.exp(-0.5 * ((y - self.beam_y) / (self.beam_width / 2.3548)) ** 2)

        if region.scan_mode == "fixed":
            seconds = region.exposure_time
        else:
            seconds = region.dwell_time
        scale = self.count_rate * seconds * region.pass_energy / 50.0
        expected = scale * np.outer(profile, edc) / self.y_axis.count
        return self.rng.poisson(expected).astype(np.float64)
