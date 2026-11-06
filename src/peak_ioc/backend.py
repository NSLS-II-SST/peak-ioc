"""
The interface between the IOC and an analyzer.

The IOC only talks to a ``Backend``. ``peak_api.PeakBackend`` drives a real
analyzer through the PEAK client; ``sim.SimBackend`` makes up data of the
same shape. Backend methods are blocking and are run in a worker thread by
the IOC, one at a time.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

SCAN_MODES = ["swept", "fixed"]


@dataclass
class Axis:
    minimum: float
    maximum: float
    count: int

    def array(self):
        return np.linspace(self.minimum, self.maximum, self.count)


@dataclass
class Region:
    """Analyzer settings for one spectrum, as the IOC's setting PVs hold them."""

    scan_mode: str = "swept"
    pass_energy: float = 50.0
    lens_mode: str = "Angular"
    acq_mode: str = "Image"
    dwell_time: float = 0.1
    # Fixed mode only
    exposure_time: float = 1.0
    energy_center: float = 2000.0
    # Swept mode only
    energy_width: float = 15.0
    energy_step: float = 0.1
    region_name: str = "Region"
    description: str = "Description"


@dataclass
class Spectrum:
    """One acquired spectrum. ``data`` has shape (y count, x count)."""

    data: np.ndarray
    x_axis: Axis
    y_axis: Axis
    metadata: dict = field(default_factory=dict)


class Backend(ABC):
    @abstractmethod
    def connect(self):
        """Connect to the analyzer server."""

    @abstractmethod
    def get_state(self):
        """The analyzer state string, e.g. "Ready", "Measuring", "Acquiring"."""

    @abstractmethod
    def read_region(self):
        """The current analyzer settings, as validated by the server."""

    @abstractmethod
    def current_axes(self):
        """The (x, y) axes the next spectrum will have."""

    @abstractmethod
    def set_scan_mode(self, scan_mode):
        """Switch between "swept" and "fixed" energy mode."""

    @abstractmethod
    def activate(self):
        """Power up the analyzer servers (live mode)."""

    @abstractmethod
    def deactivate(self):
        """Power down the analyzer servers."""

    @abstractmethod
    def set_live(self, pass_energy=None, lens_mode=None, kinetic_energy=None,
                 dwell_time=None):
        """Change settings on the live analyzer immediately."""

    @abstractmethod
    def start_measurement(self, region):
        """Start a measurement and set up the spectrum from ``region``."""

    @abstractmethod
    def acquire(self):
        """Acquire one spectrum and return it as a ``Spectrum``."""

    @abstractmethod
    def clear_spectrum(self):
        """Discard accumulated data, so the next acquire starts from zero."""

    @abstractmethod
    def finish_measurement(self):
        """Finish the measurement once any acquisition has ended."""
