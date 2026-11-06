import numpy as np
import pytest

from peak_ioc.backend import Region
from peak_ioc.sim import SimBackend


@pytest.fixture
def sim():
    backend = SimBackend(time_scale=0, seed=1)
    backend.connect()
    return backend


def test_swept_spectrum_shape(sim):
    sim.start_measurement(Region(energy_center=2000, energy_width=15,
                                 energy_step=0.1))
    spectrum = sim.acquire()
    assert spectrum.data.shape == (200, 151)
    assert spectrum.x_axis.count == 151
    assert spectrum.x_axis.minimum == pytest.approx(1992.5)
    assert spectrum.x_axis.maximum == pytest.approx(2007.5)
    assert spectrum.data.sum() > 0


def test_fixed_spectrum_shape(sim):
    sim.start_measurement(Region(scan_mode="fixed", pass_energy=100,
                                 energy_center=1999.5))
    spectrum = sim.acquire()
    assert spectrum.data.shape == (200, 500)
    assert spectrum.x_axis.maximum - spectrum.x_axis.minimum == pytest.approx(8)


def test_peak_lands_at_its_energy(sim):
    sim.start_measurement(Region(energy_center=1999.5, energy_width=4,
                                 energy_step=0.05, dwell_time=10))
    spectrum = sim.acquire()
    edc = spectrum.data.sum(axis=0)
    assert spectrum.x_axis.array()[np.argmax(edc)] == pytest.approx(1999.5, abs=0.1)


def test_accumulates_until_cleared(sim):
    sim.start_measurement(Region())
    first = sim.acquire().data
    second = sim.acquire().data
    assert second.sum() > first.sum()
    sim.clear_spectrum()
    assert sim.acquire().data.sum() < second.sum()


def test_states(sim):
    assert sim.get_state() == "Ready"
    sim.start_measurement(Region())
    assert sim.get_state() == "Measuring"
    sim.finish_measurement()
    assert sim.get_state() == "Ready"
    with pytest.raises(RuntimeError):
        sim.acquire()


def test_live_needs_active(sim):
    with pytest.raises(RuntimeError):
        sim.set_live(pass_energy=20)
    sim.activate()
    sim.set_live(pass_energy=20, kinetic_energy=1990)
    assert sim.live == {"pass_energy": 20, "kinetic_energy": 1990}
