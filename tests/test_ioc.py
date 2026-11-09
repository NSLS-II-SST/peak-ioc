"""Drive the IOC's PVs directly, without a network server."""

import h5py
import numpy as np
import pytest

from peak_ioc.ioc import PeakIOC
from peak_ioc.sim import SimBackend


@pytest.fixture
async def ioc():
    ioc = PeakIOC(prefix="TEST:", backend=SimBackend(time_scale=0, seed=1),
                  preposition_wait=0.0)
    await ioc.preposition_wait.write(0.0)
    await ioc.call(ioc.backend.connect)
    await ioc.read_back(setpoints=True)
    return ioc


async def test_acquire_starts_measurement_and_publishes(ioc):
    await ioc.acquire.write("Acquire")
    assert ioc.acquire.value == "Done"
    assert ioc.active.value == "On"
    assert ioc.measure.value == "Start"
    assert ioc.measuring.value == "On"
    assert ioc.analyzer_state.value == "Measuring"
    assert ioc.array_counter.value == 1
    size_x, size_y = ioc.size_x.value, ioc.size_y.value
    assert (size_y, size_x) == (200, 151)
    image = np.asarray(ioc.image.value).reshape(size_y, size_x)
    assert ioc.total_counts.value == pytest.approx(image.sum())
    assert np.allclose(ioc.edc.value, image.sum(axis=0))
    assert len(ioc.x_axis.value) == size_x
    assert len(ioc.y_axis.value) == size_y
    assert ioc.y_com.value == pytest.approx(0, abs=0.5)
    assert ioc.opt_val.value == pytest.approx(
        ioc.total_counts.value - 0.5 * abs(ioc.y_com.value))
    # Live preposition went to the low end of the region.
    assert ioc.backend.live["kinetic_energy"] == pytest.approx(1992.5)


async def test_changed_settings_restart_measurement(ioc):
    await ioc.acquire.write("Acquire")
    await ioc.energy_width.write(5.0)
    await ioc.scan_mode.write("swept")
    await ioc.acquire.write("Acquire")
    assert ioc.size_x.value == 51
    assert ioc.energy_width_rbv.value == 5.0


async def test_fixed_mode(ioc):
    await ioc.scan_mode.write("fixed")
    await ioc.acquire.write("Acquire")
    assert ioc.scan_mode_rbv.value == "fixed"
    assert ioc.size_x.value == 500


async def test_accumulate(ioc):
    await ioc.accumulate.write("On")
    await ioc.acquire.write("Acquire")
    first = ioc.total_counts.value
    await ioc.acquire.write("Acquire")
    assert ioc.total_counts.value > 1.5 * first


async def test_measure_stop_powers_down(ioc):
    await ioc.measure.write("Start")
    assert ioc.measuring.value == "On"
    await ioc.measure.write("Stop")
    assert ioc.measuring.value == "Off"
    assert ioc.active.value == "Off"
    assert ioc.analyzer_state.value == "Ready"


async def test_autosave(ioc, tmp_path):
    await ioc.file_path.write(f"{tmp_path}/")
    await ioc.file_name.write("scan")
    await ioc.file_number.write(7)
    await ioc.auto_save.write("On")
    await ioc.region_name.write("Au4f")
    await ioc.acquire.write("Acquire")
    filename = f"{tmp_path}/scan_0007.h5"
    assert ioc.full_file_name.value == filename
    assert ioc.file_number.value == 8
    with h5py.File(filename) as f:
        image = f["entry/data/image"][()]
        assert image.shape == (200, 151)
        assert image.sum() == pytest.approx(ioc.total_counts.value)
        assert len(f["entry/data/energy"]) == 151
        assert f["entry/region"].attrs["region_name"] == "Au4f"
        assert f["entry"].attrs["simulated"]


async def test_live_controls_need_active(ioc):
    with pytest.raises(RuntimeError):
        await ioc.live_pass_energy.write(20.0)
    await ioc.active.write("On")
    await ioc.live_pass_energy.write(20.0)
    assert ioc.backend.live["pass_energy"] == 20.0
    assert ioc.pass_energy.value == 20.0


async def test_measure_without_preposition(ioc):
    await ioc.preposition.write("Off")
    await ioc.active.write("On")
    await ioc.live_energy.write(1980.0)
    await ioc.measure.write("Start")
    assert ioc.measuring.value == "On"
    # The client's live energy is left alone.
    assert ioc.backend.live["kinetic_energy"] == 1980.0
