# peak-ioc

A caproto IOC for Scienta Omicron analyzers driven by PEAK, with a simulation
mode. It is deliberately not an areaDetector: it serves the analyzer settings,
the image, a few fixed statistics, and optional one-file-per-acquisition HDF5
saving.

- `peak-ioc` talks to a real PEAK server through the vendor `peak` Python
  client, using the same calls as `haxpes.devices.peak_analyzer`. It has not
  been run against hardware yet. The `peak` package is not on PyPI or
  conda-forge, so install it separately wherever the real IOC runs.
- `peak-sim-ioc` serves the same PVs from a simulated analyzer: Gaussian core
  levels on a step background, spread over a Gaussian y profile, with Poisson
  noise. The shapes and states match what the HAXPES device saw; the numbers
  are made up.

## Running

```bash
pixi run peak-sim-ioc --prefix "PEAKSIM:" --list-pvs
```

```bash
pixi run peak-ioc --prefix "XF:07ID-ES{PEAK}:" --url http://xf07id-ws29-713.nsls2.bnl.gov:8087
```

The server URL can also come from `PEAK_URL`. `peak-sim-ioc --time-scale 0`
acquires instantly; the default of 1 waits as long as the real analyzer would
(points x dwell time when swept, the exposure time when fixed). Other options
are caproto's usual `ioc_arg_parser` ones.

```bash
pixi run -e test test
```

## How it works

The IOC (`ioc.py`) talks to a `Backend` (`backend.py`): `PeakBackend` in
`peak_api.py` or `SimBackend` in `sim.py`. Backend calls block and run in a
worker thread, one at a time.

- `Measure` = Start powers up the analyzer, moves it to the low end of the
  region with the live controls, waits `PrepositionWait` seconds (8 for the
  real IOC, 0 for the sim), and sets up the spectrum from the setting PVs.
  This is the HAXPES device's `stage()`. Stop finishes the measurement and
  powers down, like `unstage()`.
- `Acquire` = Acquire takes one spectrum; the put completes once the data PVs
  are written, so a put with completion is a whole `trigger()`. It starts a
  measurement first if none is running, and restarts it if the settings have
  changed since. With `Accumulate` Off (the default) the spectrum is cleared
  after each acquisition.
- `Active` and the `Live*` PVs change the powered-up analyzer immediately.

## PVs

All names follow the prefix.

| PV | Type | Notes |
| --- | --- | --- |
| `ScanMode` | enum | `swept`, `fixed` |
| `PassEnergy`, `DwellTime`, `ExposureTime` | float | `ExposureTime` is for fixed mode |
| `EnergyCenter`, `EnergyWidth`, `EnergyStep` | float | kinetic energy, eV; width and step are for swept mode |
| `LensMode`, `AcqMode`, `RegionName` | string | |
| `Description` | char waveform | |
| `<setting>_RBV` | | the analyzer's values, read back after each measurement start |
| `ReadSettings` | int | any put copies the analyzer's settings to the setpoints |
| `Active` | enum Off/On | power up or down the analyzer servers |
| `Measure` | enum Stop/Start | see above |
| `Acquire` | enum Done/Acquire | see above |
| `Accumulate` | enum Off/On | keep summing sweeps instead of clearing |
| `PrepositionWait` | float | seconds to settle after prepositioning |
| `LiveEnergy`, `LivePassEnergy`, `LiveLensMode`, `LiveDwellTime` | | only while `Active` is On |
| `State_RBV` | string | the PEAK state: Ready, Measuring or Acquiring |
| `Acquiring_RBV`, `Measuring_RBV` | enum Off/On | |
| `StatusMessage_RBV` | char waveform | last action or error |
| `ArrayCounter_RBV` | int | acquisitions so far |
| `Image` | float waveform | `SizeY_RBV` rows of `SizeX_RBV` energy channels, row-major |
| `SizeX_RBV`, `SizeY_RBV` | int | |
| `XAxis`, `YAxis` | float waveform | energy and y (angle) axes |
| `EDC` | float waveform | image summed over y |
| `YProfile` | float waveform | image summed over energy |
| `TotalCounts` | float | |
| `YCoM` | float | intensity-weighted y; 50 when there are no counts |
| `OptPar`, `OptVal` | float | `OptVal = TotalCounts - OptPar * abs(YCoM)`, for alignment |
| `AutoSave` | enum Off/On | write an HDF5 file per acquisition |
| `FilePath`, `FileName`, `FileTemplate` | char waveform | file is `FileTemplate % (FilePath, FileName, FileNumber)`; default template `%s%s_%04d.h5` |
| `FileNumber` | int | |
| `AutoIncrement` | enum Off/On | |
| `FullFileName_RBV` | char waveform | the last file written |

The char waveform PVs work like areaDetector's `FilePath`: read and write
them as char arrays (in ophyd, `EpicsSignal(..., string=True)`). Clients need
`EPICS_CA_MAX_ARRAY_BYTES` large enough for `Image` (8 bytes per pixel).

The HDF5 file has `entry/data/{image,energy,y,edc}` as a NeXus `NXdata`
group, the region settings as attributes of `entry/region`, and the
statistics as attributes of `entry/stats`.

## ophyd device

The bluesky side lives in HAXPES, as
`haxpes.devices.peak_analyzer_ioc.PeakAnalyzerIOC`. It keeps the component
names and methods of the direct-to-PEAK `PeakAnalyzer`, so documents and plans
are unchanged.

## Known gaps

- `PeakBackend` is untested against a PEAK server.
- The simulator's lens and acquisition modes, state names and count rates
  are guesses.
- Preposition goes to the low end of the region when swept and to the centre
  when fixed. The HAXPES device always went to centre minus half the width.
- There is no abort, and no multi-acquisition-per-file capture.
