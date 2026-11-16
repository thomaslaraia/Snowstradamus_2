# Running FSM2 on your ERA5-Land grid

A practical guide from installation to hourly canopy snow mass • 10 November 2026

## 1. What you will get

This guide and the accompanying scripts turn your monthly `driving_data.nc` files into monthly model outputs containing:

- `canopy_snow_mass(time, latitude, longitude)`, in **kg m⁻²**: the main research output;
- `ground_snow_mass(time, latitude, longitude)`, in kg m⁻²;
- `snow_depth(time, latitude, longitude)`, in metres;
- the static VAI, canopy height and a mask identifying simulated forest cells.

You do not need to learn Fortran to follow this workflow. Python prepares and checks each cell's forcing, launches the compiled model, and assembles the results. The Fortran physics are left unchanged.

**Your preparation script supplies the core meteorological and vegetation variables. It does not yet supply directly runnable FSM2 forcing.** It retains ERA5-Land units and accumulation conventions. The supplied runner performs the additional conversions.

I inspected your attached `19-tl-ERA5-FSM2-prep-n-download.py`, the upstream FSM2 source and its user guide. I did not have your actual NetCDF files. Check their contents and units in section 5 before starting a production run. Existing files may have been created by an earlier script: the attached script skips existing outputs.

The source inspected identifies itself as **FSM2 2.1.2**, at commit:

`a76945e75b36628161f4679428ed72dce090b368`

The published 2025 description is for version 2.1.1. The build script fixes the code to the inspected commit so later repository changes do not silently alter this workflow. Source links are in section 16.

## 2. How a grid run works

FSM2 advances the snow, vegetation and soil state through time in a vertical column. For this workflow, each ERA5-Land cell is a separate column with its own weather, VAI and canopy height. There is no exchange of snow or water between neighbouring cells, and no terrain redistribution is introduced.

The order of work is:

1. Compile FSM2 once to create an executable program.
2. Read one cell's weather across all consecutive monthly files.
3. Convert its accumulated weather into hourly forcing.
4. Spin up its snow, canopy and soil state.
5. Run the full chronological series without resetting at month boundaries.
6. Save that cell's output and repeat for other cells.
7. Put the completed cell outputs back onto the latitude–longitude grid.

**Do not set `Npnts` equal to your number of ERA5 grid cells and provide one weather series.** In this source version, all `Npnts` points receive the same meteorology. That option is useful for different vegetation conditions under common forcing, not for a regional grid with spatially varying weather. Our runner uses `Npnts=1` for each independent cell, then parallelises cells with Slurm.

This produces results on your existing ERA5-Land grid. It does not downscale to 15 m or 30 m ICESat-2 segments. Comparing model states with those segments will require a separate spatial and temporal matching step.

## 3. Files in this starter package

| File | Purpose |
|---|---|
| `GUIDE.md` | This guide, readable in a text editor or Markdown viewer |
| `build_fsm2.sh` | Downloads the pinned FSM2 source and compiles the chosen configuration |
| `run_grid.py` | Converts forcing, runs cells, checks outputs and saves completed cells |
| `assemble_grid.py` | Assembles completed cells into monthly gridded NetCDFs |
| `run_fsm2.slurm` | Example job array: one latitude row per job |
| `VALIDATION.md` | What was tested and what remains to check on your data |

Keep these files together in a new project directory, for example `~/projects/FSM2_starter`. Keep the large input and result files in your normal scratch allocation. Do not put results over the prepared inputs.

The attachment's defaults are:

- raw weather: `../../../scratch/ERA5/<region>/YYYY_MM/YYYY_MM.nc`;
- prepared weather plus vegetation: `../../../scratch/fsm2/<region>/YYYY_MM/driving_data.nc`.

Use the **second** set as model inputs. `<region>` is whatever name the preparation used, such as `S01` or `ontario`. Its current default dates are July 2018–June 2026, but the runner uses the files actually present. It does not assume those months all exist.

Relative paths such as `../../../scratch` depend on where you ran the preparation command. They are not automatically relative to the script file. Locate your actual files and use absolute paths below.

## 4. Install the tools and compile

### 4.1 Open a Linux terminal

Run this on the Linux system where your data live, through your usual SSH connection. The Slurm example is for that system. These are terminal commands, not notebook cells. You can also use Linux or WSL on a PC for a small test, but would need the data there.

A few terms:

| Term | Meaning here |
|---|---|
| Fortran source | The model's `.F90` text files |
| Compiler | `gfortran`, which turns the source into a program |
| Executable | The resulting file called `FSM2` |
| Namelist | A short settings file that FSM2 reads when it starts |
| Shell script | A file of terminal commands, run with `bash` |
| Spin-up | A preliminary simulation used to establish model state |

### 4.2 Check your compiler and Python

```bash
cd ~/projects/FSM2_starter
git --version
gfortran --version
python --version
```

If `gfortran` is unavailable on your cluster, inspect its available compiler modules:

```bash
module avail gcc
module avail gfortran
```

Load the GCC/compiler module listed by your system, then repeat `gfortran --version`. Module names and accounts are cluster-specific; I cannot infer their exact current names from the preparation script. Ask support for a GNU Fortran compiler if neither command finds one. Use the same environment for compilation and batch execution.

On a personally administered Ubuntu/WSL installation, the usual installation is:

```bash
sudo apt update
sudo apt install git gfortran
```

Do not run that installation command on a managed cluster.

Create a Python environment if you have Conda:

```bash
conda create -n fsm2 -c conda-forge python=3.11 numpy pandas xarray dask netcdf4 matplotlib
conda activate fsm2
```

Alternatively, install those Python packages in your existing suitable environment. The Fortran build here does **not** need the Fortran NetCDF library: Python handles NetCDF input and final output.

### 4.3 Build the executable

From the folder containing the supplied scripts:

```bash
bash build_fsm2.sh
ls -lh FSM2_source/FSM2
```

Compilation means making the program; it does not run your data. The message about a “detached HEAD” from Git is expected: it means the source is fixed to a particular commit.

The script writes `physics_options.h` and `source_commit.txt` beside the runner. Keep them with your experiment record. Recompile whenever you change a compile-time physics option. Changing a Python setting does not change physics compiled into an existing executable.

The supplied build uses:

| Setting | Value | Reason |
|---|---:|---|
| `DRIV1D` | 2 | Read specific humidity calculated from dew point |
| `SWPART` | 1 | Partition total shortwave using solar geometry |
| `ZOFFST` | 1 | Treat reanalysis reference heights as offsets above canopy top, as instructed by FSM2's user guide |
| `PROFNC` | 0 | Use text output, which includes canopy snow mass |
| `CANMOD` | 1 | One canopy layer for the first baseline |
| `CANINT` | 1 | Upstream default linear interception |
| `CANUNL` | 1 | Upstream default time/melt-dependent unloading |
| `CANRAD` | 1 | Upstream default canopy radiation scheme |

Other physics options remain as in upstream `compil.sh`. This is an explicit baseline, not a recommendation that those options are optimal for every forest.

**Do not substitute `compil_nc.sh`.** In this version, native NetCDF output is limited to one point and its output list does not include the total canopy snow mass diagnostic. Python's final NetCDF contains it because we extract the text state output instead.

Also, do not feed the repository's original Alptal weather file directly into this modified executable: that example uses the other forcing-column format. For an upstream demonstration, use an independent source checkout with its unchanged `compil.sh` and `./FSM2 < nlst_Alptal.txt`.

## 5. Inspect your prepared files first

Run this in a notebook on the system with your data. Replace the one path with a file that exists:

```python
from pathlib import Path
import xarray as xr

path = Path('/scratch/s1803229/fsm2/S01/2018_07/driving_data.nc')
with xr.open_dataset(path) as ds:
    print(ds)
    for name in ['t2m', 'd2m', 'sf', 'tp', 'ssrd', 'strd', 'u10', 'v10', 'sp', 'VAI', 'vegh']:
        print(name, ds[name].dims, ds[name].attrs)
    print('Latitude rows:', ds.sizes['latitude'])
    print('Longitude columns:', ds.sizes['longitude'])
    print('VAI range:', float(ds.VAI.min()), float(ds.VAI.max()))
    print('Height range:', float(ds.vegh.min()), float(ds.vegh.max()))
```

The location above is an example based on your scratch naming, not a verified path. Change it to the actual location. `valid_time` or `time` is accepted by the runner; latitude and longitude must use those full names.

Your attachment creates the following variables or retains them from ERA5-Land:

| Prepared variable | Expected stored units | Model use |
|---|---|---|
| `t2m` | K | Air temperature; no Celsius conversion for input |
| `d2m` | K | Calculate vapour pressure and specific humidity |
| `sf` | m water equivalent, accumulated | De-accumulate, then convert to snowfall rate |
| `tp` | m water equivalent, accumulated | De-accumulate, subtract snowfall to obtain rain |
| `ssrd` | J m⁻², accumulated | De-accumulate, divide by 3600 to obtain W m⁻² |
| `strd` | J m⁻², accumulated | Same conversion for longwave |
| `u10`, `v10` | m s⁻¹ | Wind speed `sqrt(u10² + v10²)` |
| `w10` | m s⁻¹ | Already calculated by your script; the runner recomputes from the components |
| `sp` | Pa | Surface air pressure |
| `VAI` | m² m⁻², dimensionless ratio | Vegetation area per ground area |
| `vegh` | m | Canopy height |

The preparation script checks that vegetation layers are static and two-dimensional, but **does not check or convert their physical units**. Confirm the source height is in metres and VAI is a physical area index, not a percentage, scaled integer, canopy-cover fraction or unconverted LAI product. LAI and VAI are related but not identical because woody vegetation contributes to VAI.

The runner requires a consistent grid and identical static vegetation across months. It refuses duplicate, missing or out-of-order hours. Do not fill missing forcing with zero: that would create artificial dry periods or impossible temperatures.

This baseline runs cells with finite `VAI > 0` and finite `vegh > 2 m`. The 2 m restriction avoids using the default 2 m canopy-base parameter for shorter vegetation. Nonforest, missing-vegetation and shorter-canopy cells are excluded and remain **NaN** in the final output. A short-vegetation configuration would need a deliberate parameter change. This is not a WorldCover mask; incorporate that separately if you need a specific forest definition.

## 6. The essential weather conversions

### 6.1 ERA5-Land accumulations

Your script requests `reanalysis-era5-land` and preserves its source values. For its documented cumulative convention, each forecast starts at 00 UTC; the value at 01 UTC contains one hour, at 02 UTC contains two hours, and the next 00 UTC contains the preceding day's 24-hour accumulation.

For each accumulated variable `A`, the hourly amount ending at time `t` is:

- at **01 UTC**: `A(t)`;
- at all other hours, including **00 UTC**: `A(t) - A(t - 1 hour)`.

This subtraction must cross month boundaries. The runner joins the chronological series first, so July's 00 UTC first record can use June's 23 UTC last record when June is available. It only drops the first record of the entire simulation if that record needs an unavailable preceding accumulation. It does not drop the first hour of every month.

Use `--accumulation cumulative` for files retaining this documented convention. If inspection establishes that your particular files have already been de-accumulated, use `--accumulation hourly` instead, including in the Slurm file. Both modes expect precipitation amounts in metres and radiation energy in J m⁻², not rates or fluxes. Do not choose based solely on the units: cumulative and hourly amounts can have the same units.

Inspect a rainy or snowy day at a valid cell. Within each forecast cycle, cumulative precipitation should build through the day, then restart at 01 UTC. Confirm that hourly increments sum back to that day's 00 UTC total. Significant negative increments are a reason to stop and investigate, not to clip them away. The runner rejects these; only small numerical negatives are clipped.

### 6.2 Units and partitioning

For hourly increments:

```text
snowfall rate = hourly sf × 1000 / 3600          [kg m⁻² s⁻¹]
rainfall rate = (hourly tp − hourly sf) × 1000 / 3600
shortwave    = hourly ssrd / 3600               [W m⁻²]
longwave     = hourly strd / 3600               [W m⁻²]
```

One metre water equivalent is 1000 kg m⁻². No snow-density conversion is involved. The snowfall field already supplies the precipitation phase, so the runner does not repartition all precipitation using an arbitrary temperature threshold.

### 6.3 Humidity and heights

The runner uses a conventional over-water saturation-vapour-pressure approximation on dew point, in Celsius:

```text
e = 611.2 × exp(17.67 × Td_C / (Td_C + 243.5))    [Pa]
q = 0.622 × e / (sp − 0.378 × e)                 [kg kg⁻¹]
```

This is a transparent approximation, not an exact implementation of the full IFS saturation formula. Dew point calls for an over-water relation, including in subfreezing air. The corresponding RH column is supplied for format completeness, but the selected FSM2 reader uses `q` directly. This avoids a mismatch with FSM2's alternative internal RH-to-humidity conversion.

The namelist uses `zT=2`, `zU=10` with **`ZOFFST=1`**. These are reanalysis reference-height offsets above the canopy in the model's convention, not observations measured inside the model forest. This convention does not make ERA5-Land's weather perfectly representative of any individual stand.

### 6.4 Timing

An increment labelled 13:00 represents the interval 12:00–13:00 UTC. The runner sends 12:30 to FSM2 for solar geometry and labels the resulting state 13:00 in the output. Temperature, dew point, pressure and wind use the available instantaneous values at 13:00; treating these as representative of the hour is a stated approximation.

Longitude uses east-positive degrees, and solar noon is set to `12 − longitude/15` in UTC hours. No daylight-saving or local-time conversion is made. For western Canada, solar noon is therefore later than 12 UTC. Keeping this convention is important for canopy radiation and matching ICESat-2 observation times.

## 7. Understand the model input and canopy output

The runner writes one temporary text forcing file per cell. With `DRIV1D=2`, its columns are exactly:

```text
year month day hour SW LW Rf Sf Ta Qa RH Ua Ps
```

Notice that **rain comes before snow** in this format. The other FSM input format has a different order. There is no header row.

A namelist is simply a settings file with sections beginning `&name` and ending `/`. The runner generates all seven required sections, in this order:

```text
&params
/
&gridpnts
 Npnts=1
/
&gridlevs
/
&drive
 met_file='met.txt', dt=3600, lat=55, noon=18.6666667, zT=2, zU=10
/
&veg
 VAI=3, vegh=15, alb0=0.2
/
&initial
 start_file='none'
/
&outputs
 runid='', dump_file='dump.txt'
/
```

Those vegetation and position values are illustrations. The runner substitutes each cell's own values. Empty sections retain model defaults.

In terminal terms, executing one model run is `./FSM2 < namelist.txt`: `<` sends the settings file to the program. The Python runner does the same without requiring you to manage thousands of files manually.

In `stat.txt`, for **one point** and the default four soil layers and one canopy layer:

| Column, counting from 1 | Meaning |
|---:|---|
| 1–4 | Year, month, day, hour |
| 5 | Ground snow depth, m |
| 6 | Ground snow mass, kg m⁻² |
| **7** | **Total snow mass on canopy, kg m⁻²** |
| 8–11 | Soil temperatures, K |
| 12 | Surface temperature, K |
| 13 | Canopy temperature, K |

The source calls the total-canopy diagnostic **`svg`**. Internally, **`Sveg`** holds individual canopy-layer masses. The diagnostic totals them; with one layer they coincide for forest cells. The runner extracts zero-based Python column `6` and calls it `canopy_snow_mass`.

It is a **mass storage**, not a snowfall rate or canopy snow-cover fraction. Numerically, 1 kg m⁻² is 1 mm water equivalent. Do not divide it by 3600. Do not multiply by VAI again. It is expressed per model ground area, not per square metre of foliage. A grid cell represented as a homogeneous canopy is also not automatically a fractional forest/open-land mosaic; do not apply a canopy-cover weighting without defining that representation.

## 8. First test: one forest cell

From the project folder, set absolute input and output paths. These examples must be adjusted to your real paths:

```bash
export FSM2_PROJECT="$PWD"
export FSM2_INPUT=/scratch/s1803229/fsm2/S01
export FSM2_OUTPUT=/scratch/s1803229/FSM2_results/S01_baseline_cells
```

Run one eligible forest cell through the complete input period:

```bash
python run_grid.py \
  --input "$FSM2_INPUT" \
  --output "$FSM2_OUTPUT" \
  --accumulation cumulative \
  --spinup-cycles 2 \
  --limit 1
```

Use an interactive compute allocation for this test if required by your cluster's policy. The script chooses the first eligible forest cell in grid-index order. It prints a saved filename, such as `cell_y0000_x0004.nc`. `y` and `x` are zero-based latitude and longitude indices; row 0 is whatever latitude occurs first in your files, not necessarily the southernmost row.

Default spin-up repeats the first complete 12 calendar months twice, transferring the full dump state between runs. It then starts the actual simulation using that state. These are repeated-forcing spin-up cycles, not real observations from earlier years. Their output is discarded.

At least 12 months are required for this option. For a short software-only test, `--spinup-cycles 0` is permitted. Put that test in a separate output directory and do not treat it as a spin-up-checked scientific result.

For an earlier January-to-December forcing series, no manual change is needed. For July-to-June, the repeated cycle is July-to-June. You should compare, for example, 2 and 4 spin-up cycles at representative cells in **separate output folders**, and verify that the canopy-snow series and relevant soil/snow states are sufficiently stable for your purpose. Two cycles are a starter choice, not a demonstrated convergence criterion.

To inspect the first cell in a notebook:

```python
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt

folder = Path('/scratch/s1803229/FSM2_results/S01_baseline_cells')
path = sorted(folder.glob('cell_y*_x*.nc'))[0]
with xr.open_dataset(path) as ds:
    print(ds)
    print(ds[['canopy_snow_mass', 'ground_snow_mass']].to_array().min().item())
    print('Canopy maximum:', float(ds.canopy_snow_mass.max()), 'kg m-2')
    ds.canopy_snow_mass.sel(time=slice('2018-10-01', '2019-05-31')).plot()
    plt.ylabel('Canopy snow mass (kg m$^{-2}$)')
    plt.title('FSM2 canopy snow mass')
    plt.show()
```

Adjust the plotting dates to your actual period. Expect event-driven loading and subsequent losses, not a monotonically increasing store. Inspect a winter snowfall period, melt periods, summer values and month boundaries. Some snowfall may pass through or unload quickly, so a snow event need not produce a large persistent canopy mass.

## 9. Run all cells

For a small region, the sequential command is:

```bash
python run_grid.py \
  --input "$FSM2_INPUT" \
  --output "$FSM2_OUTPUT" \
  --accumulation cumulative \
  --spinup-cycles 2
```

It runs one forest cell at a time. It does not load the entire multiyear domain into memory. Original file chunks are retained, and cell data are loaded with a synchronous Dask scheduler to bound concurrency. Compressed spatial input chunks may still have to be read repeatedly; this prioritises a straightforward, auditable first implementation rather than maximum I/O performance.

The model runs the full time series in one invocation after spin-up. Monthly forcing files are storage partitions, not separate model initializations. **Never reset canopy snow, ground snow or soil state at the start of each month.**

A completed cell is written atomically and can be skipped on a repeated command. Output metadata record an executable and wrapper fingerprint, input paths/sizes/modification times, settings and source commit. If these change, use a new output directory; the script rejects a mismatched existing result.

Successful temporary text files are deleted. Failed cells retain a `work_y...` folder with forcing, namelist and `model.log` for diagnosis. A killed cell is rerun from the beginning on the next attempt; this is completed-cell recovery, not a timestep checkpoint system. Do not launch overlapping jobs for the same cells in the same output directory.

## 10. Run the grid with Slurm

The supplied array script assigns **one latitude row to each task**, with cells within that row run sequentially. Each task uses one CPU. Merely requesting more CPUs will not make this unmodified model multithreaded.

First, edit `run_fsm2.slurm` to load the same compiler and Python environment used in your successful test. Add your account and partition if required. Your previous jobs used `geos_extra`; use that only if it remains appropriate on this cluster. The supplied 8 GB memory and 24-hour wall time are starting requests, not measured requirements for your region.

Obtain the latitude count:

```bash
python - <<'PY'
import os
from pathlib import Path
import xarray as xr
file = sorted(Path(os.environ['FSM2_INPUT']).glob('????_??/driving_data.nc'))[0]
with xr.open_dataset(file) as ds:
    print('Nlat =', ds.sizes['latitude'])
    print('Last array index =', ds.sizes['latitude'] - 1)
PY
```

If this prints `Nlat = 42`, for example, submit indices **0–41**:

```bash
mkdir -p slurm
sbatch --array=0-41%4 run_fsm2.slurm
```

Replace 41 with your actual last index. `%4` permits at most four simultaneous jobs. Start with low concurrency to avoid overwhelming shared NetCDF storage, then increase only after measuring performance and confirming cluster limits. The default line in the supplied Slurm file runs just row 0; the command-line array option overrides it.

The three exported `FSM2_*` variables from section 8 must still be set in the submission shell. Slurm normally exports the submission environment; activate your environment explicitly in the script if your cluster requires it. Create the log directory **before** submitting because Slurm opens log files before running the script.

Monitor and inspect logs:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS
```

Replace `JOB_ID` with the number returned by `sbatch`. Look inside `slurm/fsm2-JOBID_TASKID.out` and `.err` for failed cells. A row containing no eligible forest cells legitimately finishes without outputs.

On timeout or interruption, resubmit the same row range with the same settings. Completed cells are reused. Before launching every region, time a representative row and check `MaxRSS`. If a row exceeds the wall limit, use a longer permitted time limit or divide work further; the current starter's row task is not internally parallel.

For a different region, change `FSM2_INPUT` and `FSM2_OUTPUT` and recalculate the array bounds. Grid dimensions need not be identical between regions.

## 11. Assemble monthly gridded NetCDFs

Wait until all row jobs finish successfully. Then run this on a compute node or in an appropriately resourced batch job:

```bash
python assemble_grid.py \
  --template "$FSM2_INPUT/2018_07/driving_data.nc" \
  --cells "$FSM2_OUTPUT" \
  --output /scratch/s1803229/FSM2_results/S01_baseline_grid
```

Change the template month to one that exists. This uses its coordinates and vegetation, verifies that all eligible cells are present, and checks that their times and run signatures agree. It refuses to hide failed cells by treating them as nonforest.

Outputs follow your familiar monthly organisation:

`/scratch/s1803229/FSM2_results/S01_baseline_grid/YYYY_MM/fsm2_output.nc`

There is one time-by-latitude-by-longitude array per model variable. Excluded cells are NaN. The `simulated_forest_mask` distinguishes simulated forest from excluded cells. The first simulation month can contain one fewer record if its first accumulation could not be reconstructed. Do not pad that missing hour with zero.

Assembly allocates one month's three output arrays at a time and refuses arrays exceeding 512 MiB in total. That is suitable for many regional domains, not a guarantee for a global domain. Larger domains need tiled assembly. It also refuses existing monthly outputs; if an assembly was interrupted, choose a new empty assembly folder or deliberately remove only its incomplete derived output tree before restarting. Per-cell simulations do not need to be repeated.

Approximate uncompressed storage for three float32 variables is:

`12 × number_of_hours × number_of_grid_cells bytes`.

For 70,000 hours and 2,000 cells, that is about 1.68 GB (1.56 GiB), before compression and metadata. Temporary text files and intermediate per-cell results add overhead. Results are compressed, but do not assume a particular compression ratio.

## 12. Open a map and extract a series

Notebook example:

```python
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt

root = Path('/scratch/s1803229/FSM2_results/S01_baseline_grid')
files = sorted(root.glob('????_??/fsm2_output.nc'))
ds = xr.open_mfdataset(files, combine='by_coords', chunks={'time': 24})
print(ds)

# Pick an hour that exists in your simulation.
field = ds.canopy_snow_mass.sel(time='2019-02-01T12:00:00', method='nearest')
field.plot(x='longitude', y='latitude', cbar_kwargs={'label': 'Canopy snow mass (kg m$^{-2}$)'})
plt.title(str(field.time.values))
plt.show()

# Substitute the location of interest; nearest does not guarantee it is a simulated forest cell.
series = ds.canopy_snow_mass.sel(latitude=63.0, longitude=-123.0, method='nearest')
print('Selected coordinate:', float(series.latitude), float(series.longitude))
series.sel(time=slice('2018-10-01', '2019-05-31')).plot()
plt.show()
# When finished:
# ds.close()
```

Check the selected coordinate lies inside the domain: unrestricted nearest selection can choose an edge cell for an outside point. Check its forest mask too. For ICESat-2, retain hourly output and match UTC observation times explicitly. Record whether you use the closest model state or interpolate between states; a daily average can miss brief canopy loading.

A positive canopy mass is evidence of modelled canopy snow, but converting it into your observational canopy-snow class needs a tested threshold. The model does not directly output your three-class JointSnow labels or a remotely sensed canopy snow fraction.

## 13. Choices still needed for scientific use

Your inputs are sufficient to drive a baseline with model defaults. They do not eliminate the following choices:

| Choice | Starter treatment | What to check for the research |
|---|---|---|
| Soil properties | Default clay/sand fractions and model parameters | Soil data or sensitivity to defaults |
| Initial conditions | Default initialization, followed by repeated first-year spin-up | Convergence; preferably earlier forcing if available |
| Vegetation | Static VAI and canopy height | Product units, woodland meaning, seasonal/deciduous behaviour |
| Ground albedo | Constant `alb0=0.2` | Sensitivity or spatial data |
| Forest representation | One homogeneous column per eligible ERA5 cell | Forest/open mosaics and consistency of area basis |
| Canopy interception/unloading | `CANINT=1`, `CANUNL=1` | Sensitivity to alternatives and parameters |
| Radiation | Total SW partitioned by model; reanalysis LW | No terrain shading or stand-specific radiation correction |
| Reference heights | Reanalysis offset convention | How well coarse weather represents the forest |

For canopy mass, prioritise VAI, interception capacity, interception/unloading options, wind, snowfall and sublimation-related humidity. The default capacity parameter `svai` is 4.4 kg m⁻² per unit VAI. `svai × VAI` provides a useful baseline capacity scale; do not mistake that parameterisation for a measured capacity at every site.

Keep all settings fixed for the first successful run. Then make separate named experiments for sensitivity tests. The starter explicitly checks one-layer output layout; changing `CANMOD` to two layers requires updating that validation and output handling before running the same wrapper. Changing `CANINT` or `CANUNL` is a compile-time change: edit the generated build configuration deliberately, rebuild, and keep a separate output folder. Re-running `build_fsm2.sh` recreates its baseline configuration.

No calibration is performed by these scripts. Getting plausible output and getting a scientifically validated canopy-snow product are different milestones.

## 14. Checks before accepting a run

1. **Forcing:** confirm units, hourly timestamps and accumulation convention. For several wet days, verify hourly precipitation totals against raw daily accumulation. Check radiation is in W m⁻² after conversion.
2. **Coverage:** all expected months are present; vegetation interpolation has not produced large gaps or unrealistic values.
3. **One-cell behaviour:** all values finite; canopy and ground mass nonnegative; snowfall, loading and loss events have credible timing.
4. **Month boundaries:** no unexplained reset on the first of each month. Inspect a snow-covered winter boundary explicitly.
5. **Spatial variation:** different cells receive their own meteorology, VAI and height. Check two contrasting cells, not just a visually smooth map.
6. **Spin-up:** compare a longer spin-up at representative cells; pay attention to first-season states and persistent ground snow.
7. **Completion:** every intended forest cell has a successful output. Assembly enforces completeness for the starter mask.
8. **Research comparison:** inspect timing against independent observations before choosing canopy-mass thresholds or interpreting bias.

## 15. Troubleshooting

| Symptom | Likely cause / next step |
|---|---|
| `gfortran: command not found` | Load/install a compiler, then rebuild |
| `No YYYY_MM/driving_data.nc files` | Input path is wrong or points one level too high/low |
| Missing variable or unexpected dimensions | Existing file differs from the attached preparation; inspect rather than rename blindly |
| Time must be gap-free | Missing month/hour, duplicate time, or inconsistent time dimension |
| Static vegetation differs across months | Different preparation inputs/grids; reconcile before modelling |
| Negative accumulation increment | Wrong accumulation mode, corrupt data, or an incorrect reset convention |
| Snowfall exceeds total precipitation | Inspect both increments and units; do not silently discard the discrepancy |
| Spin-up requires 12 months | Provide a complete cycle; zero cycles is only a short software test |
| Configuration/input changed | Use a fresh experiment folder or restore the original settings |
| All cells skipped | Check VAI/height units and the `VAI>0`, `height>2` mask |
| Unexpected model output or NaNs | Read retained `model.log`, namelist and forcing; check geometry, humidity and fluxes |
| Job ends at wall time | Resubmit to reuse completed cells; benchmark rows and adjust resources |
| Missing cells during assembly | Inspect failed/unfinished row jobs before proceeding |
| Zero canopy snow all winter | Check positive VAI, snowfall phase/order, temperature, and the actual output column |
| Huge snowfall or radiation | Accumulations may have been mistaken for hourly amounts or rates |

## 16. Sources and provenance

The model-specific instructions above were checked against these exact upstream files at the pinned revision. The source code is authoritative where the user-guide tables contain typographical errors (including a canopy-state unit typo and restart-order inconsistencies). Let the model write and read its own restart files rather than constructing them manually.

- [FSM2 repository](https://github.com/RichardEssery/FSM2)
- [Pinned user guide source](https://github.com/RichardEssery/FSM2/blob/a76945e75b36628161f4679428ed72dce090b368/docs/FSMuser.tex)
- [Main program: inputs, state, shared meteorology and restart](https://github.com/RichardEssery/FSM2/blob/a76945e75b36628161f4679428ed72dce090b368/src/FSM2.F90)
- [Driving-data reader and humidity handling](https://github.com/RichardEssery/FSM2/blob/a76945e75b36628161f4679428ed72dce090b368/src/FSM2_DRIVE.F90)
- [Text output: exact column order and units](https://github.com/RichardEssery/FSM2/blob/a76945e75b36628161f4679428ed72dce090b368/src/FSM2_OUTPUT.F90)
- [Canopy snow mass balance](https://github.com/RichardEssery/FSM2/blob/a76945e75b36628161f4679428ed72dce090b368/src/INTERCEPT.F90)
- [Essery et al. (2025), model description](https://doi.org/10.5194/gmd-18-3583-2025)
- [ECMWF ERA5-Land documentation: accumulations](https://confluence.ecmwf.int/spaces/CKB/pages/140385202/ERA5-Land+data+documentation)
- [ECMWF ERA5 documentation: near-surface humidity](https://confluence.ecmwf.int/pages/viewpage.action?pageId=185075432)
- Your supplied preparation script, inspected on 10 November 2026.

The Python and shell scripts in this package are a companion workflow prepared for your folder layout; they are not an official FSM2 distribution. The build retrieves the official source under its own licence. See `VALIDATION.md` for the actual test coverage.
