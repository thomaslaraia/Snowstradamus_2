# FSM2 with named settings and Slurm row jobs

Extract at the root of Snowstradamus_2. The runner imports scripts/fsm2_flux.py from that repository. Use a shared filesystem and a Python 3.11+ environment with numpy, pandas, xarray, dask and netCDF4. The initial build needs git, tar, bash, gfortran and network access to GitHub. Slurm submission also uses flock and sbatch.

Copy template.toml to any name. It contains every upstream process switch, numeric parameter overrides and the spin-up count. Omitted parameters use upstream defaults. VAI, canopy height, coordinates and hourly weather come from the input site. The grid adapter fixes DRIV1D=2, SWPART=1, ZOFFST=1, PROFNC=0 to match that input and output format. It uses a one-hour timestep, 2 m temperature, 10 m wind and alb0=0.2.

    cp template.toml my_model.toml
    bash submit.sh my_model.toml /scratch/s1803229/fsm2_flux/S01 /scratch/s1803229/fsm2_results baseline_S01 4

Call submit.sh with bash from the login node. It builds once before submitting an array. Each task handles one contiguous slice of latitude rows, skipping cells with missing canopy properties, VAI <= 0 or canopy height <= 2 m. N is capped at the number of rows. Slurm controls when tasks start. The output directory is OUTPUT_PARENT/RUN_NAME. It contains the settings snapshot model.toml, Slurm logs and one cell_yNNNN_xNNNN.nc per eligible cell, with snow depth, ground snow mass and canopy snow mass. For a retry, wait for the previous array to finish and reuse the same arguments. Keep each run name specific to one site and configuration.

For one local test use:

    python run.py --config my_model.toml --input /path/to/fsm2_flux/S01 --output /path/to/test --limit 1

The default spin-up repeats the first year twice. For a short test set spinup_cycles = 0. The first run obtains RichardEssery/FSM2 at a76945e75b36628161f4679428ed72dce090b368. FSM2_source contains the upstream checkout. The wrapper builds with the upstream compil.sh in a temporary directory. The shared .build directory retains FSM2 and OPTS.h per distinct physics/custom combination. Parameter and spin-up edits use the same binary. The wrapper translates [params] into native &params input. Upstream declares the accepted names in src/FSM2_PARAMS.F90. The stock exponential timescale eunl is 864000 s and isn't in that namelist.

## Custom equations

In [custom], uncomment loading or unloading in the template and replace its right-hand side. These are trusted single-line Fortran expressions compiled into the corresponding dsvg assignment in src/INTERCEPT.F90. Supply the intercepted or unloaded mass for this layer and timestep in kg/m². Sveg(n) is current layer snow mass, Scap(n) is capacity, Sfg is snowfall rate reaching the layer, dt is seconds, Ta and Tveg(n) are kelvin, and Ua is wind speed. Return a physically valid amount, including the available snowfall limit for loading. FSM2 still applies its subsequent upstream capacity and unloading caps. Omit either expression to use the corresponding upstream switch. Follow Fortran free-form line limits (the generated line begins with four spaces and 'dsvg = '). These hooks cover loading and unloading only. Changing the canopy capacity law or adding new state variables requires a Fortran source change. The template describes stock FSM2 physics. Naming it hp98.toml does not implement the published HP98 model.
