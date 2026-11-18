# ERA5-Land → FSM2 ensemble → ICESat-2 comparison

Use your FSM2 Python environment, with gfortran available.

1. From the repository root, prepare forcing. Check paths in the Slurm file
   and set CANDIDATES='None' for all ten sites:

   ```bash
   cd scripts
   mkdir -p slurm
   sbatch 19-tl-ERA5-FSM2-prep-n-download.slurm
   ```

2. Once preparation finishes, obtain/build the pinned FSM2 source once:

   ```bash
   cd 22-tl-FSM2
   bash build_fsm2.sh
   ```

3. Build and submit the ensemble:

   ```bash
   FSM2_FACTORS='0.8 1 1.2' bash submit_ensemble.sh \
       /scratch/s1803229/fsm2_flux/S01 \
       /scratch/s1803229/fsm2_ensemble 4
   ```

   Repeat for S02–S10. Assembly runs automatically after successful model jobs.

   This creates 27 members: three structures × three capacity multipliers ×
   three unloading-rate multipliers. Omit FSM2_FACTORS for three baseline
   members. Four jobs can run concurrently per submission. Account: geos_extra.

   Outputs: <experiment>/<member>/S01.nc.
   The submission script prints the experiment directory.

4. After assembly, compare each member:

   ```bash
   python compare_icesat2.py \
       --observations /scratch/s1803229/ATL08_parquets/S01_grouped.parquet \
       --lookup /path/to/probabilities.nc \
       --model /path/to/experiment/clm5_cap1_loss1/S01.nc \
       --output /path/to/comparison.parquet
   ```

   Repeat for other members/sites. All three probabilities, model masses
   and classes are retained. No assimilation or parameter fitting occurs.

## Model choices

- These are CLM5, Noah-MP and HP98 loading/unloading adaptations inside FSM2.
  They share FSM2 energy balance, melt and sublimation; they are not full
  implementations of CLM5/Noah-MP or exact reproductions of Harvey's runs.
- CLM5: constant capacity 6 × VAI; temperature/wind unloading.
- Noah-MP: temperature-dependent capacity; snow density capped at 120 kg/m³;
  temperature/wind unloading.
- HP98: temperature-dependent capacity without that density cap;
  exponential unloading with a 10-day baseline timescale.
- Noah-MP/HP98 use capacity coefficient 5.87 and dense-stand interception.
  VAI represents one-sided leaf/stem area.
- Capacity factors multiply svai. Unloading factors multiply unloading
  rates; HP98 timescales change inversely. Melt and sublimation are unchanged.
  These are exploratory perturbations, not calibrated uncertainty ranges.
- Additional runner arguments follow concurrency, for example:
  ... 4 --accumulation cumulative
  Default forcing mode is flux.

## Comparison choices

- Class 0: ground mass ≤ --ground-threshold, default 0 kg/m²,
  regardless of canopy mass.
- Otherwise class 2 requires canopy mass >0.5 kg/m²; class 1 is ground only.
  Raw masses remain unchanged. The ground threshold is configurable,
  not a validated snow-cover detector.
- Match to the nearest forcing cell and nearest UTC model hour.
- Supply observation_time for exact observation timestamps. Otherwise,
  granule start time is recovered from resource; this is approximate.
- Interpolate all three probabilities from Eg_strong/Ev_strong.
  Out-of-range radiometry or unavailable model output remains unmatched.
- Retain all 1 km groups, with total comparison weight 1 per cell/overpass.
  This avoids extra influence from cells containing more groups; it does
  not model correlated errors for assimilation.

## Validation

All three baseline adaptations compiled and completed 72-hour synthetic runs.
Comparison checks covered classes, threshold boundaries, repeated groups,
missing coordinates, out-of-domain observations, probability retention and
Parquet writing. Full ERA5 runs and actual Slurm scheduling remain to be
tested on the cluster.

## References

- Harvey et al.: https://doi.org/10.1029/2023WR036996
- CLM5: https://escomp.github.io/CTSM/release-clm5.0/tech_note/Hydrology/CLM50_Tech_Note_Hydrology.html
- Noah-MP: https://ncar.github.io/ccpp-physics/V8.0.0/module__sf__noahmplsm_8F90_source.html
