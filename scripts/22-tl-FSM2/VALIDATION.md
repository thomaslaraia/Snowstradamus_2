# Validation performed

Tested on 10 November 2026 against upstream FSM2 commit `a76945e75b36628161f4679428ed72dce090b368`.

## Passed here

- Compiled the supplied configuration using GNU Fortran 14.2.0.
- Checked Python syntax and ran the actual compiled model through the Python wrapper.
- Constructed monthly synthetic ERA5-Land-style input files for a 2 × 2 grid spanning 1 January 2001 to 2 January 2002. Two cells represented forests with different snowfall; two were excluded by the vegetation mask.
- Ran two repeated first-year spin-up cycles, carrying the complete model dump state between cycles, then ran the full series for each eligible cell.
- Verified 8,784 output hours after the initial unavailable accumulation was dropped.
- Assembled 13 monthly gridded NetCDF files, with excluded cells retained as NaN.
- Verified finite, nonnegative forest canopy mass and different output for the two differently forced cells. Maximum canopy masses were about 4.161 and 6.022 kg m⁻² for this artificial forcing. These values are software-test results, not observations or predictions for S01.
- Checked unique, ordered output timestamps and alignment of model midpoint timestamps with interval-end NetCDF timestamps.
- Re-ran the same command and confirmed both completed cells were recognised and skipped.
- Checked de-accumulation through 23:00 → 00:00 → 01:00 → 02:00 at a month boundary, including the 01:00 reset.
- Confirmed significant negative accumulation increments are rejected.
- Confirmed cumulative forcing and equivalent pre-de-accumulated hourly forcing produce identical converted forcing matrices.
- Checked precipitation conversion from metres to kg m⁻² s⁻¹ and radiation conversion from J m⁻² to W m⁻².

## Not established by these tests

- The contents, units, accumulation convention or completeness of your real NetCDF files: only the preparation script was supplied.
- Scientific calibration or validation of canopy snow mass at your sites.
- Adequacy of two spin-up cycles or of default vegetation/soil parameters.
- Runtime and memory for your real grid, filesystem and cluster.
- Execution of the example Slurm submission on your account; module, account and partition configuration remain local setup steps.
- Two-layer canopy support in the supplied wrapper. It deliberately requires the tested one-layer configuration.

The guide describes the checks to perform before accepting a production run. No real S01 model results are included in this package.
