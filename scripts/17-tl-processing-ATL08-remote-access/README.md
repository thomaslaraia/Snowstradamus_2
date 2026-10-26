# Remote site dataframe batch

Place all three files together in your project's scripts directory. `remote_batch_core.py` is a bundled copy of the existing remote pipeline with atmospheric screening moved ahead of photon retrieval; keep it beside the runner. Your other modules are not overwritten.

Activate snow_env, then submit from this directory:

```bash
sbatch process_remote_sites.slurm /absolute/path/selected_sites.csv /absolute/path/remote_site_output --config /absolute/path/config.yaml
```

Defaults: all sites, all available release-007 overpasses from 2018-10-01 to UTC submission time, 30 m bins, beam-mean msw_flag and layer_flag maxima of 1.0 (equality passes), ATL08 landcover (DW disabled). Other processing settings come from your config, including small_box_half_side, individual atmospheric trimming, saturation, landcover, altitude, minimum segment count, regression and outlier settings. Check those settings before your full run. The original atmospheric OR condition is retained.

Optional arguments: `--site-indices 0 1`, `--end 2020-01-01`, `--length 15`, `--msw-threshold 0.8`, `--layer-threshold 0.8`.

The early atmospheric test uses ATL08 parents within the actual site rectangle, excluding the query buffer. No photons are requested for rejected beams. Surviving beams are retrieved and rebinned once, then individual filters and the existing spatial-block regression run. Beam numbering in the results follows the existing orientation-aware spot mapping: odd strong, even weak. Full-site mean screening can exclude locally clear areas inside a large site.

## Resume and output

Logs print the output directory `<output>/<settings-hash>/`. `data/` holds one compressed Parquet file per site/acquisition; `status/` records completed results including empty and rejected results. `errors/` records failed attempts, which are retried on resubmission. Historical error entries may remain after success; status is authoritative. Missing ATL08 beam records leave the acquisition pending rather than permanently treating ingestion lag as bad quality. This conservative rule can retry permanently absent beams and delay saving that acquisition.

Re-submit the same command after the time limit. Completed acquisitions are skipped. An interrupted acquisition restarts; individual photon requests are not checkpointed. A newer product revision replaces the older acquisition partition. Changed code/config/site selection creates a separate settings-hash directory. The job does not automatically requeue. Do not edit the code while a job is running. Only one writer can run against a given settings directory.

Only result subsets and metadata are saved, not full ATL03/08 files. Memory use depends on the site and photon density; the 32 GB Slurm request is a starting allocation, not a guarantee. Compute nodes need internet access to the services.

## Read in notebook 05

```python
from pathlib import Path
import pandas as pd
DATASET = Path('/absolute/path/remote_site_output/<settings-hash>/data')
files = sorted(DATASET.glob('*.parquet'))
df = pd.read_parquet(DATASET) if files else pd.DataFrame()
```

`folder` is the site ID; `lat`/`lon` are regression-block centres, while `latitude`/`longitude` are segment locations. `file_index` is a stable numeric acquisition identifier, not a chronological row number. Retain `resource` when grouping overpasses to avoid combining two same-day acquisitions. Existing notebook 05 classification/model application is still required; this batch produces regression inputs/results, not snow classes or uncertainty estimates.

`ATL03` is the remote resource name. `ATL08` is nullable because the public database need not expose an exact source filename; `atl08_source`, parent identifiers and `resource` provide provenance without fabricating a file path. Canopy openness is inherited at ATL08 resolution (100 m), not recalculated at 30 m.

Matching gaps remain as discussed in the diagnostic notebook: only matched, quality-passing bins enter regression. This batch does not fix missing ATL08 coverage. It is restricted to the current pipeline's release 007, not an automatic switch to another release.

Dependencies: sliderule, icesat2db, geopandas, pyproj, numpy, pandas, scipy, scikit-learn, matplotlib, pyyaml, pyarrow. Use the environment in which your remote notebook works.

Validation: Python compilation and Slurm shell syntax checks; inspection of early-screening order. Live multi-year retrieval has not been executed here.
