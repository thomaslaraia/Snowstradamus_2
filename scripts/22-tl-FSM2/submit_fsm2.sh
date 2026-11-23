#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

region=${1:?Usage: bash submit_fsm2.sh S01}
input="/scratch/s1803229/fsm2_flux/$region"
output="/scratch/s1803229/fsm2_flux_results/$region"
max_jobs=6

mkdir -p slurm

last_row=$("$HOME/venvs/fsm2/bin/python" - "$input" <<'PY'
from pathlib import Path
import sys
import xarray as xr

root = Path(sys.argv[1])
files = sorted(root.glob("????_??/driving_data.nc"))
if not files:
    raise SystemExit(f"No monthly driving_data.nc files under {root}")

with xr.open_dataset(files[0]) as ds:
    if ds.attrs.get("forcing_format") != "hourly_mean_flux_v1":
        raise SystemExit(f"Unexpected forcing format in {files[0]}")
    print(ds.sizes["latitude"] - 1)
PY
)

echo "Submitting $region: rows 0–$last_row, up to $max_jobs running at once"
sbatch --array="0-${last_row}%${max_jobs}" run_fsm2.slurm "$input" "$output"
