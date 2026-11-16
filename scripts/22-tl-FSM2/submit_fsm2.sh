#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

region=${1:?Usage: bash submit_fsm2.sh S01}
input="/scratch/s1803229/fsm2/$region"
output="/scratch/s1803229/fsm2_results/$region"
mkdir -p slurm

last_row=$("$HOME/venvs/fsm2/bin/python" - "$input" <<'PY'
from pathlib import Path
import sys
import xarray as xr

folder = Path(sys.argv[1])
files = sorted(folder.glob("????_??/driving_data.nc"))
if not files:
    sys.exit(f"No monthly driving_data.nc files found in {folder}")

with xr.open_dataset(files[0]) as ds:
    print(ds.sizes["latitude"] - 1)
PY
)

echo "Submitting $region: rows 0–$last_row, up to 8 running at once"
sbatch --array="0-${last_row}%8" run_fsm2.slurm "$input" "$output"
