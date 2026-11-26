#!/usr/bin/env bash
#SBATCH --account=geos_extra
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
set -euo pipefail
if [[ ${1:-} == --finish ]]; then
    project=$2 input=$3 dest=$4 py=$5 array=$6 tasks=$7
    exec 9>"$dest/.submit.lock"
    flock 9
    "$py" "$project/finish_region.py" "$input" "$dest"
    for ((i=0; i<tasks; i++)); do rm -f "$dest/logs/${array}_${i}.out"; done
    rm -f "$dest/logs/finish-${SLURM_JOB_ID}.out" "$dest/.active_jobs"
    rm -f "$dest/model.toml"
    rmdir "$dest/logs" 2>/dev/null || true
    exit 0
fi
if [[ ${1:-} == --worker ]]; then
    shift
    project=$1 input=$2 dest=$3 rows=$4 tasks=$5 py=$6
    i=$SLURM_ARRAY_TASK_ID
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    exec "$py" "$project/run.py" --config "$dest/model.toml" --input "$input" --output "$dest" \
        --row-start "$((i*rows/tasks))" --row-stop "$(((i+1)*rows/tasks))"
fi
[[ $# == 5 ]] || { echo 'Usage: bash submit.sh MODEL.toml INPUT_SITE OUTPUT_PARENT RUN_NAME N_JOBS' >&2; exit 2; }
project=$(cd "$(dirname "$0")" && pwd)
config=$(realpath "$1")
input=$(realpath "$2")
dest=$(realpath -m "$3/$4")
[[ $4 != */* && $4 != . && $4 != .. && -n $4 && $5 =~ ^[1-9][0-9]*$ ]] || exit 2
py=$(command -v python)
rows=$("$py" - "$input" <<'PY'
import sys
from pathlib import Path
import xarray as xr
files = sorted(Path(sys.argv[1]).glob("????_??/driving_data.nc"))
if not files:
    raise SystemExit("No monthly forcing files found")
with xr.open_dataset(files[0]) as ds:
    print(ds.sizes["latitude"])
PY
)
tasks=$5
(( tasks <= rows )) || tasks=$rows
mkdir -p "$dest/logs"
exec 9>"$dest/.submit.lock"
flock 9
if [[ -s "$dest/.active_jobs" ]]; then
    active=$(squeue -h -u "$USER" -o '%F' | sort -u)
    for id in $(cat "$dest/.active_jobs"); do
        if grep -Fxq "$id" <<< "$active"; then
            echo "Job $id still owns this run" >&2
            exit 2
        fi
    done
fi
if [[ -f "$dest/model.toml" ]]; then
    cmp "$config" "$dest/model.toml"
else
    cp "$config" "$dest/model.toml.tmp"
    mv "$dest/model.toml.tmp" "$dest/model.toml"
fi
cd "$project"
"$py" -c 'from model import prepare; from pathlib import Path; import sys; prepare(Path(sys.argv[1]))' "$dest/model.toml"
if [[ -f "$dest/$4.nc" ]]; then
    "$py" "$project/finish_region.py" "$input" "$dest"
    rm -f "$dest/model.toml" "$dest/.active_jobs"
    rmdir "$dest/logs" 2>/dev/null || true
    exit 0
fi
job=$(sbatch --parsable --array="0-$((tasks-1))" --job-name="$4" --output="$dest/logs/%A_%a.out" \
    "$project/submit.sh" --worker "$project" "$input" "$dest" "$rows" "$tasks" "$py")
job=${job%%;*}
echo "$job" > "$dest/.active_jobs"
finish=$(sbatch --parsable --dependency="afterok:$job" --kill-on-invalid-dep=yes --job-name="$4-merge" \
    --output="$dest/logs/finish-%j.out" "$project/submit.sh" --finish "$project" "$input" "$dest" "$py" "$job" "$tasks")
echo "${finish%%;*}" >> "$dest/.active_jobs"
echo "Submitted workers $job and regional merge ${finish%%;*}. Output: $dest/$4.nc"
