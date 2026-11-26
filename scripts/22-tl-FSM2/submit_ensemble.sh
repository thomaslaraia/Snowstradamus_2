#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --account=geos_extra
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
set -euo pipefail

if [[ ${1:-} == --row || ${1:-} == --assemble ]]; then
    mode=$1
    cd "$2"
    input=$3
    experiment=$4
    current=$(sha256sum build_members.py run_grid.py assemble_grid.py submit_ensemble.sh ../fsm2_flux.py | sha256sum | cut -c1-16)
    [[ $current == "$(cat "$experiment/scripts.sha256")" ]] || { echo "Scripts changed after submission" >&2; exit 2; }
    rows=$5
    py=$6
    shift 6
    mapfile -t members < "$experiment/members.txt"
    task=$SLURM_ARRAY_TASK_ID
    if [[ $mode == --row ]]; then
        member=${members[task / rows]}
        row=$((task % rows))
    else
        member=${members[task]}
    fi
    dir="$experiment/$member"
    cells="$dir/$(basename "$input")"
    mkdir -p "$cells"
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    echo "$(date -Is) $mode $member task=$task"
    if [[ $mode == --row ]]; then
        exec flock -n "$dir/row-$row.lock" "$py" -u run_grid.py \
            --input "$input" --output "$cells" --exe "$experiment/build/$member/FSM2" \
            --spinup-cycles 2 --row-start "$row" --row-stop "$((row + 1))" "$@"
    fi
    template=("$input"/????_??/driving_data.nc)
    if [[ ! -f "$cells.nc" ]]; then
        "$py" -u assemble_grid.py --template "${template[0]}" --cells "$cells" --output "$dir"
    fi
    exit 0
fi

project=$(cd "$(dirname "$0")" && pwd)
cd "$project"
input=$(realpath "${1:?Supply the prepared region folder, ending in S01}")
root=${2:?Supply the ensemble results root}
parallel=${3:-4}
[[ $parallel =~ ^[1-9][0-9]*$ ]] || { echo "Concurrency must be positive" >&2; exit 2; }
shift "$(( $# < 3 ? $# : 3 ))"
read -ra factors <<< "${FSM2_FACTORS:-1}"
py=$(command -v python)
region=$(basename "$input")
scripts_hash=$(sha256sum build_members.py run_grid.py assemble_grid.py submit_ensemble.sh ../fsm2_flux.py | sha256sum | cut -c1-16)
key=$(printf '%s\n' "$scripts_hash" "${factors[*]}" "$*" | sha256sum | cut -c1-16)
mkdir -p "$root/$region/$key/logs"
experiment=$(realpath "$root/$region/$key")
printf '%s\n' "$scripts_hash" > "$experiment/scripts.sha256"

help=$("$py" run_grid.py --help)
flags=()
if [[ $help == *--accumulation* ]]; then
    if [[ $help =~ --accumulation[[:space:]]+\{([^}]+)\} && ,${BASH_REMATCH[1]}, == *,flux,* ]]; then
        flags=(--accumulation flux)
    else
        echo "This runner advertises an accumulation option without a flux mode. Use the flux-input run_grid.py." >&2
        exit 2
    fi
fi
[[ $help == *--exe* ]] || { echo "The runner must accept --exe" >&2; exit 2; }

rows=$("$py" - "$input" <<'PY'
import sys
from pathlib import Path
import xarray as xr

files = sorted(Path(sys.argv[1]).glob("????_??/driving_data.nc"))
if not files:
    raise SystemExit("No monthly driving_data.nc found")
with xr.open_dataset(files[0]) as ds:
    print(ds.sizes["latitude"])
PY
)

flock "$experiment/build.lock" "$py" build_members.py "$project/FSM2_source" "$experiment/build" --factors "${factors[@]}"
members=("$experiment"/build/*/member.json)
count=${#members[@]}
list=$(mktemp "$experiment/members.XXXXXX")
for file in "${members[@]}"; do basename "$(dirname "$file")"; done > "$list"
mv "$list" "$experiment/members.txt"

script="$project/submit_ensemble.sh"
args=("$project" "$input" "$experiment" "$rows" "$py")
logs="$experiment/logs"
job=$(sbatch --parsable --job-name=fsm2-rows --array="0-$((rows*count-1))%$parallel" \
    --output="$logs/row-%A_%a.out" --error="$logs/row-%A_%a.err" \
    "$script" --row "${args[@]}" "${flags[@]}" "$@")
echo "Submitted $rows rows × $count configurations as array $job"

sbatch --job-name=fsm2-assemble --dependency="afterok:${job%%;*}" --kill-on-invalid-dep=yes \
    --array="0-$((count-1))%$parallel" --output="$logs/grid-%A_%a.out" \
    --error="$logs/grid-%A_%a.err" "$script" --assemble "${args[@]}"
echo "Output under $experiment"
