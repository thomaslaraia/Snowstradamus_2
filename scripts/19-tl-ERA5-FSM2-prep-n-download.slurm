#!/bin/bash
#SBATCH --job-name=19-era5-fsm2
#SBATCH --time=4-00:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=./slurm/19-era5-fsm2-%j.out
#SBATCH --error=./slurm/19-era5-fsm2-%j.err
#SBATCH --account=geos_extra
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail
# Submit from the directory containing both scripts, with your Python environment active.
# Create ./slurm BEFORE sbatch. Relative data paths resolve from the submission directory.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SITES_CSV="../../scratch/climate_site_selection/regional_site_selection/candidates_d8bbbaa5ea8a7a28/selection_e5c0ae68a48ece02/n09/selected_sites.csv"  # Empty for manual extent; otherwise path to selected_sites.csv.
CANDIDATES='[0]'  # '[0]', '[0,1,2]', or 'None' for every CSV row.
NAME="ontario"  # Manual-mode output folder name.
AREA=(53 -95 48 -87)  # North west south east; used in manual mode.
START_MONTH="2018-07"
END_MONTH="2026-06"
ERA5_ROOT="../../scratch/ERA5"
FSM2_ROOT="../../scratch/fsm2"
VAI_PATH="../../scratch/LAI/VAI_summer_climatology_2018_2020_global_0p1deg.nc"
VEGH_PATH="../../scratch/canopy_height/609802/ETH_GlobalCanopyHeight_2020_0p1deg_vegh.nc"

region_args=(--name "$NAME" --area "${AREA[@]}")
if [[ -n "$SITES_CSV" ]]; then
    region_args=(--sites-csv "$SITES_CSV" --candidates "$CANDIDATES")
fi

requeue_on_timeout() {
    trap '' USR1
    echo "[$(date -Is)] Wall time approaching. Stopping Python and requeueing job ${SLURM_JOB_ID}."

    kill -TERM "$python_pid" 2>/dev/null || true
    wait "$python_pid" || true

    if scontrol requeue "$SLURM_JOB_ID"; then
        exit 0
    fi

    echo "Requeue failed for job ${SLURM_JOB_ID}. Resubmit with sbatch to resume." >&2
    exit 1
}

SECONDS=0
time_limit_signal_received=0
trap 'time_limit_signal_received=1' USR1

python -u 19-tl-ERA5-FSM2-prep-n-download.py \
    "${region_args[@]}" \
    --start-month "$START_MONTH" --end-month "$END_MONTH" \
    --era5-root "$ERA5_ROOT" --fsm2-root "$FSM2_ROOT" \
    --vai-path "$VAI_PATH" --vegh-path "$VEGH_PATH"
python_pid=$!

status=0
wait "$python_pid" || status=$?
echo "Elapsed this attempt: ${SECONDS}s"

if (( time_limit_signal_received && status != 0 )); then
    requeue_on_timeout
fi

exit "$status"
