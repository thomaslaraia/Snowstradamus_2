#!/usr/bin/env bash
#SBATCH --job-name=19-era5-fsm2
#SBATCH --time=4-00:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=./slurm/19-era5-fsm2-%j.out
#SBATCH --error=./slurm/19-era5-fsm2-%j.err
#SBATCH --account=geos_extra
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --array=1-9

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SITES_CSV="../../scratch/climate_site_selection/regional_site_selection/candidates_nov16/selection/n10/selected_sites.csv"
CANDIDATES="[$SLURM_ARRAY_TASK_ID]"
NAME="ontario"
AREA=(53 -95 48 -87)
# Empty for manual extent; otherwise path to selected_sites.csv.
# '[0]', '[0,1,2]', or 'None' for every CSV row.
# Manual-mode output folder name.
# North west south east; used in manual mode.

START_MONTH="2018-07"
END_MONTH="2026-06"
ERA5_ROOT="../../scratch/ERA5"
FSM2_ROOT="../../scratch/fsm2_flux"

VAI_PATH="../../scratch/LAI/VAI_summer_climatology_2018_2020_global_0p1deg.nc"
VEGH_PATH="../../scratch/canopy_height/609802/ETH_GlobalCanopyHeight_2020_0p1deg_vegh.nc"

region_args=(--name "$NAME" --area "${AREA[@]}")
if [[ -n "$SITES_CSV" ]]; then
    region_args=(--sites-csv "$SITES_CSV" --candidates "$CANDIDATES")
fi

time_limit_signal=0
trap 'time_limit_signal=1' USR1

python -u 19-tl-ERA5-FSM2-prep-n-download.py \
    "${region_args[@]}" \
    --start-month "$START_MONTH" \
    --end-month "$END_MONTH" \
    --era5-root "$ERA5_ROOT" \
    --fsm2-root "$FSM2_ROOT" \
    --vai-path "$VAI_PATH" \
    --vegh-path "$VEGH_PATH" \
    --workers "$SLURM_CPUS_PER_TASK" &

python_pid=$!
status=0
wait "$python_pid" || status=$?

if (( time_limit_signal )); then
    trap '' USR1
    kill -TERM "$python_pid" 2>/dev/null || true
    wait "$python_pid" 2>/dev/null || true
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

exit "$status"
