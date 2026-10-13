#!/bin/bash
#SBATCH --job-name=climate-site-selection
#SBATCH --output=./slurm/%x_%j.out
#SBATCH --error=./slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

set -euo pipefail

# Create ./slurm BEFORE submitting this script:
# mkdir -p ./slurm
# sbatch climate_site_selection_requeue.sh
#
# Python must support resuming from saved progress and recovering partial
# writes. Requeuing terminates the current run; it does not save Python's
# in-memory state or make interrupted NetCDF writes safe.

restart_job() {
    echo "Approaching the time limit; requeuing job ${SLURM_JOB_ID}."
    if scontrol requeue "${SLURM_JOB_ID}"; then
        exit 0
    else
        echo "Requeue failed; stopping this job." >&2
        exit 1
    fi
}

trap restart_job USR1

echo "Starting job ${SLURM_JOB_ID}; restart count: ${SLURM_RESTART_COUNT:-0}"

# Background execution and wait allow Bash to handle the signal promptly.
python -u 11-tl-copernicus-site-selection.py --area 73 -180 45 180 \
    --start-year 2019 --end-year 2021 --keep-tcc &

wait "$!"

