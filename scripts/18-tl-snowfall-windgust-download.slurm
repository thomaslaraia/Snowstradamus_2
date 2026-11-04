#!/bin/bash
#SBATCH --job-name=era5-daily
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=./slurm/era5-daily-%j.out
#SBATCH --error=./slurm/era5-daily-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --account=geos_extra

set -euo pipefail

OUTPUT_DIR=../../scratch/era5_land_daily
START_YEAR=2018
END_YEAR=2023

requeue_job() {
    trap '' USR1
    echo "Wall time approaching. Requeuing job ${SLURM_JOB_ID}."
    kill -TERM "${python_pid}" 2>/dev/null || true
    wait "${python_pid}" || true
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}

trap requeue_job USR1

time python -u 18-tl-snowfall-windgust-download.py \
    --start-year "${START_YEAR}" \
    --end-year "${END_YEAR}" \
    --area 73 -180 45 180 \
    --output "${OUTPUT_DIR}" &

python_pid=$!
wait "${python_pid}"
