#!/usr/bin/env bash
#SBATCH --job-name=fsm2-grid
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --account=geos_extra
#SBATCH --output=slurm/fsm2-%A_%a.out
#SBATCH --error=slurm/fsm2-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

input=$1
output=$2
row=$SLURM_ARRAY_TASK_ID

echo "Running latitude row $row for $input"

exec "$HOME/venvs/fsm2/bin/python" -u run_grid.py \
    --input "$input" \
    --output "$output" \
    --accumulation flux \
    --spinup-cycles 2 \
    --row-start "$row" \
    --row-stop "$((row + 1))"
