#!/bin/bash
#SBATCH --job-name=remote-sites
#SBATCH --output=remote-sites_%j.out
#SBATCH --error=remote-sites_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Activate your snow_env before sbatch; the job inherits its Python environment.
# Submit from the directory containing these three Python/Slurm files.
python -u process_remote_sites.py "$@"
