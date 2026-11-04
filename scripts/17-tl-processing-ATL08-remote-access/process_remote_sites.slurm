#!/bin/bash
#SBATCH --job-name=remote-sites
#SBATCH --output=../slurm/remote-sites_%j.out
#SBATCH --error=../slurm/remote-sites_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --account=geos_extra
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Activate your snow_env before sbatch; the job inherits its Python environment.
# Submit from the directory containing these three Python/Slurm files.
SITES = ../../../scratch/climate_site_selection/regional_site_selection/candidates_d8bbbaa5ea8a7a28/selection_e5c0ae68a48ece02/n09/selected_sites.csv
OUTPUT = ../../../scratch/ATL08_parquets
CONFIG = ../../../config.yaml

python -u process_remote_sites.py "$SITES" "$OUTPUT" --config "$CONFIG" --length 30 --box-side-km 1