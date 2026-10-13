#!/bin/bash
#SBATCH --job-name=climate-site-selection
#SBATCH --output=./slurm/%x_%j.out
#SBATCH --error=./slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

set -euo pipefail

mkdir -p ./slurm

python 11-tl-copernicus-site-selection.py --area 73 -180 45 180 \
  --start-year 2019 --end-year 2021 --keep-tcc # --months 10 4
