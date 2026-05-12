#!/bin/bash
#SBATCH --job-name=era5_land_download
#SBATCH --output=./slurm/%x_%j.out
#SBATCH --error=./slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

set -euo pipefail

mkdir -p ./slurm

source ~/.bashrc
conda activate snow_win

python download_era5_land.py \
  --out-dir ../../scratch/ERA5