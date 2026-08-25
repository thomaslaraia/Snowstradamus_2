#!/bin/bash
#SBATCH --job-name=era5_land_download
#SBATCH --output=./slurm/%x_%j.out
#SBATCH --error=./slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail

mkdir -p ./slurm

python 07-tl-copernicus-era5-download.py \
  --out-dir ../../data_store/ERA5
