#!/usr/bin/env bash
#SBATCH --job-name=fsm2-assemble
#SBATCH --account=geos_extra
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm/fsm2-assemble-%j.out
#SBATCH --error=slurm/fsm2-assemble-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

region=${1:?Usage: sbatch assemble_grid.slurm S01}
input="/scratch/s1803229/fsm2_flux/$region"
results="/scratch/s1803229/fsm2_flux_results"

shopt -s nullglob
templates=("$input"/????_??/driving_data.nc)
if (( ${#templates[@]} == 0 )); then
    echo "No monthly driving_data.nc files under $input" >&2
    exit 1
fi

echo "Assembling $region into $results/$region.nc"

exec "$HOME/venvs/fsm2/bin/python" -u assemble_grid.py \
    --template "${templates[0]}" \
    --cells "$results/$region" \
    --output "$results"
