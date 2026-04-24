#!/bin/bash
#SBATCH --job-name=process-atl08-df
#SBATCH --output=./slurm/%x_%j.out
#SBATCH --error=./slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch 04-tl-process-atl08-to-df.slurm <folder> <output.pkl>"
    echo "Example: sbatch 04-tl-process-atl08-to-df.slurm ontario_test ontario_test.pkl"
    exit 1
fi

FOLDER="$1"
OUTPUT="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running on host: $(hostname)"
echo "Started at: $(date)"
echo "Script dir: $SCRIPT_DIR"
echo "Input folder: $FOLDER"
echo "Output pickle: $OUTPUT"

# Uncomment/edit these if needed on your cluster
# module load python
# source ~/.bashrc
# conda activate your_env_name

python "$SCRIPT_DIR/04-tl-process-atl08-to-df.py" "$FOLDER" "$OUTPUT"

echo "Finished at: $(date)"