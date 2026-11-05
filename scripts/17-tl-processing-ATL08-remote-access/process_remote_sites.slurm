#!/bin/bash
#SBATCH --job-name=17-atl08-remote
#SBATCH --output=../slurm/17-atl08-remote_%j.out
#SBATCH --error=../slurm/17-atl08-remote_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --account=geos_extra
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Activate your snow_env before sbatch; the job inherits its Python environment.
# Submit from the directory containing these three Python/Slurm files.
SITES=../../../scratch/climate_site_selection/regional_site_selection/candidates_d8bbbaa5ea8a7a28/selection_e5c0ae68a48ece02/n09/selected_sites.csv
OUTPUT=../../../scratch/ATL08_parquets
CONFIG=../../config.yaml

# Zero-based CSV rows. Use () for all candidates or e.g. (0 2 4).
CANDIDATE_INDICES=(0)
candidate_args=()
if (( ${#CANDIDATE_INDICES[@]} > 0 )); then
    candidate_args=(--site-indices "${CANDIDATE_INDICES[@]}")
fi

requeue_on_timeout() {
    trap '' USR1
    if ! kill -0 "$worker_pid" 2>/dev/null; then
        wait "$worker_pid"
        exit $?
    fi
    echo "$(date -Is) Time limit approaching. Stopping Python and requeueing job $SLURM_JOB_ID."
    kill -TERM "$worker_pid" 2>/dev/null || true
    wait "$worker_pid" || true
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}

python -u process_remote_sites.py "$SITES" "$OUTPUT" \
    --config "$CONFIG" --length 30 --box-side-km 1 "${candidate_args[@]}" &
worker_pid=$!
trap requeue_on_timeout USR1
wait "$worker_pid"
