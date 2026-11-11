#!/bin/bash
#SBATCH --job-name=ipcsp_decomp
#SBATCH --partition=lowpriority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/decomposition_benchmark/slurm-%j.out
#SBATCH --error=runs/decomposition_benchmark/slurm-%j.err

# If executed directly outside Slurm (e.g. bash submit_decomposition.slurm),
# dispatch the master submission script across all p experiments.
if [ -z "${SLURM_JOB_ID}" ]; then
    echo "================================================================================"
    echo "  Direct execution detected outside Slurm."
    echo "  Forwarding to master submission script across all p experiments..."
    echo "================================================================================"
    exec bash ./submit_decomposition_experiments.sh "$@"
fi

# Load GULP from environment modules if available
module purge
module load gulp 2>/dev/null || true

# Ensure ~/.local/bin is in PATH for uv
export PATH="$HOME/.local/bin:$PATH"

# Change to project root directory
PROJECT_DIR="/users/sgsdoust/fastscratch/ipcsp"
cd "${PROJECT_DIR}"

# Read p and output directory from environment (passed via sbatch --export) or default
P_VAL="${P:-4}"
CPUS_ALLOC="${SLURM_CPUS_PER_TASK:-$((P_VAL + 2))}"
OUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs/decomposition_benchmark/p${P_VAL}}"

# Ensure output directory exists
mkdir -p "${OUT_DIR}"

echo "================================================================================"
echo "  Starting IPCSP Decomposition Benchmark Task: p=${P_VAL} | Host=$(hostname)"
echo "  Slurm Job ID: ${SLURM_JOB_ID} | CPUs: ${CPUS_ALLOC} (p+2 allocation)"
echo "  Partition: ${SLURM_JOB_PARTITION:-lowpriority} | Output Dir: ${OUT_DIR}"
echo "  Start Time: $(date)"
echo "================================================================================"

# Execute the parallel scaling experiment for p
uv run python -u benchmark_decomposition.py \
    --p "${P_VAL}" \
    --output-dir "${OUT_DIR}" \
    "$@"

EXIT_CODE=$?

echo "================================================================================"
echo "  Finished IPCSP Decomposition Benchmark Task | p=${P_VAL} | Exit Code: ${EXIT_CODE}"
echo "  End Time: $(date)"
echo "================================================================================"

exit ${EXIT_CODE}
