#!/bin/bash
# Master script to submit IPCSP decomposition parallel experiments to Slurm.
# For each parameter p, allocates p + 2 CPUs and submits an independent Slurm job.
# Follows the architecture of MC-EMMA's submit_yba2ca2fe5o13_nodes.sh.
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

PARTITION="${SLURM_PARTITION:-lowpriority}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
RUN_NAME="${RUN_NAME:-table1_benchmark}"

# P values representing number of processors for parallelisation
# Default scaling experiments: p = 1, 2, 4, 8, 16, 32, 64
P_VALUES=(1 2 4 8 16 32 64)

# Parse custom options if provided
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --p-values|--p)
            shift
            P_VALUES=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                P_VALUES+=("$1")
                shift
            done
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "${P_LIST}" ]]; then
    read -r -a P_VALUES <<< "${P_LIST}"
fi

echo "================================================================================"
echo " Submitting IPCSP Decomposition Experiments across p = (${P_VALUES[*]})"
echo " Partition: ${PARTITION} | Allocation per job: p + 2 CPUs | Time: ${TIME_LIMIT}"
echo " Run Directory: runs/${RUN_NAME}"
echo "================================================================================"

# Ensure base run directory exists at repo root
mkdir -p "${PROJECT_DIR}/runs/${RUN_NAME}"

SUBMITTED_JOBS=()

for P in "${P_VALUES[@]}"; do
    CPUS=$((P + 2))
    OUT_DIR="${PROJECT_DIR}/runs/${RUN_NAME}/p${P}"
    mkdir -p "${OUT_DIR}"

    JOB_ID=$(sbatch \
        --partition="${PARTITION}" \
        --nodes=1 \
        --ntasks=1 \
        --job-name="ipcsp_p${P}" \
        --cpus-per-task="${CPUS}" \
        --time="${TIME_LIMIT}" \
        --output="${OUT_DIR}/slurm-%j.out" \
        --error="${OUT_DIR}/slurm-%j.err" \
        --export=ALL,P="${P}",CPUS="${CPUS}",OUTPUT_DIR="${OUT_DIR}" \
        submit_decomposition.slurm "${EXTRA_ARGS[@]}" | awk '{print $NF}')

    SUBMITTED_JOBS+=("${JOB_ID}")
    echo "  Submitted p = $(printf "%2d" $P): Job ID ${JOB_ID} (${CPUS} CPUs allocated [p+2]) -> ${OUT_DIR}"
done

echo ""
echo "All ${#P_VALUES[@]} jobs successfully submitted to partition '${PARTITION}'!"
echo "Job IDs: ${SUBMITTED_JOBS[*]}"
echo ""
echo "Monitor status with:   squeue -u \$USER"
echo "Merge CSV outputs:     uv run python merge_decomposition_results.py --runs-dir runs/${RUN_NAME} --output-dir results/${RUN_NAME}"
echo ""
squeue -u "$USER"
