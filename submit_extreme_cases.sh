#!/bin/bash
# Master script to submit IPCSP Extreme Cases Benchmark experiments to Barkla (Slurm).
# Submits one independent Slurm job per (structure, p) pair under runs/extreme_cases_benchmark/p{P}/{STRUCT}
# with p + 2 CPUs allocated and no per-IP Gurobi time limit.
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

PARTITION="${SLURM_PARTITION:-lowpriority,nodes}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
RUN_NAME="${RUN_NAME:-extreme_cases_benchmark}"

P_VALUES=(1 2 4 8 16 32 64)
STRUCTURES=(
    "Ca3Al2Si3O12_P1"
    "SrTiO3_3x3x3_P1"
    "Y2Ti2O7_P1"
    "Y2O3_P1"
    "MgAl2O4_P1"
    "SrTiO3_2x2x2_P1"
    "AuCu_8x8x8"
    "AuCu_16x16x16"
)

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
        --structures|--tests)
            shift
            STRUCTURES=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                STRUCTURES+=("$1")
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

echo "================================================================================"
echo " Submitting IPCSP Extreme Cases Benchmark across:"
echo "   Structures (${#STRUCTURES[@]}): ${STRUCTURES[*]}"
echo "   Processors (${#P_VALUES[@]}): p = (${P_VALUES[*]})"
echo "   Partition: ${PARTITION} | Allocation per job: p + 2 CPUs | Walltime: ${TIME_LIMIT}"
echo "   Run Directory: runs/${RUN_NAME}"
echo "================================================================================"

mkdir -p "${PROJECT_DIR}/runs/${RUN_NAME}"

cat > "${PROJECT_DIR}/runs/${RUN_NAME}/structures_manifest.csv" << 'EOF'
Test ID,Compound Formula,Structural Prototype,Unit Cell Size (Nions / Z),Cell Parameter (a_A),Grid Resolution (g / Unique Sites n),Combinatorial Space (Omega)
Ca3Al2Si3O12_P1,Ca3Al2Si3O12,Garnet (Cubic),160 ions (Z=8),11.9,g=8 (n=512 in P1),10^210
SrTiO3_3x3x3_P1,SrTiO3 (Supercell),Perovskite (3x3x3),135 ions (Z=27),11.7,g=8 (n=512 in P1),10^180
Y2Ti2O7_P1,Y2Ti2O7,Pyrochlore (Cubic),88 ions (Z=8),10.2,g=8 (n=512 in P1),10^133
Y2O3_P1,Y2O3,Bixbyite (Cubic),80 ions (Z=16),10.7,g=8 (n=512 in P1),10^97
MgAl2O4_P1,MgAl2O4,Spinel (Cubic),56 ions (Z=8),8.2,g=8 (n=512 in P1),10^97
SrTiO3_2x2x2_P1,SrTiO3 (Supercell),Perovskite (2x2x2),40 ions (Z=8),7.8,g=8 (n=512 in P1),10^74
AuCu_8x8x8,Au0.5Cu0.5,FCC L10 Alloy Supercell (8x8x8),2048 atoms (Z=512 FCC cells),30.4,8^3 FCC cells (n=2048),2^2048
AuCu_16x16x16,Au0.5Cu0.5,FCC L10 Alloy Supercell (16x16x16),16384 atoms (Z=4096 FCC cells),60.8,16^3 FCC cells (n=16384),2^16384
EOF

SUBMITTED_JOBS=()
SKIPPED_JOBS=0

for P in "${P_VALUES[@]}"; do
    CPUS=$((P + 2))
    for STRUCT in "${STRUCTURES[@]}"; do
        OUT_DIR="${PROJECT_DIR}/runs/${RUN_NAME}/p${P}/${STRUCT}"
        mkdir -p "${OUT_DIR}"

        # Skip if summary.csv already exists and has results (> 50 bytes)
        if [[ -f "${OUT_DIR}/summary.csv" ]] && [[ $(stat -c%s "${OUT_DIR}/summary.csv" 2>/dev/null || echo 0) -gt 50 ]]; then
            echo "  Skipping (${STRUCT}, p=$(printf "%2d" $P)): summary.csv already complete"
            SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
            continue
        fi

        # Skip if a job with this exact output dir is currently running or pending
        ACTIVE_JOB=$(squeue -u "$USER" -h -o "%i %o" | grep "${OUT_DIR}" | awk '{print $1}' || true)
        if [[ -n "$1" && "$1" == "--force" ]] || [[ -z "${ACTIVE_JOB}" ]]; then
            # Check by job name + output file existence in squeue
            ACTIVE_BY_LOG=""
            for jid in $(squeue -u "$USER" -h -o "%i"); do
                if [[ -f "${OUT_DIR}/slurm-${jid}.out" ]]; then
                    ACTIVE_BY_LOG="${jid}"
                    break
                fi
            done
            if [[ -n "${ACTIVE_BY_LOG}" ]]; then
                echo "  Skipping (${STRUCT}, p=$(printf "%2d" $P)): Job ${ACTIVE_BY_LOG} is already active"
                SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
                continue
            fi
        fi

        # Clean up previous failed slurm logs before resubmitting
        rm -f "${OUT_DIR}"/slurm-*.out "${OUT_DIR}"/slurm-*.err

        JOB_ID=$(sbatch \
            --partition="${PARTITION}" \
            --nodes=1 \
            --ntasks=1 \
            --job-name="ext_${STRUCT}_p${P}" \
            --cpus-per-task="${CPUS}" \
            --time="${TIME_LIMIT}" \
            --output="${OUT_DIR}/slurm-%j.out" \
            --error="${OUT_DIR}/slurm-%j.err" \
            --export=ALL,P="${P}",CPUS="${CPUS}",OUTPUT_DIR="${OUT_DIR}" \
            submit_decomposition.slurm --tests "${STRUCT}" "${EXTRA_ARGS[@]}" | awk '{print $NF}')

        SUBMITTED_JOBS+=("${JOB_ID}")
        echo "  Submitted (${STRUCT}, p=$(printf "%2d" $P)): Job ID ${JOB_ID} (${CPUS} CPUs) -> ${OUT_DIR}"
    done
done

echo ""
echo "All ${#SUBMITTED_JOBS[@]} jobs successfully submitted to partition '${PARTITION}'!"
echo "Job IDs: ${SUBMITTED_JOBS[*]}"
echo ""
echo "Monitor status with:   squeue -u \$USER"
echo "Merge CSV outputs:     uv run python merge_decomposition_results.py --runs-dir runs/${RUN_NAME} --output-dir results/${RUN_NAME}"
echo ""
squeue -u "$USER"
