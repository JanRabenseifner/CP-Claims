#!/bin/bash
#SBATCH --partition=std
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --job-name=cp_claims
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -e
set -x

# HPC environment setup - adjust these to match your cluster
source /sw/batch/init.sh
module unload env
module load env/gcc-13.2.0_openmpi-4.1.6

# Activate conda environment - adjust to your setup
# source /path/to/anaconda3/bin/activate your_env_name

# Threading controls (single-thread for MPI)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=${TMPDIR:-/tmp}/matplotlib_${SLURM_JOB_ID}

# Project configuration
PROJECT_DIR=${PROJECT_DIR:-$PWD}
RESULTS_DIR=${RESULTS_DIR:-${PROJECT_DIR}/results/run_${SLURM_JOB_ID}}
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH}"

mkdir -p "$RESULTS_DIR"

cd "$PROJECT_DIR"

echo "=============================================="
echo "COUNT DATA CP SIMULATION"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Results dir: ${RESULTS_DIR}"
echo "Start time: $(date)"
echo "=============================================="

mpirun python scripts/run_simulation.py \
    --output-dir "$RESULTS_DIR" \
    --repetitions 96 \
    --seed 42 \
    --alpha 0.1 \
    --n-obs 1000,2000,4000,6000,10000 \
    --dim-x 10,20,50 \
    --dgp-types poisson,zip,negbin,hurdle \
    --correlation-strength 0.0,0.3,0.5 \
    --categorical-ratio 0.1,0.3,0.5 \
    --interaction-effects false,true \
    --spatial-effects false,true \
    --exposure-types realistic \
    --correlation-types toeplitz \
    --learners Poisson_GLM,LightGBM,RandomForest \
    --dgcp-methods hybrid,full,exact \
    --min-group-sizes 10,20,50,100,200

# Combine rank-level outputs into a single parquet file
python scripts/combine_results.py "$RESULTS_DIR" --output "$RESULTS_DIR/results_combined.parquet"

echo "=============================================="
echo "Simulation completed at: $(date)"
echo "=============================================="
