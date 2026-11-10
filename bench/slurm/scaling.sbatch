#!/bin/bash
#SBATCH --account=def-CHANGEME
#SBATCH --job-name=simple-scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=bench/slurm/logs/%x-%j.out
# Strong scaling of the OpenMP solver on 32 cores, plus STREAM on the same cores as the ceiling.
# Submit from the repo root after setup.sh:  sbatch bench/slurm/scaling.sbatch
# ponytail: shared (bycore) node, so neighbours can steal bandwidth; repeats + same-job STREAM
# make it visible. Use --exclusive later if the numbers are too noisy.
set -euo pipefail
source bench/slurm/env.sh

export OMP_PLACES=cores OMP_PROC_BIND=close
export TMPDIR=${SLURM_TMPDIR:-/tmp}  # run_bench writes each run's fields here (~GBs at 4096²)
THREADS="1 2 4 8 16 24 32"
SOLVERS="--set momentum_solver=gs_multicolour --set pressure_solver=mg"
TAG=cluster_${SLURM_JOB_ID}

echo "== node"; hostname; lscpu | grep -E 'Model name|Socket|Core|NUMA|L2|L3'
numactl --show; module -t list 2>&1 | grep -E 'gcc|python'

echo "== STREAM (3 x 1 GiB arrays, well above any L3)"
OMP_NUM_THREADS=32 build-cpu/stream_triad $((1 << 27)) | tee bench/results/${TAG}_stream.csv

# Fixed work per grid: fewer outer iterations on bigger grids keeps each 1-thread run ~1 min.
for g_it in 512:400 1024:100 2048:25 4096:6; do
    g=${g_it%:*}; it=${g_it#*:}
    echo "== grid ${g}^2, ${it} iterations"
    python3 bench/run_bench.py --exe build-cpu/simple --grids "$g" --iters "$it" --repeats 3 \
        --threads $THREADS --tag "${TAG}_g${g}" $SOLVERS
done
echo "== done"
