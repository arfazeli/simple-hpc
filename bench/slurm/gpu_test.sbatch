#!/bin/bash
#SBATCH --account=def-CHANGEME
#SBATCH --job-name=simple-gpu-test
#SBATCH --gpus=mi300a:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=bench/slurm/logs/%x-%j.out
# T23: build the GPU backend, run the CPU-vs-GPU tests, device STREAM, then Ghia on the GPU.
# Submit from the repo root:
#   AMD MI300A (HIP):   sbatch bench/slurm/gpu_test.sbatch
#   NVIDIA H100 (CUDA): sbatch --gpus=h100:1 --export=ALL,VENDOR=cuda bench/slurm/gpu_test.sbatch
set -uo pipefail # no -e: later steps still run and report if one fails
source bench/slurm/env.sh
VENDOR=${VENDOR:-hip}
B=build-$VENDOR
[ "$VENDOR" = cuda ] && module load cuda
mkdir -p bench/slurm/logs bench/results
export TMPDIR=${SLURM_TMPDIR:-/tmp} OMP_NUM_THREADS=8
step() { echo; echo "== $*"; }

step node
hostname; g++ --version | head -n 1
if [ "$VENDOR" = cuda ]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nvcc --version | tail -n 1
else
    rocm-smi --showproductname 2>/dev/null | grep -m1 'Card Series'; hipcc --version | head -n 1
fi

step configure
CATCH=""
for d in "$HOME/scratch/simple-hpc/build-cpu/_deps/catch2-src" build-cpu/_deps/catch2-src; do
    [ -d "$d" ] && CATCH="-DFETCHCONTENT_SOURCE_DIR_CATCH2=$d" && break # reuse; no download
done
if [ "$VENDOR" = cuda ]; then
    GPU_FLAGS="-DSIMPLE_CUDA=ON"
else
    GPU_FLAGS="-DSIMPLE_HIP=ON -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_COMPILER=$(hipconfig -l)/clang++"
fi
cmake -S . -B $B -DCMAKE_BUILD_TYPE=Release $GPU_FLAGS \
    -DPython3_EXECUTABLE="$(command -v python3)" $CATCH 2>&1 | tail -n 12

step build
if ! cmake --build $B -j 8 > $B.log 2>&1; then
    grep -m 1 -B 2 -A 25 -E 'error' $B.log
    echo "BUILD FAILED"; exit 1
fi
grep -cE 'warning' $B.log | sed 's/^/warnings: /'

step "GPU tests (CPU vs GPU)"
$B/simple_hip_tests 2>&1 | tail -n 60

step "CPU fast tests in this build"
ctest --test-dir $B -LE 'ghia|hip' 2>&1 | grep -E 'tests passed|Failed|\*\*\*'

step "device STREAM (1 GiB arrays)"
if [ "$VENDOR" = cuda ]; then
    nvcc -O3 -arch=sm_90 -x cu -DSIMPLE_CUDA -Iinclude bench/stream_hip.hip -o $B/stream_gpu
else
    hipcc -O3 --offload-arch=gfx942 -Iinclude bench/stream_hip.hip -o $B/stream_gpu
fi
$B/stream_gpu

# Ghia on the GPU: copy a validation config and replace keys (duplicate keys are an error).
run_case() { # name base_cfg backend
    local name=$1 cfg=$B/$1.cfg out=$B/runs/$1
    cp "$2" "$cfg"
    sed -i -e "s|^backend *=.*|backend = $3|" \
           -e "s|^momentum_solver *=.*|momentum_solver = gs_multicolour|" \
           -e "s|^pressure_solver *=.*|pressure_solver = gs_redblack|" \
           -e "s|^output_dir *=.*|output_dir = $out|" "$cfg"
    $B/simple "$cfg" | grep -E '^gpu:|converged'
    python3 postprocess/validate_ghia.py "$out" --check --max-rms-u 0.025 --max-rms-v 0.015 | tail -n 4
}
step "Ghia Re1000 80^2, GPU";  run_case gpu_re1000_80  configs/validation/re1000_80.cfg  gpu
step "Ghia Re1000 128^2, GPU"; run_case gpu_re1000_128 configs/validation/re1000_128.cfg gpu
step "Same 128^2 case on the 8 CPU cores of this node, for comparison"
run_case cpu_re1000_128 configs/validation/re1000_128.cfg openmp

step "Rough GPU time per outer iteration (fixed work)"
python3 bench/run_bench.py --exe $B/simple --grids 512 2048 --iters 20 --repeats 1 \
    --threads 8 --tag "gpu_${VENDOR}_${SLURM_JOB_ID}" --set backend=gpu \
    --set momentum_solver=gs_multicolour --set pressure_solver=gs_redblack 2>&1 | tail -n 6
step done
