#!/bin/bash
#SBATCH --account=def-CHANGEME
#SBATCH --job-name=simple-gpu-test
#SBATCH --gpus=mi300a:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=bench/slurm/logs/%x-%j.out
# T23: build the HIP backend on an MI300A node, run the CPU-vs-GPU tests, then Ghia on the GPU.
# Submit from the repo root:  sbatch bench/slurm/gpu_test.sbatch
set -uo pipefail # no -e: later steps still run and report if one fails
source bench/slurm/env.sh
mkdir -p bench/slurm/logs bench/results
export TMPDIR=${SLURM_TMPDIR:-/tmp} OMP_NUM_THREADS=8
step() { echo; echo "== $*"; }

step node
hostname; rocm-smi --showproductname 2>/dev/null | grep -m1 'Card Series'; hipcc --version | head -n 1
g++ --version | head -n 1

step configure
CATCH=""
for d in "$HOME/scratch/simple-hpc/build-cpu/_deps/catch2-src" build-cpu/_deps/catch2-src; do
    [ -d "$d" ] && CATCH="-DFETCHCONTENT_SOURCE_DIR_CATCH2=$d" && break # reuse; no download
done
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DSIMPLE_HIP=ON \
    -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_COMPILER="$(hipconfig -l)/clang++" \
    -DPython3_EXECUTABLE="$(command -v python3)" $CATCH 2>&1 | tail -n 12

step build
if ! cmake --build build-hip -j 8 > build-hip.log 2>&1; then
    grep -m 1 -B 2 -A 25 -E 'error' build-hip.log
    echo "BUILD FAILED"; exit 1
fi
grep -cE 'warning' build-hip.log | sed 's/^/warnings: /'

step "HIP tests (CPU vs GPU)"
build-hip/simple_hip_tests 2>&1 | tail -n 60

step "CPU fast tests in this build"
ctest --test-dir build-hip -LE 'ghia|hip' 2>&1 | grep -E 'tests passed|Failed|\*\*\*'

# Ghia on the GPU: copy a validation config and replace keys (duplicate keys are an error).
run_case() { # name base_cfg backend
    local name=$1 cfg=build-hip/$1.cfg out=build-hip/runs/$1
    cp "$2" "$cfg"
    sed -i -e "s|^backend *=.*|backend = $3|" \
           -e "s|^momentum_solver *=.*|momentum_solver = gs_multicolour|" \
           -e "s|^pressure_solver *=.*|pressure_solver = gs_redblack|" \
           -e "s|^output_dir *=.*|output_dir = $out|" "$cfg"
    build-hip/simple "$cfg" | grep -E '^gpu:|converged'
    python3 postprocess/validate_ghia.py "$out" --check --max-rms-u 0.025 --max-rms-v 0.015 | tail -n 4
}
step "Ghia Re1000 80^2, GPU";  run_case gpu_re1000_80  configs/validation/re1000_80.cfg  gpu
step "Ghia Re1000 128^2, GPU"; run_case gpu_re1000_128 configs/validation/re1000_128.cfg gpu
step "Same 128^2 case on the 8 CPU cores of this node, for comparison"
run_case cpu_re1000_128 configs/validation/re1000_128.cfg openmp

step "Rough GPU time per outer iteration (fixed work)"
python3 bench/run_bench.py --exe build-hip/simple --grids 512 2048 --iters 20 --repeats 1 \
    --threads 8 --tag "gpu_t23_${SLURM_JOB_ID}" --set backend=gpu \
    --set momentum_solver=gs_multicolour --set pressure_solver=gs_redblack 2>&1 | tail -n 6
step done
