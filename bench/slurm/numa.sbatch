#!/bin/bash
#SBATCH --account=def-CHANGEME
#SBATCH --job-name=simple-numa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=bench/slurm/logs/%x-%j.out
# T37: does NUMA page placement cap scaling at large grids? Same job, same cores:
#   old  = build-cpu/simple (serial first touch; the T36 binary)
#   ilv  = old binary under numactl --interleave=all (placement fix without code change)
#   new  = build-cpu-ft/simple (parallel first touch in Field2D)
# plus perf counters for local vs remote DRAM loads at 4096^2, 32 threads.
# Submit from the repo root (needs build-cpu/ from setup.sh):  sbatch bench/slurm/numa.sbatch
set -uo pipefail
source bench/slurm/env.sh
export OMP_PLACES=cores OMP_PROC_BIND=close TMPDIR=${SLURM_TMPDIR:-/tmp}
TAG=numa_${SLURM_JOB_ID}
SOLVERS="--set momentum_solver=gs_multicolour --set pressure_solver=mg"
step() { echo; echo "== $*"; }

step node
hostname; lscpu | grep -E 'Model name|NUMA node[0-9]'; numactl --show | grep -E 'physcpubind|nodebind'

step "build new code (first touch) into build-cpu-ft"
cmake -S . -B build-cpu-ft -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(command -v python3)" \
    -DFETCHCONTENT_SOURCE_DIR_CATCH2="$PWD/build-cpu/_deps/catch2-src" 2>&1 | tail -n 2
cmake --build build-cpu-ft -j 16 > build-cpu-ft.log 2>&1 || { grep -m1 -A20 error build-cpu-ft.log; exit 1; }
ctest --test-dir build-cpu-ft -LE ghia 2>&1 | grep -E 'tests passed|Failed'

printf '#!/bin/sh\nexec numactl --interleave=all %s/build-cpu/simple "$@"\n' "$PWD" > build-cpu-ft/old_interleave.sh
chmod +x build-cpu-ft/old_interleave.sh

step "STREAM on these cores"
OMP_NUM_THREADS=32 build-cpu/stream_triad $((1 << 27)) | tee bench/results/${TAG}_stream.csv

bench() { # variant exe grid iters threads...
    local v=$1 exe=$2 g=$3 it=$4; shift 4
    step "$v ${g}^2"
    python3 bench/run_bench.py --exe "$exe" --grids "$g" --iters "$it" --repeats 3 \
        --threads "$@" --tag "${TAG}_${v}_g${g}" $SOLVERS 2>&1 | grep median
}
T="1 8 16 24 32"
bench old build-cpu/simple                 4096 6 $T
bench ilv build-cpu-ft/old_interleave.sh   4096 6 $T
bench new build-cpu-ft/simple              4096 6 $T
bench old build-cpu/simple                 2048 25 $T
bench new build-cpu-ft/simple              2048 25 $T

step "perf: DRAM loads, local vs remote node (4096^2, 32 threads, 6 iterations)"
python3 - > "$TMPDIR/perf.cfg" <<EOF
import sys; sys.path.insert(0, "bench"); import run_bench as b
print(b.make_config(4096, 6, "$TMPDIR/perf_out",
      {"momentum_solver": "gs_multicolour", "pressure_solver": "mg"}), end="")
EOF
EV=mem_load_l3_miss_retired.local_dram:u,mem_load_l3_miss_retired.remote_dram:u,node-loads:u,node-load-misses:u
if command -v perf > /dev/null; then
    for v in old new; do
        exe=build-cpu/simple; [ $v = new ] && exe=build-cpu-ft/simple
        echo "-- $v"
        OMP_NUM_THREADS=32 perf stat -x, -e "$EV" -o "$TMPDIR/perf_$v.txt" "$exe" "$TMPDIR/perf.cfg" > /dev/null ||
            OMP_NUM_THREADS=32 perf stat -x, -e node-loads:u,node-load-misses:u \
                -o "$TMPDIR/perf_$v.txt" "$exe" "$TMPDIR/perf.cfg" > /dev/null ||
            echo "perf stat failed (events not permitted?)"
        grep -v '^#' "$TMPDIR/perf_$v.txt" | grep -v "^$" | sed "s/^/$v,/" | tee -a bench/results/${TAG}_perf.csv
    done
else
    echo "perf not available on this node"
fi
step done
