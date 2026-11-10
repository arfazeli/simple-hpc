#!/bin/bash
#SBATCH --account=def-CHANGEME
#SBATCH --job-name=simple-mg-numa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=bench/slurm/logs/%x-%j.out
# T38: first touch + descending row split for the multigrid arrays. Same job, same cores:
#   ft = build-cpu-ft/simple (T37: Field2D first touch only)
#   mg = build-cpu-mg/simple (T38: also MGWork arrays and multigrid loop split)
# plus perf with this CPU's own local/remote DRAM events (names discovered from perf list).
# Submit from the CPU folder's repo root (needs build-cpu/ and build-cpu-ft/):
#   sbatch bench/slurm/mg_numa.sbatch
set -uo pipefail
source bench/slurm/env.sh
export OMP_PLACES=cores OMP_PROC_BIND=close TMPDIR=${SLURM_TMPDIR:-/tmp}
TAG=mgnuma_${SLURM_JOB_ID}
SOLVERS="--set momentum_solver=gs_multicolour --set pressure_solver=mg"
step() { echo; echo "== $*"; }

step node
hostname; numactl --show | grep -E 'physcpubind|nodebind'

step "build T38 into build-cpu-mg"
cmake -S . -B build-cpu-mg -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(command -v python3)" \
    -DFETCHCONTENT_SOURCE_DIR_CATCH2="$PWD/build-cpu/_deps/catch2-src" > build-cpu-mg.log 2>&1
cmake --build build-cpu-mg -j 16 >> build-cpu-mg.log 2>&1 || { grep -m1 -A20 error build-cpu-mg.log; exit 1; }
ctest --test-dir build-cpu-mg -LE ghia 2>&1 | grep -E 'tests passed|Failed'

step "STREAM on these cores"
OMP_NUM_THREADS=32 build-cpu/stream_triad $((1 << 27)) | tee bench/results/${TAG}_stream.csv

bench() { # variant exe grid iters
    step "$1 ${3}^2"
    python3 bench/run_bench.py --exe "$2" --grids "$3" --iters "$4" --repeats 3 \
        --threads 1 16 24 32 --tag "${TAG}_$1_g$3" $SOLVERS 2>&1 | grep median
}
bench ft build-cpu-ft/simple 4096 6
bench mg build-cpu-mg/simple 4096 6
bench ft build-cpu-ft/simple 2048 25
bench mg build-cpu-mg/simple 2048 25

step "perf events on this CPU (local/remote memory)"
perf list --no-desc 2>/dev/null | grep -iE 'remote|local_dram|snc|upi' | head -n 40 | tee bench/results/${TAG}_perf_list.txt
EV=$(grep -oiE '\b[a-z0-9_.]+\.(local_dram|remote_dram|local_dram_[a-z0-9_]+|remote_dram_[a-z0-9_]+|snc_dram|remote_[a-z0-9_]*dram[a-z0-9_]*)\b' \
     bench/results/${TAG}_perf_list.txt | sort -u | head -n 6 | sed 's/$/:u/' | paste -sd, -)
echo "events used: ${EV:-none found}"

step "perf stat: DRAM loads by source (4096^2, 32 threads, 6 iterations)"
python3 - > "$TMPDIR/perf.cfg" <<EOF
import sys; sys.path.insert(0, "bench"); import run_bench as b
print(b.make_config(4096, 6, "$TMPDIR/perf_out",
      {"momentum_solver": "gs_multicolour", "pressure_solver": "mg"}), end="")
EOF
if command -v perf > /dev/null && [ -n "${EV:-}" ]; then
    for v in old ft mg; do
        exe=build-cpu/simple; [ $v = ft ] && exe=build-cpu-ft/simple; [ $v = mg ] && exe=build-cpu-mg/simple
        OMP_NUM_THREADS=32 perf stat -x, -e "$EV" -o "$TMPDIR/perf_$v.txt" "$exe" "$TMPDIR/perf.cfg" > /dev/null ||
            echo "perf stat failed for $v"
        grep -v '^#' "$TMPDIR/perf_$v.txt" | grep -v '^$' | sed "s/^/$v,/" | tee -a bench/results/${TAG}_perf.csv
    done
else
    echo "skipped: perf missing or no matching events (see the list above)"
fi
step done
