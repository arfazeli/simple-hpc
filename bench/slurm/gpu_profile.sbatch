#!/bin/bash
#SBATCH --account=def-CHANGEME
#SBATCH --job-name=simple-gpu-profile
#SBATCH --gpus=mi300a:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=bench/slurm/logs/%x-%j.out
# T24 baseline profile on one MI300A: device STREAM, rocprofv3 kernel trace + memory counters,
# rocprof-compute roofline. No code changes are measured here; this is the "before" picture.
# Submit from the GPU folder's repo root:  sbatch bench/slurm/gpu_profile.sbatch
# Rerun only some steps:  sbatch --export=ALL,STEPS="pmc roof" bench/slurm/gpu_profile.sbatch
set -uo pipefail
STEPS=${STEPS:-build stream trace pmc roof}
has() { [[ " $STEPS " == *" $1 "* ]]; }
source bench/slurm/env.sh
export TMPDIR=${SLURM_TMPDIR:-/tmp}
OUT=bench/results/gpu_prof_${SLURM_JOB_ID}
mkdir -p "$OUT" bench/slurm/logs
step() { echo; echo "== $*"; }

step node
hostname; hipcc --version | head -n 1; which rocprofv3 rocprof-compute

if has build; then
step build
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DSIMPLE_HIP=ON -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_HIP_COMPILER="$(hipconfig -l)/clang++" \
    -DFETCHCONTENT_SOURCE_DIR_CATCH2="$HOME/scratch/simple-hpc/build-cpu/_deps/catch2-src" \
    > "$OUT/configure.log" 2>&1 || { tail -n 20 "$OUT/configure.log"; exit 1; }
cmake --build build-hip -j 8 --target simple simple_hip_tests > "$OUT/build.log" 2>&1 ||
    { grep -m1 -A20 error "$OUT/build.log"; exit 1; }
build-hip/simple_hip_tests 2>&1 | tail -n 2
hipcc -O3 --offload-arch=gfx942 bench/stream_hip.hip -o build-hip/stream_hip
fi

has stream && { step "device STREAM (3 x 1 GiB)"; build-hip/stream_hip | tee "$OUT/stream_hip.csv"; }

# Fixed-work configs: 2048^2 (bandwidth regime) and 512^2 (launch regime).
cfg() { # grid iters
    python3 - <<EOF
import sys; sys.path.insert(0, "bench"); import run_bench as b
print(b.make_config($1, $2, "$TMPDIR/out_$1",
      {"backend": "gpu", "momentum_solver": "gs_multicolour", "pressure_solver": "gs_redblack"}), end="")
EOF
}
cfg 2048 20 > "$TMPDIR/g2048.cfg"
cfg 512 50 > "$TMPDIR/g512.cfg"
cfg 2048 3 > "$TMPDIR/g2048_short.cfg"

has trace && step "rocprofv3 kernel trace + stats (2048^2 x 20, 512^2 x 50)"
has trace && for g in 2048 512; do
    timeout 15m rocprofv3 --kernel-trace --stats --output-format csv -d "$OUT" -o "trace_$g" \
        -- build-hip/simple "$TMPDIR/g$g.cfg" > /dev/null 2> "$OUT/rocprofv3_$g.err" ||
        { echo "rocprofv3 failed:"; tail -n 15 "$OUT/rocprofv3_$g.err"; }
    f=$(find "$OUT" -name "trace_${g}_kernel_stats.csv" | head -1)
    [ -n "$f" ] && { echo "-- $g^2 top kernels"; head -n 12 "$f" | cut -c1-200; }
done

if has pmc; then
step "rocprofv3 memory counters (2048^2 x 3)"
rocprofv3 -L > "$OUT/counter_list.txt" 2>&1
grep -oE '\b(FETCH_SIZE|WRITE_SIZE|TCC_EA0_RDREQ_sum|TCC_EA0_WRREQ_sum|TCC_EA0_RDREQ_64B_sum|TCC_EA0_WRREQ_64B_sum)\b' \
    "$OUT/counter_list.txt" | sort -u | tr '\n' ' '; echo
# One pass per counter, each with a timeout: in job 26754980 FETCH_SIZE+WRITE_SIZE aborted
# inside rocprofv3 at the first launch and its signal handler then hung for 45 min.
# ponytail: raw TCC_EA0 request counts first; bytes = 64 * requests (check against _64B_ variants).
n=0
# Job 26754980 log: "error code 38: Request exceeds the capabilities of the hardware" -> one counter per pass.
for set in FETCH_SIZE WRITE_SIZE TCC_EA0_RDREQ_sum TCC_EA0_WRREQ_sum TCC_EA0_RDREQ_64B_sum TCC_EA0_WRREQ_64B_sum; do
    n=$((n + 1)); echo "-- pass $n: $set"
    timeout -k 30s 8m rocprofv3 --pmc $set --output-format csv -d "$OUT" -o "counters_$n" \
        -- build-hip/simple "$TMPDIR/g2048_short.cfg" > /dev/null 2> "$OUT/rocprofv3_pmc_$n.err"
    echo "exit $?"; grep -m3 -iE 'error|abort|invalid' "$OUT/rocprofv3_pmc_$n.err" | cut -c1-200
    find "$OUT" -name "counters_${n}_counter_collection.csv" -exec wc -l {} \;
done
fi

has roof || { step done; exit 0; }
step "rocprof-compute roofline (2048^2 x 3)"
( cd "$TMPDIR" && timeout 30m rocprof-compute profile -n simple2048 --roof-only \
    -- "$OLDPWD/build-hip/simple" "$TMPDIR/g2048_short.cfg" > "$TMPDIR/rpc.log" 2>&1 ) ||
    { echo "rocprof-compute failed:"; tail -n 20 "$TMPDIR/rpc.log"; }
tail -n 5 "$TMPDIR/rpc.log"
[ -d "$TMPDIR/workloads" ] && cp -r "$TMPDIR/workloads" "$OUT/" && find "$OUT/workloads" -name '*.pdf' -o -name '*.html' | head

step "sizes"; du -sh "$OUT"
step done
