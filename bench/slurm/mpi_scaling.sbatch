#!/bin/bash
#SBATCH --account=def-CHANGEME
#SBATCH --job-name=simple-mpi-scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=bench/slurm/logs/%x-%j.out
# MPI scaling study, one job per node count (pick it at submit time):
#   for n in 1 2 4 8; do sbatch --nodes=$n bench/slurm/mpi_scaling.sbatch; done
# Strong: 4096² and 8192² at fixed iterations. Weak: a 2048² block per node (2048x2048, 4096x2048,
# 4096x4096, 8192x4096 on 1, 2, 4, 8 nodes). Each in two layouts: hybrid (one rank per NUMA domain,
# OpenMP threads filling it) and pure MPI (one rank per core). 3 repeats, layouts alternating.
# Inter-node ping-pong for the alpha-beta model on 2+ nodes.
# Needs build-mpi built from the current tree (mpi_test.sbatch does that).
# ponytail: --ntasks-per-node=192 is nibi's core count; on another cluster set it to the cores per
# node so the pure-MPI step fits in the allocation.
set -uo pipefail
source bench/slurm/env.sh
module load openmpi
export OMP_PLACES=cores OMP_PROC_BIND=close
N=$SLURM_NNODES
OUT=bench/results/mpi_scaling/n${N}_${SLURM_JOB_ID}
mkdir -p "$OUT"

NUMA=$(lscpu | awk -F: '/^NUMA node\(s\)/ {gsub(/ /, "", $2); print $2}')
CORES=$(lscpu -p=SOCKET,CORE | grep -v '^#' | sort -u | wc -l) # physical cores per node
echo "== $N nodes: $SLURM_JOB_NODELIST; $CORES cores and $NUMA NUMA domains per node"
lscpu | grep -E 'Model name|NUMA node[0-9]'
git log -1 --oneline 2>/dev/null

if [ "$N" -ge 2 ]; then
    echo "== ping-pong, 2 nodes, 1 rank each"
    mpicxx -O2 bench/mpi_pingpong.cpp -o "$OUT/mpi_pingpong"
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=1 "$OUT/mpi_pingpong" |
        tee "$OUT/pingpong.csv"
fi

# One run: grid_x grid_y iters ranks_per_node threads label rep
run() {
    local gx=$1 gy=$2 it=$3 rpn=$4 t=$5 kind=$6 rep=$7
    local tag=${kind}_${gx}x${gy}_${rpn}x${t}_r${rep} d
    d=$OUT/$tag # shared scratch: every rank reads the cfg
    sed -e "s/^N_cells_x = .*/N_cells_x = $gx/" -e "s/^N_cells_y = .*/N_cells_y = $gy/" \
        -e "s/^maxit = .*/maxit = $it/" -e "s/^error_tolerance = .*/error_tolerance = 0/" \
        -e "s/^backend = .*/backend = openmp/" \
        -e "s/^momentum_solver = .*/momentum_solver = gs_multicolour/" \
        -e "s/^pressure_solver = .*/pressure_solver = mg/" -e "s|^output_dir = .*|output_dir = $d|" \
        configs/cavity_re1000_80.cfg > "$d.cfg"
    echo "write_fields = False" >> "$d.cfg"
    if OMP_NUM_THREADS=$t srun --nodes=$N --ntasks=$((N * rpn)) --ntasks-per-node=$rpn \
        --cpus-per-task=$t --cpu-bind=cores build-mpi/simple "$d.cfg" > "$d.log" 2>&1; then
        python3 - "$d/run_info.json" "$N,$kind,$((N * rpn)),$t,$gx,$gy,$it,$rep" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
keys = ["seconds", "halo_seconds_mean", "halo_seconds_max", "collective_seconds_mean",
        "collective_seconds_max"]
print(sys.argv[2] + "," + ",".join(str(r[k]) for k in keys))
EOF
        cp "$d/timing.csv" "$OUT/timing_$tag.csv"
    else
        echo "$N,$kind,$((N * rpn)),$t,$gx,$gy,$it,$rep,FAILED"; head -n 5 "$d.log" >&2
    fi
    rm -rf "$d" "$d.cfg" "$d.log"
}

case $N in # weak scaling: 2048² cells per node
1) WX=2048 WY=2048 ;; 2) WX=4096 WY=2048 ;; 4) WX=4096 WY=4096 ;; 8) WX=8192 WY=4096 ;;
*) WX=0 ;;
esac

echo "== runs"
{
    echo "nodes,study,ranks,threads,grid_x,grid_y,iters,rep,seconds,halo_mean,halo_max,collective_mean,collective_max"
    for rep in 1 2 3; do
        for layout in "$NUMA $((CORES / NUMA))" "$CORES 1"; do # hybrid, pure MPI
            set -- $layout
            run 4096 4096 100 "$1" "$2" strong "$rep"
            run 8192 8192 50 "$1" "$2" strong "$rep"
            [ "$WX" -gt 0 ] && run "$WX" "$WY" 200 "$1" "$2" weak "$rep"
        done
    done
} | tee "$OUT/summary.csv"
echo "== done"
