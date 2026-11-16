#!/bin/bash
#SBATCH --account=def-CHANGEME
#SBATCH --job-name=simple-overlap
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=bench/slurm/logs/%x-%j.out
# Halo overlap on vs off: 64 MPI ranks (32 per node, 1 thread), mg pressure, fixed iterations.
# Overlap and blocking runs alternate so drift on the node hits both alike.
# Submit from the repo root after mpi_test.sbatch has built build-mpi:  sbatch bench/slurm/overlap.sbatch
# ponytail: one pure-MPI layout (small blocks, most communication); the full layout matrix is T46.
set -uo pipefail
source bench/slurm/env.sh
module load openmpi
export OMP_NUM_THREADS=1
OUT=bench/results/overlap_${SLURM_JOB_ID}
mkdir -p "$OUT"
echo "$SLURM_JOB_NODELIST"; git log -1 --oneline 2>/dev/null

{
echo "grid,iters,overlap,rep,seconds"
for g_it in 1024:1000 2048:300 4096:100; do
    g=${g_it%:*}; it=${g_it#*:}
    for rep in 1 2 3; do
        for ov in True False; do
            run=$OUT/g${g}_${ov}_${rep} # shared scratch: every rank on every node reads the cfg
            sed -e "s/^N_cells_x = .*/N_cells_x = $g/" -e "s/^N_cells_y = .*/N_cells_y = $g/" \
                    -e "s/^maxit = .*/maxit = $it/" -e "s/^error_tolerance = .*/error_tolerance = 0/" \
                    -e "s/^momentum_solver = .*/momentum_solver = gs_multicolour/" \
                    -e "s/^pressure_solver = .*/pressure_solver = mg/" -e "s|^output_dir = .*|output_dir = $run|" \
                    configs/cavity_re1000_80.cfg > "$run.cfg"
            echo "halo_overlap = $ov" >> "$run.cfg"
            if srun build-mpi/simple "$run.cfg" > "$run.log" 2>&1; then
                s=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['seconds'])" "$run/run_info.json")
                echo "$g,$it,$ov,$rep,$s"
                cp "$run/timing.csv" "$OUT/timing_g${g}_${ov}_${rep}.csv"
            else
                echo "$g,$it,$ov,$rep,FAILED"; head -n 5 "$run.log"
            fi
            rm -rf "$run" "$run.cfg" "$run.log" # fields are GBs at 4096²
        done
    done
done
} | tee "$OUT/summary.csv"
echo "== done"
