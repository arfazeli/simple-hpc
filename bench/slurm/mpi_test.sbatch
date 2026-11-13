#!/bin/bash
#SBATCH --account=def-CHANGEME
#SBATCH --job-name=simple-mpi-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --output=bench/slurm/logs/%x-%j.out
# Build with MPI, run the fast tests, then the MPI byte-identity test across two nodes: fields and
# residual history on 2 and 4 ranks (1 and 2 per node, 1 and 4 OpenMP threads) must match one rank.
# Submit from the repo root:  sbatch bench/slurm/mpi_test.sbatch
set -uo pipefail # no -e: later steps still run and report if one fails
source bench/slurm/env.sh
module load openmpi
mkdir -p bench/slurm/logs
export TMPDIR=${SLURM_TMPDIR:-/tmp}
step() { echo; echo "== $*"; }

step nodes
echo "$SLURM_JOB_NODELIST"; srun --ntasks-per-node=1 hostname
mpicxx --version | head -n 1; mpiexec --version | head -n 1

step configure
CATCH=""
for d in "$HOME/scratch/simple-hpc/build-cpu/_deps/catch2-src" build-cpu/_deps/catch2-src; do
    [ -d "$d" ] && CATCH="-DFETCHCONTENT_SOURCE_DIR_CATCH2=$d" && break # reuse; no download
done
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release -DSIMPLE_MPI=ON \
    -DPython3_EXECUTABLE="$(command -v python3)" $CATCH 2>&1 | grep -E 'MPI|OpenMP|Error'

step build
if ! cmake --build build-mpi -j "$SLURM_CPUS_PER_TASK" > build-mpi.log 2>&1; then
    grep -m 1 -B 2 -A 25 -E 'error' build-mpi.log
    echo "BUILD FAILED"; exit 1
fi

step "fast tests (mpi_slab_n1..4 run on this node via mpiexec)"
ctest --test-dir build-mpi -LE ghia 2>&1 | grep -E 'tests passed|Failed|\*\*\*'

# Across nodes with srun; each line ends in "fields byte-identical; residual log identical".
for cfg in "2 1 1" "4 2 1" "4 2 4"; do # ranks, ranks per node, OpenMP threads per rank
    set -- $cfg
    step "srun: $1 ranks on 2 nodes ($2 per node), $3 OpenMP threads each"
    OMP_NUM_THREADS=$3 srun --nodes=2 --ntasks=$1 --ntasks-per-node=$2 --cpus-per-task=$3 \
        build-mpi/simple_mpi_tests | grep -v '^iter'
    echo "exit code: ${PIPESTATUS[0]}"
done
step done
