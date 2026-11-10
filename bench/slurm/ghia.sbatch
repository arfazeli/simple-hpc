#!/bin/bash
#SBATCH --account=def-CHANGEME
#SBATCH --job-name=simple-ghia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=bench/slurm/logs/%x-%j.out
# Full test suite on the cluster build, including the Ghia validation runs.
# Submit from the repo root after setup.sh:  sbatch bench/slurm/ghia.sbatch
set -euo pipefail
source bench/slurm/env.sh
hostname; lscpu | grep 'Model name'
ctest --test-dir build-cpu --output-on-failure 2>&1 | tail -n 40
