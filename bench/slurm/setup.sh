#!/bin/bash
# Run on a cluster LOGIN node (compute nodes may have no internet, and CMake fetches Catch2).
# Usage: bash bench/slurm/setup.sh      (from the repo root)
set -euo pipefail
source bench/slurm/env.sh
mkdir -p bench/slurm/logs bench/results

cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(command -v python3)" \
    2>&1 | tail -n 5
cmake --build build-cpu -j 8 2>&1 | tail -n 5
g++ -O3 -fopenmp bench/stream_triad.cpp -o build-cpu/stream_triad

# Quick sanity check on the login node: fast tests, 1 thread.
ctest --test-dir build-cpu -LE ghia 2>&1 | tail -n 3
