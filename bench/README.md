# Benchmark runner

`run_bench.py` times fixed-work runs of `build/simple` on square QUICK cavities
(Re 1000, diffusion_order 4, error_tolerance 0, so exactly `--iters` outer iterations run).
Per-phase times come from each run's `timing.csv`.

Command (after building `build/simple`):

    python3 bench/run_bench.py --grids 64 128 256 512 --iters 200 --repeats 3 --threads 1 --tag serial

Results go to `bench/results/<tag>_<YYYYmmdd-HHMM>.csv`, one row per run, with per-phase seconds
and `cpu`, `compiler`, `git_hash` columns. The runner prints the median `seconds_per_iter` per (grid, threads).
The CSVs cited in `docs/RESULTS.md` are committed under `bench/results/`.

`--set KEY=VALUE` (repeatable) overrides a config key after the built-in overrides, e.g.
`--set convection_scheme=UDS --set maxit_inner_p=16`. Use `--exe build-omp/simple` for the OpenMP build;
`--threads` sets `OMP_NUM_THREADS` per run and the thread count is recorded in `run_info.json` (`"threads"`).
- `--cooldown SECONDS` sleeps before each run. Runs are ordered repeat by repeat with the thread list rotated, so throttling on a laptop does not bias one thread count. Report medians; on a fanless machine single-run differences under ~15–20 % are noise.
