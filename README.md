# simple-hpc

A C++17 finite-volume solver for the steady 2D incompressible Navier–Stokes equations, using the SIMPLE pressure–velocity coupling on a staggered grid, applied to the lid-driven cavity. QUICK convection, 2nd- or 4th-order diffusion, and a choice of inner solvers, including geometric multigrid for the pressure correction.

One code base runs on:

- **CPU cores**, with OpenMP
- **cluster nodes**, with MPI (2D domain decomposition; pure MPI or hybrid MPI + OpenMP)
- **GPUs**: AMD with HIP, NVIDIA with CUDA, from the same source

Every parallel path gives the same result as the serial code, bit for bit, and the solver reproduces the Ghia et al. (1982) cavity benchmark. Performance results: [`docs/RESULTS.md`](docs/RESULTS.md) and the [case study](https://arfazeli.github.io/simple-hpc.html). Numerical method: [`docs/SPEC.md`](docs/SPEC.md).

## Build

Needs CMake ≥ 3.20 and a C++17 compiler. Python 3 with `numpy` and `matplotlib` (`postprocess/requirements.txt`) for the validation and plots.

```bash
cmake -S . -B build && cmake --build build -j       # CPU, OpenMP on by default
```

Optional backends, each in its own build directory:

| | CMake flag | needs |
|---|---|---|
| MPI | `-DSIMPLE_MPI=ON` | an MPI library |
| AMD GPU | `-DSIMPLE_HIP=ON` | ROCm (default target `gfx942`, MI300A) |
| NVIDIA GPU | `-DSIMPLE_CUDA=ON` | CUDA toolkit (default target `sm_90`, H100) |

Override the GPU target with `-DCMAKE_HIP_ARCHITECTURES=...` or `-DCMAKE_CUDA_ARCHITECTURES=...`.

## Run a simulation

A run is described by a config file of `key = value` lines:

```ini
# 128 x 128 cavity at Re 1000
N_cells_x = 128
N_cells_y = 128
Re = 1000
convection_scheme = QUICK        # or UDS
diffusion_order = 4              # or 2
backend = openmp                 # serial, openmp, gpu
momentum_solver = gs_multicolour # gs_lex, jacobi, gs_multicolour
pressure_solver = mg             # gs_lex, jacobi, gs_redblack, cg, mg
error_tolerance = 1e-5
maxit = 60000
output_dir = results/re1000_128
```

```bash
OMP_NUM_THREADS=8 build/simple my_case.cfg
```

Unset keys take their defaults, and unknown keys are an error. All keys are listed in [`docs/SPEC.md`](docs/SPEC.md#1-parameters). Ready-made cases are in `configs/`, including the Ghia validation cases in `configs/validation/`.

**On several nodes (MPI).** Use the MPI build and launch it with your MPI runner. More than one rank needs `momentum_solver = gs_multicolour` and any `pressure_solver` except `gs_lex`. The process grid is chosen automatically, or set it with `mpi_grid = 2x2`.

```bash
mpirun -n 4 build-mpi/simple my_case.cfg              # or srun on a Slurm cluster
```

**On a GPU.** Use a HIP or CUDA build with `backend = gpu`. The GPU backend supports `momentum_solver = gs_multicolour` with `pressure_solver = gs_redblack`.

## Output and post-processing

Each run writes to `output_dir`:

- `u.csv`, `v.csv`, `p.csv`: the staggered fields, and `ux_centre.csv`, `uy_centre.csv`: velocities at the cell centres
- `alog.csv`: residual history
- `timing.csv`: time per solver phase
- `run_info.json`: config, iterations, run time, git hash, and on MPI runs the halo and reduction times

```bash
python3 postprocess/make_figures.py results/re1000_128                     # streamlines, pressure, centrelines, residuals
python3 postprocess/validate_ghia.py results/re1000_128                    # error against Ghia et al.
```

For timing runs, `write_fields = False` skips the field CSVs, which run to gigabytes on large grids.

## Tests

```bash
ctest --test-dir build -LE ghia    # unit, snapshot and parallel-consistency tests (seconds)
ctest --test-dir build -L ghia     # Ghia validation runs (minutes; needs numpy)
```

The MPI build adds tests that run on 1–4 ranks and check that the fields match one rank bit for bit. GPU builds add `simple_hip_tests`, which checks every GPU kernel against the CPU code.

## Cluster jobs

`bench/slurm/` holds the Slurm scripts behind the results: Ghia validation, OpenMP and NUMA scaling, two-node MPI tests, MPI strong and weak scaling, and GPU tests and profiling (`VENDOR=cuda` for NVIDIA). Set your allocation with `sbatch --account=...`, and see each script's header for how to submit it.

## License

MIT
