#pragma once
#include "simple/config.hpp"

#include <algorithm>
#include <climits>
#include <ostream>
#include <string>

enum class ResidualNorm { Absolute, RunningMax };
enum class Convection { UDS, QUICK };
enum class Backend { Serial, OpenMP, GPU };
enum class MomentumSolver { GsLex, Jacobi, GsMulticolour };
enum class PressureSolver { GsLex, Jacobi, GsRedBlack, CG, MG };

struct Params {
    int NX = 80, NY = 80, H = 2;
    int N_halo_x = 2, N_halo_y = 2;
    double Re = 1000.0;
    int maxit_inner_u = 4, maxit_inner_v = 4, maxit_inner_p = 8;
    double error_tolerance = 1e-4;
    int maxit = 15000;
    double relaxation_factor_u = 0.3, relaxation_factor_v = 0.3, relaxation_factor_p = 0.2;
    Convection convection_scheme = Convection::QUICK;
    int diffusion_order = 4;
    bool Hybrid_switch = false;
    ResidualNorm residual_normalisation = ResidualNorm::Absolute;
    Backend backend = Backend::Serial;
    MomentumSolver momentum_solver = MomentumSolver::GsLex;
    PressureSolver pressure_solver = PressureSolver::GsLex;
    double pressure_tolerance = 1e-6; // cg: stop when ||r|| / ||b|| <= this
    int pressure_maxit = 1000;        // cg: iteration cap per outer iteration
    int mg_cycles = 1;                // mg: V-cycles per outer iteration
    int mg_pre = 2, mg_post = 2;      // mg: red-black sweeps before / after the coarse correction
    double mg_omega = 1.8;            // mg: over-correction factor of the prolongated correction
    std::string output_dir = "results/run";
    // derived
    int nrows = 0, ncols = 0;
    double dx = 0.0, dy = 0.0, nu = 0.0;
    // MPI process grid (config mpi_grid = auto | RxC, 0 = auto).
    int grid_rows_cfg = 0, grid_cols_cfg = 0;
    // MPI block decomposition (set by decompose(), decomp.hpp). Defaults = one rank owning the
    // whole grid, so serial callers that never call decompose() are unaffected.
    int rank = 0, nranks = 1;
    int grid_rows = 1, grid_cols = 1;   // process grid Pr x Pc; rank = grid_i * Pc + grid_j
    int grid_i = 0, grid_j = 0;         // this rank's position in it
    int row_lo = 0, row_hi = INT_MAX;   // global rows this rank computes (inclusive)
    int col_lo = 0, col_hi = INT_MAX;   // global columns this rank computes (inclusive)
    int alloc_row0 = 0, alloc_rows = 0; // stored rows: owned plus H halo rows (0 = all nrows)
    int alloc_col0 = 0, alloc_cols = 0; // stored columns: owned plus H halo (0 = all ncols)
    // pressure_solver = mg on > 1 rank: levels 0..mg_split-1 are decomposed, level mg_split is
    // gathered onto every rank. The interior is split in units of 2^mg_split cells. 0 = no split.
    int mg_split = 0;
    // > 1 rank: overlap the halo exchanges inside coloured sweeps with the interior update.
    bool halo_overlap = true;
    // False: skip the field CSVs (u, v, p, centrelines) and their MPI gather; alog, timing and
    // run_info are still written. For timing runs, where 8192² fields are GBs per run.
    bool write_fields = true;
};

// Clamp a kernel's global row / column range [a, b] to the cells this rank owns.
inline int row_from(const Params& p, int a) { return std::max(a, p.row_lo); }
inline int row_to(const Params& p, int b) { return std::min(b, p.row_hi); }
inline bool owns_row(const Params& p, int i) { return i >= p.row_lo && i <= p.row_hi; }
inline int col_from(const Params& p, int a) { return std::max(a, p.col_lo); }
inline int col_to(const Params& p, int b) { return std::min(b, p.col_hi); }
inline bool owns_col(const Params& p, int j) { return j >= p.col_lo && j <= p.col_hi; }

Params make_params(const ConfigMap& cfg);
void print_params(std::ostream& os, const Params& p);
