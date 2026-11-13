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
    // MPI slab decomposition (set by decompose(), decomp.hpp). Defaults = one rank owning all rows,
    // so serial callers that never call decompose() are unaffected.
    int rank = 0, nranks = 1;
    int row_lo = 0, row_hi = INT_MAX;   // global rows this rank computes (inclusive)
    int alloc_row0 = 0, alloc_rows = 0; // stored rows: owned plus H halo rows (0 = all nrows)
};

// Clamp a kernel's global row range [a, b] to the rows this rank owns.
inline int row_from(const Params& p, int a) { return std::max(a, p.row_lo); }
inline int row_to(const Params& p, int b) { return std::min(b, p.row_hi); }
inline bool owns_row(const Params& p, int i) { return i >= p.row_lo && i <= p.row_hi; }

Params make_params(const ConfigMap& cfg);
void print_params(std::ostream& os, const Params& p);
