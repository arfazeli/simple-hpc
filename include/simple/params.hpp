#pragma once
#include "simple/config.hpp"

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
};

Params make_params(const ConfigMap& cfg);
void print_params(std::ostream& os, const Params& p);
