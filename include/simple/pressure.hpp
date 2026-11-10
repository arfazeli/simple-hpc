#pragma once
#include "simple/field2d.hpp"
#include "simple/params.hpp"

#include <vector>

struct PressureWork {
    Field2D AEP, AWP, ANP, ASP, APP, b, pc;
    PressureWork() = default;
    PressureWork(int rows, int cols);
};

// SPEC 3.7 steps 1-3. Returns res_p.
double assemble_p(const Params& p, const Field2D& APU, const Field2D& APV, const Field2D& u,
                  const Field2D& v, PressureWork& w);
// SPEC 3.7 step 4: pc = 0, then maxit_inner_p lexicographic GS sweeps (two-statement update).
void solve_pc_gs_lex(const Params& p, PressureWork& w);
// Parallel variants (T18), same two-statement update. Red-black: colour (i + j) mod 2, colour 0
// first, in place. Jacobi: neighbours from `old`, a copy of p' taken before each sweep, then
// pc = w*pc + (1-w)*old with w = kPcJacobiWeight.
// Damped: undamped Jacobi has the checkerboard eigenvalue -1 on the 5-point Neumann operator.
constexpr double kPcJacobiWeight = 0.8;
void solve_pc_jacobi(const Params& p, PressureWork& w, Field2D& old);
void solve_pc_redblack(const Params& p, PressureWork& w);
// SPEC 3.7 step 5.
void correct(const Params& p, const PressureWork& w, const Field2D& APU, const Field2D& APV,
             Field2D& u, Field2D& v, Field2D& p_field);

// Work arrays for the conjugate-gradient p' solver (pressure_solver = cg), allocated once.
struct CGWork {
    Field2D r, z, d, q; // residual, preconditioned residual, search direction, A*d
    int last_iterations = 0;
    double last_rel_residual = 0.0;
    double last_mean_removed = 0.0;         // mean of the iterate before the final mean removal
    std::vector<double>* history = nullptr; // if set, ||r||/||b|| per iteration (index 0: 1.0)
    CGWork() = default;
    CGWork(int rows, int cols);
};

// p' = A^+ b by Jacobi-preconditioned CG, matrix-free over interior cells, where
// A p' = APP*p'_P - (AEP*p'_E + AWP*p'_W + ANP*p'_N + ASP*p'_S). A is symmetric positive
// semi-definite with null space = constants, so b is projected onto mean zero, every
// preconditioned residual is projected too (the iterate stays mean-free), and the mean of
// p' is removed at the end. Stops at ||r||/||b|| <= pressure_tolerance or pressure_maxit
// iterations. Writes w.pc; returns the iteration count. ||b|| == 0 gives p' = 0.
int solve_pc_cg(const Params& p, PressureWork& w, CGWork& cg);
// out = A x over interior cells (halo entries of out untouched).
void apply_A_p(const Params& p, const PressureWork& w, const Field2D& x, Field2D& out);
