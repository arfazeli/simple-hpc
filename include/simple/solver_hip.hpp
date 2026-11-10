#pragma once
// HIP backend (T23): the whole outer iteration on the GPU, fields device-resident for the run.
// Supports momentum_solver = gs_multicolour with pressure_solver = gs_redblack only. Per-cell
// arithmetic matches the CPU kernels expression for expression (SPEC 7), so fields are expected
// to be bit-identical to the CPU path; residual sums use a different (fixed) order.
// Plain C++ header: no HIP types, so CPU-compiled tests can include it.
#include "simple/solver.hpp"

#include <memory>

class HipSolver {
  public:
    HipSolver(const Params& p, const State& s); // allocates device copies of all State arrays
    ~HipSolver();
    HipSolver(const HipSolver&) = delete;
    HipSolver& operator=(const HipSolver&) = delete;

    // One phase each, same order and semantics as the CPU functions in outer_iteration.
    void assemble_u();
    void bc_u();
    double residual_u();
    void sweep_u(int nsweeps);
    void assemble_v();
    void bc_v();
    double residual_v();
    void sweep_v(int nsweeps);
    double assemble_p();
    void solve_p();
    void correct();

    Residuals outer_iteration(PhaseTimers& t);
    void download(State& s) const; // device -> host for every State array the GPU uses

  private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};

// run_simple on the GPU; s holds the final fields on return.
RunResult run_simple_hip(const Params& p, State& s);
