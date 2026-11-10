#pragma once
#include "simple/coefficients.hpp"
#include "simple/field2d.hpp"
#include "simple/multigrid.hpp"
#include "simple/params.hpp"
#include "simple/pressure.hpp"
#include "simple/timer.hpp"

#include <array>
#include <functional>
#include <vector>

struct State {
    Field2D u, v, p;
    Coeffs cu, cv;
    PressureWork pw;
    Field2D scratch; // previous iterate for the Jacobi sweeps (u, v and p'), allocated once
    CGWork cg;       // pressure_solver = cg
    MGWork mg;       // pressure_solver = mg: level hierarchy, allocated once (empty otherwise)
    explicit State(const Params& p);
};

struct Residuals {
    double u, v, p;
};

// Steps 1-3 of SPEC 3.4.
Residuals outer_iteration(const Params& p, State& s, PhaseTimers& t);

struct RunResult {
    int iterations = 0;
    bool converged = false;
    std::vector<std::array<double, 4>> log;
    double seconds = 0.0;
    PhaseTimers timers; // filled by run_simple; write_outputs adds Output on a copy
    long long cg_iterations = 0; // pressure_solver = cg: CG iterations summed over outer iterations
};

RunResult run_simple(const Params& p, State& s);

// The outer loop and stopping rule (SPEC 5) around one-iteration function `step`; shared by the
// CPU (run_simple) and GPU (run_simple_hip) backends.
RunResult run_outer_loop(const Params& p, const std::function<Residuals(PhaseTimers&)>& step);
