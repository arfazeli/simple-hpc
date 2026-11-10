#include "simple/config.hpp"
#include "simple/solver.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
double max_abs_diff(const Field2D& a, const Field2D& b) {
    double m = 0.0;
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j)
            m = std::max(m, std::fabs(a(i, j) - b(i, j)));
    return m;
}
} // namespace

TEST_CASE("cg run converges to the gs_lex solution (32x40, Re 400)", "[cg]") {
    const std::string root = SIMPLE_SOURCE_DIR;
    ConfigMap c = load_config(root + "/configs/snapshot/converged_32x40.cfg");
    c["error_tolerance"] = "1e-6";
    Params pg = make_params(c);
    c["pressure_solver"] = "cg";
    Params pc = make_params(c);
    State sg(pg), sc(pc);
    RunResult rg = run_simple(pg, sg);
    RunResult rc = run_simple(pc, sc);
    std::printf("32x40: gs_lex %d iterations, cg %d iterations (mean %.1f CG iterations)\n",
                rg.iterations, rc.iterations,
                static_cast<double>(rc.cg_iterations) / rc.iterations);
    REQUIRE(rg.converged);
    REQUIRE(rc.converged);
    REQUIRE(rc.cg_iterations > 0);
    const double du = max_abs_diff(sc.u, sg.u), dv = max_abs_diff(sc.v, sg.v);
    std::printf("32x40: cg vs gs_lex max abs diff u %.3e v %.3e\n", du, dv);
    REQUIRE(du <= 1e-4);
    REQUIRE(dv <= 1e-4);
}

TEST_CASE("cg run: 1, 2, 4, 8 threads agree to 1e-10", "[cg][threads]") {
#ifdef _OPENMP
    // Fixed work (300 outer iterations). The CG tolerance is tight so that a CG iteration count
    // flipping by one between thread counts (reduction order) changes p' by far less than 1e-10.
    ConfigMap c = {{"N_cells_x", "32"},
                   {"N_cells_y", "32"},
                   {"Re", "1000"},
                   {"error_tolerance", "0"},
                   {"maxit", "300"},
                   {"pressure_solver", "cg"},
                   {"pressure_tolerance", "1e-10"}};
    const Params p = make_params(c);
    const int saved = omp_get_max_threads();
    std::vector<std::unique_ptr<State>> runs;
    for (int t : {1, 2, 4, 8}) {
        omp_set_num_threads(t);
        runs.push_back(std::make_unique<State>(p));
        run_simple(p, *runs.back());
    }
    omp_set_num_threads(saved);
    for (std::size_t k = 1; k < runs.size(); ++k) {
        const double du = max_abs_diff(runs[k]->u, runs[0]->u);
        const double dv = max_abs_diff(runs[k]->v, runs[0]->v);
        const double dp = max_abs_diff(runs[k]->p, runs[0]->p);
        std::printf("threads run %zu vs 1 thread: max abs diff u %.3e v %.3e p %.3e\n", k, du, dv,
                    dp);
        REQUIRE(du <= 1e-10);
        REQUIRE(dv <= 1e-10);
        REQUIRE(dp <= 1e-10);
    }
#else
    SKIP("built without OpenMP");
#endif
}
