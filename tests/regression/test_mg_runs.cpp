#include "simple/config.hpp"
#include "simple/solver.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
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
#ifdef _OPENMP
bool bit_identical(const Field2D& a, const Field2D& b) {
    return a.rows() == b.rows() && a.cols() == b.cols() &&
           std::memcmp(a.data(), b.data(), sizeof(double) * a.rows() * a.cols()) == 0;
}
#endif
// max |(a - mean a) - (b - mean b)| over interior pressure cells.
double max_abs_diff_mean_free(const Params& p, const Field2D& a, const Field2D& b) {
    double sa = 0.0, sb = 0.0;
    for (int i = p.H; i <= p.NX + p.H - 1; ++i)
        for (int j = p.H + 1; j <= p.NY + p.H; ++j) {
            sa += a(i, j);
            sb += b(i, j);
        }
    const double n = static_cast<double>(p.NX) * p.NY, ma = sa / n, mb = sb / n;
    double m = 0.0;
    for (int i = p.H; i <= p.NX + p.H - 1; ++i)
        for (int j = p.H + 1; j <= p.NY + p.H; ++j)
            m = std::max(m, std::fabs((a(i, j) - ma) - (b(i, j) - mb)));
    return m;
}
} // namespace

TEST_CASE("mg run: 1, 2, 4, 8 threads give bit-identical u, v, p (32x40)", "[mg][threads]") {
#ifdef _OPENMP
    const std::string root = SIMPLE_SOURCE_DIR;
    ConfigMap c = load_config(root + "/configs/snapshot/converged_32x40.cfg");
    c["pressure_solver"] = "mg";
    c["error_tolerance"] = "0";
    c["maxit"] = "150";
    const Params p = make_params(c);
    const int saved = omp_get_max_threads();
    std::vector<std::unique_ptr<State>> runs;
    for (int t : {1, 2, 4, 8}) {
        omp_set_num_threads(t);
        runs.push_back(std::make_unique<State>(p));
        runs.back()->mg.parallel_min_cells = 0; // run every level in parallel, even 4 x 5
        run_simple(p, *runs.back());
    }
    omp_set_num_threads(saved);
    for (std::size_t k = 1; k < runs.size(); ++k) {
        INFO("thread run " << k);
        REQUIRE(bit_identical(runs[k]->u, runs[0]->u));
        REQUIRE(bit_identical(runs[k]->v, runs[0]->v));
        REQUIRE(bit_identical(runs[k]->p, runs[0]->p));
    }
#else
    SKIP("built without OpenMP");
#endif
}

TEST_CASE("mg run converges to the gs_lex solution (32x32, Re 400, tol 1e-9)", "[mg]") {
    ConfigMap c = {{"N_cells_x", "32"},
                   {"N_cells_y", "32"},
                   {"Re", "400"},
                   {"convection_scheme", "QUICK"},
                   {"diffusion_order", "4"},
                   {"residual_normalisation", "absolute"},
                   {"error_tolerance", "1e-9"},
                   {"maxit", "40000"}};
    const Params pg = make_params(c);
    State sg(pg);
    const RunResult rg = run_simple(pg, sg);
    REQUIRE(rg.converged);
    std::printf("32x32 Re 400 tol 1e-9: gs_lex %d outer iterations\n", rg.iterations);
    c["pressure_solver"] = "mg";
    for (const char* cycles : {"1", "2"}) {
        c["mg_cycles"] = cycles;
        const Params pm = make_params(c);
        State sm(pm);
        const RunResult rm = run_simple(pm, sm);
        const double du = max_abs_diff(sm.u, sg.u), dv = max_abs_diff(sm.v, sg.v);
        const double dp = max_abs_diff_mean_free(pm, sm.p, sg.p);
        std::printf("32x32 Re 400 tol 1e-9: mg_cycles %s %d outer iterations; vs gs_lex max abs "
                    "diff u %.3e v %.3e p %.3e\n",
                    cycles, rm.iterations, du, dv, dp);
        INFO("mg_cycles " << cycles);
        REQUIRE(rm.converged);
        REQUIRE(du <= 1e-6);
        REQUIRE(dv <= 1e-6);
        REQUIRE(dp <= 1e-6);
    }
}
