#include "simple/solver.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
using Combo = std::pair<std::string, std::string>; // momentum_solver, pressure_solver

const std::vector<Combo> kCombos = {
    {"jacobi", "jacobi"},        {"gs_multicolour", "gs_redblack"}, {"gs_multicolour", "gs_lex"},
    {"gs_lex", "gs_redblack"},   {"jacobi", "gs_lex"},              {"gs_lex", "jacobi"}};

struct Run {
    std::unique_ptr<State> s;
    RunResult r;
};

Run run(ConfigMap c, const Combo& combo) {
    c["momentum_solver"] = combo.first;
    c["pressure_solver"] = combo.second;
    Params p = make_params(c);
    Run out;
    out.s = std::make_unique<State>(p);
    out.r = run_simple(p, *out.s);
    return out;
}

#ifdef _OPENMP
bool same(const Field2D& a, const Field2D& b) {
    return std::equal(a.data(), a.data() + static_cast<std::size_t>(a.rows()) * a.cols(),
                      b.data());
}
#endif

double max_abs_diff(const Field2D& a, const Field2D& b) {
    double m = 0.0;
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j)
            m = std::max(m, std::fabs(a(i, j) - b(i, j)));
    return m;
}
// p is fixed only up to a constant (the p' system is singular, SPEC 4.3), so compare the interior
// cells after removing each field's interior mean.
double max_abs_diff_p(const Params& p, const Field2D& a, const Field2D& b) {
    double ma = 0.0, mb = 0.0;
    for (int i = p.H; i <= p.NX + p.H - 1; ++i)
        for (int j = p.H + 1; j <= p.NY + p.H; ++j) {
            ma += a(i, j);
            mb += b(i, j);
        }
    ma /= p.NX * p.NY;
    mb /= p.NX * p.NY;
    double m = 0.0;
    for (int i = p.H; i <= p.NX + p.H - 1; ++i)
        for (int j = p.H + 1; j <= p.NY + p.H; ++j)
            m = std::max(m, std::fabs((a(i, j) - ma) - (b(i, j) - mb)));
    return m;
}
} // namespace

TEST_CASE("parallel solvers: fields identical for 1, 2, 4, 8 threads", "[parallel_solvers]") {
#ifdef _OPENMP
    const int saved = omp_get_max_threads();
    // Fixed work (error_tolerance 0): non-square grid, QUICK + order 4 (5 colours), and UDS +
    // order 2 (2 colours).
    for (auto [scheme, order] : {std::make_pair("QUICK", "4"), std::make_pair("UDS", "2")}) {
        ConfigMap c = {{"N_cells_x", "32"},     {"N_cells_y", "40"},
                       {"Re", "400"},           {"maxit", "150"},
                       {"error_tolerance", "0"}, {"convection_scheme", scheme},
                       {"diffusion_order", order}};
        for (const Combo& combo : kCombos) {
            omp_set_num_threads(1);
            Run ref = run(c, combo);
            for (int t : {2, 4, 8}) {
                INFO(scheme << " " << combo.first << "/" << combo.second << " threads " << t);
                omp_set_num_threads(t);
                Run r = run(c, combo);
                REQUIRE(r.r.iterations == ref.r.iterations);
                REQUIRE(same(r.s->u, ref.s->u));
                REQUIRE(same(r.s->v, ref.s->v));
                REQUIRE(same(r.s->p, ref.s->p));
            }
        }
    }
    omp_set_num_threads(saved);
#else
    SKIP("built without OpenMP");
#endif
}

TEST_CASE("parallel solvers: converged fields match gs_lex on 32x32, Re 400",
          "[parallel_solvers]") {
#ifdef _OPENMP
    const int saved = omp_get_max_threads();
    omp_set_num_threads(4);
#endif
    const ConfigMap c = {{"N_cells_x", "32"},
                         {"N_cells_y", "32"},
                         {"Re", "400"},
                         {"maxit", "60000"},
                         {"convection_scheme", "QUICK"},
                         {"diffusion_order", "4"},
                         {"residual_normalisation", "absolute"},
                         {"error_tolerance", "1e-9"}};
    Run ref = run(c, {"gs_lex", "gs_lex"});
    REQUIRE(ref.r.converged);
    std::printf("[parallel_solvers] 32x32 Re 400: gs_lex/gs_lex %d outer iterations\n",
                ref.r.iterations);
    for (const Combo& combo : kCombos) {
        Run r = run(c, combo);
        const double du = max_abs_diff(r.s->u, ref.s->u);
        const double dv = max_abs_diff(r.s->v, ref.s->v);
        const double dp = max_abs_diff_p(make_params(c), r.s->p, ref.s->p);
        std::printf("[parallel_solvers] 32x32 Re 400: %s/%s %d outer iterations, max abs diff "
                    "u %.2e v %.2e p-mean(p) %.2e (raw p %.2e)\n",
                    combo.first.c_str(), combo.second.c_str(), r.r.iterations, du, dv, dp,
                    max_abs_diff(r.s->p, ref.s->p));
        INFO(combo.first << "/" << combo.second);
        REQUIRE(r.r.converged);
        REQUIRE(du <= 1e-6);
        REQUIRE(dv <= 1e-6);
        REQUIRE(dp <= 1e-6);
    }
#ifdef _OPENMP
    omp_set_num_threads(saved);
#endif
}
