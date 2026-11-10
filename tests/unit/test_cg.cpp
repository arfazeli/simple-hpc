#include "simple/config.hpp"
#include "simple/pressure.hpp"
#include "simple/solver.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>
#include <vector>

namespace {
// A 32x32, Re 1000 QUICK state after 5 outer iterations (gs_lex). Its p' system (s.pw) is the
// one assembled in the 5th iteration.
struct System {
    Params p;
    std::unique_ptr<State> s;
};
System make_system() {
    System sys;
    sys.p = make_params(ConfigMap{{"N_cells_x", "32"}, {"N_cells_y", "32"}, {"Re", "1000"}});
    sys.s = std::make_unique<State>(sys.p);
    PhaseTimers t;
    for (int k = 0; k < 5; ++k)
        outer_iteration(sys.p, *sys.s, t);
    return sys;
}

template <class F> void interior(const Params& p, F f) {
    for (int i = p.H; i <= p.NX + p.H - 1; ++i)
        for (int j = p.H + 1; j <= p.NY + p.H; ++j)
            f(i, j);
}
double mean(const Params& p, const Field2D& x) {
    double s = 0.0;
    interior(p, [&](int i, int j) { s += x(i, j); });
    return s / (static_cast<double>(p.NX) * p.NY);
}
void remove_mean(const Params& p, Field2D& x) {
    const double m = mean(p, x);
    interior(p, [&](int i, int j) { x(i, j) -= m; });
}
double max_abs(const Params& p, const Field2D& x) {
    double m = 0.0;
    interior(p, [&](int i, int j) { m = std::max(m, std::fabs(x(i, j))); });
    return m;
}
double max_abs_diff(const Params& p, const Field2D& a, const Field2D& b) {
    double m = 0.0;
    interior(p, [&](int i, int j) { m = std::max(m, std::fabs(a(i, j) - b(i, j))); });
    return m;
}
// ||(b - mean b) - A x|| / ||b - mean b||
double true_rel_residual(const Params& p, const PressureWork& w, const Field2D& x) {
    Field2D ax(p.nrows, p.ncols);
    apply_A_p(p, w, x, ax);
    const double mb = mean(p, w.b);
    double rr = 0.0, bb = 0.0;
    interior(p, [&](int i, int j) {
        const double bi = w.b(i, j) - mb;
        rr += (bi - ax(i, j)) * (bi - ax(i, j));
        bb += bi * bi;
    });
    return std::sqrt(rr / bb);
}
} // namespace

TEST_CASE("cg: pressure_tolerance and pressure_maxit keys", "[cg]") {
    Params d = make_params(ConfigMap{});
    REQUIRE(d.pressure_tolerance == 1e-6);
    REQUIRE(d.pressure_maxit == 1000);
    Params p = make_params(ConfigMap{{"pressure_tolerance", "1e-3"}, {"pressure_maxit", "50"}});
    REQUIRE(p.pressure_tolerance == 1e-3);
    REQUIRE(p.pressure_maxit == 50);
    REQUIRE_THROWS(make_params(ConfigMap{{"pressure_tolerance", "0"}}));
    REQUIRE_THROWS(make_params(ConfigMap{{"pressure_maxit", "0"}}));
}

TEST_CASE("cg: reaches the tolerance on a realistic 32x32 p' system", "[cg]") {
    System sys = make_system();
    const Params& p = sys.p;
    State& s = *sys.s;
    const int k = solve_pc_cg(p, s.pw, s.cg);
    std::printf("cg 32x32: %d iterations, rel residual %.3e, max|p'| %.3e\n", k,
                s.cg.last_rel_residual, max_abs(p, s.pw.pc));
    REQUIRE(k > 0);
    REQUIRE(k < p.pressure_maxit);
    REQUIRE(s.cg.last_rel_residual <= p.pressure_tolerance);
    // The recursively updated residual agrees with the true one.
    REQUIRE(true_rel_residual(p, s.pw, s.pw.pc) <= 2 * p.pressure_tolerance);
    REQUIRE(std::fabs(mean(p, s.pw.pc)) <= 1e-13 * max_abs(p, s.pw.pc));
}

TEST_CASE("cg: matches 20000 Gauss-Seidel sweeps (both mean-removed)", "[cg]") {
    System sys = make_system();
    Params p = sys.p;
    State& s = *sys.s;
    p.pressure_tolerance = 1e-12;
    solve_pc_cg(p, s.pw, s.cg);
    REQUIRE(s.cg.last_rel_residual <= 1e-12);
    Field2D x_cg = s.pw.pc;

    // GS on the same system, right-hand side projected to mean zero so the iteration does not
    // drift along the null space.
    PressureWork gw = s.pw;
    const double mb = mean(p, gw.b);
    interior(p, [&](int i, int j) { gw.b(i, j) -= mb; });
    Params pg = p;
    pg.maxit_inner_p = 20000;
    solve_pc_gs_lex(pg, gw);
    remove_mean(p, gw.pc);

    const double d = max_abs_diff(p, x_cg, gw.pc);
    std::printf("cg vs 20000 GS: max abs diff %.3e (max|p'| %.3e)\n", d, max_abs(p, x_cg));
    REQUIRE(max_abs(p, x_cg) > 1e-6);
    REQUIRE(d <= 1e-8);
}

TEST_CASE("cg: error decreases in the A-norm, residual falls, iterate stays mean-free", "[cg]") {
    System sys = make_system();
    Params p = sys.p;
    State& s = *sys.s;
    // Reference solution.
    p.pressure_tolerance = 1e-13;
    solve_pc_cg(p, s.pw, s.cg);
    const Field2D xs = s.pw.pc;

    p.pressure_tolerance = 1e-6;
    std::vector<double> hist;
    s.cg.history = &hist;
    const int K = solve_pc_cg(p, s.pw, s.cg);
    s.cg.history = nullptr;
    REQUIRE(static_cast<int>(hist.size()) == K + 1);
    REQUIRE(hist.front() == 1.0);
    REQUIRE(hist.back() <= 1e-6);
    // Residual falls by orders of magnitude over each third of the run.
    REQUIRE(hist[K / 3] < hist[0]);
    REQUIRE(hist[2 * K / 3] < hist[K / 3]);
    REQUIRE(hist[K] < hist[2 * K / 3]);

    // CG with pressure_maxit = k returns the k-th iterate; its A-norm error is non-increasing,
    // and before the final mean removal it is already mean-free.
    Field2D e(p.nrows, p.ncols), ae(p.nrows, p.ncols);
    double prev = INFINITY;
    for (int k = 1; k <= K; ++k) {
        p.pressure_maxit = k;
        solve_pc_cg(p, s.pw, s.cg);
        interior(p, [&](int i, int j) { e(i, j) = s.pw.pc(i, j) - xs(i, j); });
        apply_A_p(p, s.pw, e, ae);
        double en = 0.0;
        interior(p, [&](int i, int j) { en += e(i, j) * ae(i, j); });
        INFO("k = " << k);
        REQUIRE(en <= prev * (1 + 1e-10));
        prev = en;
        REQUIRE(std::fabs(s.cg.last_mean_removed) <= 1e-12 * max_abs(p, s.pw.pc));
    }
}

TEST_CASE("cg: right-hand side with a non-zero mean", "[cg]") {
    System sys = make_system();
    const Params& p = sys.p;
    State& s = *sys.s;
    solve_pc_cg(p, s.pw, s.cg);
    const Field2D x0 = s.pw.pc;
    const double scale = max_abs(p, s.pw.b);
    interior(p, [&](int i, int j) { s.pw.b(i, j) += 0.5 * scale; });
    const int k = solve_pc_cg(p, s.pw, s.cg);
    REQUIRE(k < p.pressure_maxit);
    REQUIRE(s.cg.last_rel_residual <= p.pressure_tolerance);
    double finite = 0.0;
    interior(p, [&](int i, int j) { finite += std::isfinite(s.pw.pc(i, j)) ? 0.0 : 1.0; });
    REQUIRE(finite == 0.0);
    REQUIRE(std::fabs(mean(p, s.pw.pc)) <= 1e-13 * max_abs(p, s.pw.pc));
    // The constant part is outside the range of A and is projected away.
    REQUIRE(max_abs_diff(p, s.pw.pc, x0) <= 1e-4 * max_abs(p, x0));
}

TEST_CASE("cg: zero right-hand side gives p' = 0", "[cg]") {
    System sys = make_system();
    const Params& p = sys.p;
    State& s = *sys.s;
    s.pw.b.fill(0.0);
    s.pw.pc.fill(1.0);
    REQUIRE(solve_pc_cg(p, s.pw, s.cg) == 0);
    REQUIRE(max_abs(p, s.pw.pc) == 0.0);
}
