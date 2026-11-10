#include "simple/config.hpp"
#include "simple/multigrid.hpp"
#include "simple/pressure.hpp"
#include "simple/solver.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace {
using Dims = std::vector<std::pair<int, int>>;

// A realistic p' system: an NX x NY QUICK state after 5 gs_lex outer iterations; s.pw holds the
// p' system assembled in the 5th iteration.
struct System {
    Params p;
    std::unique_ptr<State> s;
};
System make_system(int nx, int ny, const char* re) {
    System sys;
    sys.p = make_params(ConfigMap{
        {"N_cells_x", std::to_string(nx)}, {"N_cells_y", std::to_string(ny)}, {"Re", re}});
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
template <class F> void cells(const MGView& L, F f) {
    for (int I = 0; I < L.nx; ++I)
        for (int J = 0; J < L.ny; ++J)
            f(I * L.stride + J);
}
double mean(const Params& p, const Field2D& x) {
    double s = 0.0;
    interior(p, [&](int i, int j) { s += x(i, j); });
    return s / (static_cast<double>(p.NX) * p.NY);
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

TEST_CASE("mg: hierarchy dimensions", "[mg]") {
    REQUIRE(mg_level_dims(80, 80) == Dims{{80, 80}, {40, 40}, {20, 20}, {10, 10}, {5, 5}});
    REQUIRE(mg_level_dims(32, 40) == Dims{{32, 40}, {16, 20}, {8, 10}, {4, 5}});
    REQUIRE(mg_level_dims(64, 64) == Dims{{64, 64}, {32, 32}, {16, 16}, {8, 8}, {4, 4}});
    REQUIRE(mg_level_dims(512, 512) ==
            Dims{{512, 512}, {256, 256}, {128, 128}, {64, 64}, {32, 32}, {16, 16}, {8, 8}, {4, 4}});
    REQUIRE(mg_level_dims(7, 8) == Dims{{7, 8}});
    REQUIRE(mg_level_dims(6, 6) == Dims{{6, 6}});
    // MGWork allocates the same hierarchy.
    MGWork mg(make_params(ConfigMap{{"N_cells_x", "32"}, {"N_cells_y", "40"}}));
    REQUIRE(mg.levels.size() == 4);
    REQUIRE(mg.levels[3].nx == 4);
    REQUIRE(mg.levels[3].ny == 5);
}

TEST_CASE("mg: mg_cycles, mg_pre, mg_post keys", "[mg]") {
    Params d = make_params(ConfigMap{});
    REQUIRE(d.mg_cycles == 1);
    REQUIRE(d.mg_pre == 2);
    REQUIRE(d.mg_post == 2);
    Params p = make_params(ConfigMap{{"mg_cycles", "3"}, {"mg_pre", "1"}, {"mg_post", "4"}});
    REQUIRE(p.mg_cycles == 3);
    REQUIRE(p.mg_pre == 1);
    REQUIRE(p.mg_post == 4);
    REQUIRE_THROWS(make_params(ConfigMap{{"mg_cycles", "0"}}));
    REQUIRE_THROWS(make_params(ConfigMap{{"mg_pre", "-1"}}));
    REQUIRE(d.mg_omega == 1.8);
    REQUIRE(make_params(ConfigMap{{"mg_omega", "1.5"}}).mg_omega == 1.5);
    REQUIRE_THROWS(make_params(ConfigMap{{"mg_omega", "0"}}));
    REQUIRE_THROWS(make_params(ConfigMap{{"mg_omega", "2.5"}}));
}

TEST_CASE("mg: Galerkin coarse operator A_c = R A_f P (32x40 state)", "[mg]") {
    System sys = make_system(32, 40, "400");
    State& s = *sys.s;
    MGWork mg(sys.p);
    mg_build(s.pw, mg);
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (std::size_t l = 1; l < mg.levels.size(); ++l) {
        const MGView F = mg_view(s.pw, mg, static_cast<int>(l - 1));
        const MGView C = mg_view(s.pw, mg, static_cast<int>(l));
        std::vector<double> xc_buf(C.alloc, 0.0), ac_buf(C.alloc, 0.0), rf_buf(C.alloc, 0.0);
        std::vector<double> xf_buf(F.alloc, 0.0), af_buf(F.alloc, 0.0);
        double *xc = xc_buf.data() + C.base, *ac = ac_buf.data() + C.base,
               *rac = rf_buf.data() + C.base;
        double *xf = xf_buf.data() + F.base, *af = af_buf.data() + F.base;
        cells(C, [&](int q) { xc[q] = U(rng); });
        mg_apply(C, xc, ac);
        // P: inject; A_f; R: sum of the 4 fine cells.
        for (int i = 0; i < F.nx; ++i)
            for (int j = 0; j < F.ny; ++j)
                xf[i * F.stride + j] = xc[(i / 2) * C.stride + j / 2];
        mg_apply(F, xf, af);
        for (int I = 0; I < C.nx; ++I)
            for (int J = 0; J < C.ny; ++J) {
                const int f = 2 * I * F.stride + 2 * J;
                rac[I * C.stride + J] = af[f] + af[f + 1] + af[f + F.stride] + af[f + F.stride + 1];
            }
        double dmax = 0.0, amax = 0.0;
        cells(C, [&](int q) {
            dmax = std::max(dmax, std::fabs(ac[q] - rac[q]));
            amax = std::max(amax, std::fabs(ac[q]));
        });
        INFO("level " << l << ": max |A_c x - R A_f P x| " << dmax << ", max |A_c x| " << amax);
        REQUIRE(amax > 0.0);
        REQUIRE(dmax <= 1e-12 * amax);
    }
}

TEST_CASE("mg: coarse rows sum to zero, operator symmetric, wall links zero", "[mg]") {
    System sys = make_system(32, 40, "400");
    State& s = *sys.s;
    MGWork mg(sys.p);
    mg_build(s.pw, mg);
    for (std::size_t l = 1; l < mg.levels.size(); ++l) {
        const MGView C = mg_view(s.pw, mg, static_cast<int>(l));
        const int cs = C.stride;
        INFO("level " << l);
        for (int I = 0; I < C.nx; ++I)
            for (int J = 0; J < C.ny; ++J) {
                const int q = I * cs + J;
                REQUIRE(C.AP[q] > 0.0);
                REQUIRE(std::fabs(C.AP[q] - (C.AE[q] + C.AW[q] + C.AN[q] + C.AS[q])) <=
                        1e-14 * C.AP[q]);
                if (J + 1 < C.ny)
                    REQUIRE(std::fabs(C.AE[q] - C.AW[q + 1]) <= 1e-14 * C.AE[q]);
                else
                    REQUIRE(C.AE[q] == 0.0);
                if (I + 1 < C.nx)
                    REQUIRE(std::fabs(C.AS[q] - C.AN[q + cs]) <= 1e-14 * C.AS[q]);
                else
                    REQUIRE(C.AS[q] == 0.0);
                if (J == 0)
                    REQUIRE(C.AW[q] == 0.0);
                if (I == 0)
                    REQUIRE(C.AN[q] == 0.0);
            }
    }
}

TEST_CASE("mg: V-cycle convergence on a realistic 64x64 p' system, matches cg", "[mg]") {
    System sys = make_system(64, 64, "1000");
    Params p = sys.p;
    State& s = *sys.s;
    MGWork mg(p);
    mg_build(s.pw, mg);
    s.pw.pc.fill(0.0);
    std::vector<double> rel{1.0};
    int k1e8 = -1;
    for (int k = 1; k <= 60; ++k) {
        mg_vcycle(p, s.pw, mg);
        rel.push_back(true_rel_residual(p, s.pw, s.pw.pc));
        if (k1e8 < 0 && rel.back() <= 1e-8)
            k1e8 = k;
    }
    const double factor10 = std::pow(rel[10] / rel[0], 0.1);
    std::printf("mg 64x64: rel residual after 1, 2, 5, 10 cycles %.3e %.3e %.3e %.3e; "
                "average factor over 10 cycles %.3f; 1e-8 after %d cycles; after 60 %.3e; "
                "coarsest sweeps %d\n",
                rel[1], rel[2], rel[5], rel[10], factor10, k1e8, rel[60], mg.last_coarse_sweeps);
    REQUIRE(factor10 < 0.5);
    REQUIRE(k1e8 > 0);
    REQUIRE(k1e8 <= 40);

    // Compare with CG at tolerance 1e-12 (both mean-removed).
    const MGView L0 = mg_view(s.pw, mg, 0);
    mg_remove_mean(mg, L0, L0.x);
    const Field2D x_mg = s.pw.pc;
    p.pressure_tolerance = 1e-12;
    solve_pc_cg(p, s.pw, s.cg);
    REQUIRE(s.cg.last_rel_residual <= 1e-12);
    const double d = max_abs_diff(p, x_mg, s.pw.pc);
    std::printf("mg (60 cycles) vs cg 1e-12: max abs diff %.3e (max|p'| %.3e)\n", d,
                max_abs(p, s.pw.pc));
    REQUIRE(max_abs(p, s.pw.pc) > 1e-6);
    REQUIRE(d <= 1e-8);

    // solve_pc_mg = build, p' = 0, mg_cycles cycles, mean removal.
    p.mg_cycles = 40;
    solve_pc_mg(p, s.pw, mg);
    REQUIRE(true_rel_residual(p, s.pw, s.pw.pc) <= 1e-8);
    REQUIRE(std::fabs(mean(p, s.pw.pc)) <= 1e-13 * max_abs(p, s.pw.pc));
}

TEST_CASE("mg: right-hand side with a non-zero mean does not diverge", "[mg]") {
    System sys = make_system(64, 64, "1000");
    Params p = sys.p;
    State& s = *sys.s;
    MGWork mg(p);
    p.mg_cycles = 30;
    solve_pc_mg(p, s.pw, mg);
    const Field2D x0 = s.pw.pc;
    const double scale = max_abs(p, s.pw.b);
    interior(p, [&](int i, int j) { s.pw.b(i, j) += 0.5 * scale; });
    solve_pc_mg(p, s.pw, mg);
    double nonfinite = 0.0;
    interior(p, [&](int i, int j) { nonfinite += std::isfinite(s.pw.pc(i, j)) ? 0.0 : 1.0; });
    REQUIRE(nonfinite == 0.0);
    const double rr = true_rel_residual(p, s.pw, s.pw.pc);
    const double d = max_abs_diff(p, s.pw.pc, x0);
    std::printf("mg non-zero mean rhs: rel residual (projected b) %.3e, max abs diff to the "
                "mean-free solution %.3e (max|p'| %.3e)\n",
                rr, d, max_abs(p, x0));
    REQUIRE(rr <= 1e-6);
    REQUIRE(std::fabs(mean(p, s.pw.pc)) <= 1e-12 * max_abs(p, s.pw.pc));
    // The constant part is outside the range of A and is projected away.
    REQUIRE(d <= 1e-4 * max_abs(p, x0));
}

TEST_CASE("mg: over-correction omega = 1.8 gives a V-cycle factor < 0.5 at 64^2 and 128^2",
          "[mg]") {
    for (int n : {64, 128}) {
        System sys = make_system(n, n, "1000");
        Params p = sys.p;
        State& s = *sys.s;
        MGWork mg(p);
        mg_build(s.pw, mg);
        double f[2] = {0.0, 0.0};
        const double omegas[2] = {1.0, 1.8};
        for (int k = 0; k < 2; ++k) {
            p.mg_omega = omegas[k];
            s.pw.pc.fill(0.0);
            for (int c = 0; c < 10; ++c)
                mg_vcycle(p, s.pw, mg);
            f[k] = std::pow(true_rel_residual(p, s.pw, s.pw.pc), 0.1);
        }
        std::printf("mg %dx%d: average factor over 10 V-cycles, omega 1.0 %.3f, omega 1.8 %.3f\n",
                    n, n, f[0], f[1]);
        INFO("n = " << n);
        REQUIRE(f[1] < 0.5);
        REQUIRE(f[1] < f[0]);
    }
}
