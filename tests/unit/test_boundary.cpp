#include "simple/momentum_u.hpp"
#include "simple/momentum_v.hpp"
#include "simple/params.hpp"
#include "simple/solver.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <stdexcept>

static Params make(int nx, int ny, const ConfigMap& extra = {}) {
    ConfigMap c = {{"N_cells_x", std::to_string(nx)}, {"N_cells_y", std::to_string(ny)}};
    for (auto& kv : extra)
        c[kv.first] = kv.second;
    return make_params(c);
}

TEST_CASE("params: non-square spacing and residual_normalisation key", "[boundary]") {
    Params p = make(16, 24);
    REQUIRE(p.dx == 1.0 / 24);
    REQUIRE(p.dy == 1.0 / 16);
    REQUIRE(p.residual_normalisation == ResidualNorm::Absolute);
    Params r = make(16, 16, {{"residual_normalisation", "running_max"}});
    REQUIRE(r.residual_normalisation == ResidualNorm::RunningMax);
}

TEST_CASE("boundary conditions: ghost cells mirror the interior", "[boundary]") {
    Params p = make(6, 8);
    const int H = p.H;
    Field2D u(p.nrows, p.ncols), v(p.nrows, p.ncols);
    for (int i = 0; i < p.nrows; ++i)
        for (int j = 0; j < p.ncols; ++j)
            u(i, j) = v(i, j) = 100 * i + j;
    Field2D u0 = u, v0 = v;
    apply_bc_u(p, u);
    apply_bc_v(p, v);
    for (int j = H + 1; j <= p.NY + H; ++j) {
        REQUIRE(u(p.NX + H, j) == -u0(p.NX + H - 1, j));
        REQUIRE(u(H - 1, j) == 2 - u0(H, j));
        // outer halo rows untouched
        REQUIRE(u(p.NX + H + 1, j) == u0(p.NX + H + 1, j));
        REQUIRE(u(p.NX + 2 * H, j) == u0(p.NX + 2 * H, j));
    }
    for (int i = H; i < p.NX + H; ++i) {
        REQUIRE(v(i, p.NY + H + 1) == -v0(i, p.NY + H));
        REQUIRE(v(i, H) == -v0(i, H + 1));
    }
}

TEST_CASE("run_simple stops at maxit", "[boundary]") {
    Params p = make(8, 8,
                    {{"maxit", "5"},
                     {"error_tolerance", "0"},
                     {"convection_scheme", "UDS"},
                     {"diffusion_order", "2"}});
    State s(p);
    REQUIRE(run_simple(p, s).iterations == 5);
}

TEST_CASE("normalisation: res_v = 0 at iteration 1 is safe", "[boundary]") {
    Params p = make(8, 8,
                    {{"maxit", "3"},
                     {"residual_normalisation", "running_max"},
                     {"convection_scheme", "UDS"},
                     {"diffusion_order", "2"}});
    State s(p);
    RunResult r = run_simple(p, s);
    REQUIRE(r.log[0][2] == 0.0);
    for (auto& row : r.log)
        for (double x : row)
            REQUIRE(std::isfinite(x));
}
