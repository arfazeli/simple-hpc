#include "simple/config.hpp"
#include "simple/io.hpp"
#include "simple/solver.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

// u on the vertical centreline (average of the two middle columns), top row first.
static std::vector<double> centreline(const Params& p, const State& s) {
    Field2D ux, uy;
    cell_centre(p, s.u, s.v, ux, uy);
    std::vector<double> r(p.NX);
    for (int i = 0; i < p.NX; ++i)
        r[i] = (ux(i, p.NY / 2 - 1) + ux(i, p.NY / 2)) / 2;
    return r;
}

namespace {
struct Run {
    Params p;
    std::unique_ptr<State> s;
    bool converged = false;
};
Run run_case(int nx, int ny) {
    ConfigMap c = {{"N_cells_x", std::to_string(nx)},
                   {"N_cells_y", std::to_string(ny)},
                   {"Re", "100"},
                   {"convection_scheme", "UDS"},
                   {"diffusion_order", "2"},
                   {"residual_normalisation", "running_max"},
                   {"error_tolerance", "1e-5"}};
    Run r;
    r.p = make_params(c);
    r.s = std::make_unique<State>(r.p);
    r.converged = run_simple(r.p, *r.s).converged;
    return r;
}
} // namespace

TEST_CASE("non-square 32x48 matches 48x48 centreline", "[regression]") {
    Run a = run_case(32, 48);
    Run b = run_case(48, 48);
    REQUIRE(a.converged);
    REQUIRE(b.converged);
    std::vector<double> ca = centreline(a.p, *a.s), cb = centreline(b.p, *b.s);
    double d = 0.0;
    for (int i = 0; i < 32; ++i) {
        double t = (i + 0.5) * 48.0 / 32.0 - 0.5;
        int k = std::min(std::max(static_cast<int>(std::floor(t)), 0), 46);
        double w = t - k;
        double ib = (1 - w) * cb[k] + w * cb[k + 1];
        d = std::max(d, std::fabs(ca[i] - ib));
    }
    std::printf("non-square: centreline max abs diff %.3e\n", d);
    REQUIRE(d <= 5e-2);
}
