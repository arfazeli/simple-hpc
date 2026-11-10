#include "simple/momentum_u.hpp"
#include "simple/momentum_v.hpp"
#include "simple/pressure.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {
Params small_params(const std::string& scheme, int order) {
    ConfigMap c = {{"N_cells_x", "9"},
                   {"N_cells_y", "11"},
                   {"Re", "100"},
                   {"convection_scheme", scheme},
                   {"diffusion_order", std::to_string(order)}};
    return make_params(c);
}

// Deterministic values in [lo, lo + 1).
struct Lcg {
    std::uint64_t s = 12345;
    double next(double lo) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return lo + static_cast<double>(s >> 11) / 9007199254740992.0;
    }
};

void fill(Field2D& f, Lcg& g, double lo) {
    for (int i = 0; i < f.rows(); ++i)
        for (int j = 0; j < f.cols(); ++j)
            f(i, j) = g.next(lo);
}

Coeffs random_coeffs(const Params& p, Lcg& g) {
    Coeffs c(p.nrows, p.ncols);
    for (Field2D* f : {&c.AE, &c.AW, &c.AN, &c.AS, &c.AEE, &c.AWW, &c.ANN, &c.ASS})
        fill(*f, g, -0.5);
    fill(c.AP, g, 4.0);
    return c;
}

bool same(const Field2D& a, const Field2D& b) {
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j)
            if (a(i, j) != b(i, j))
                return false;
    return true;
}

double nb(const Coeffs& c, const Field2D& f, int i, int j) {
    return (c.AE(i, j) * f(i, j + 1) + c.AW(i, j) * f(i, j - 1) + c.AN(i, j) * f(i - 1, j) +
            c.AS(i, j) * f(i + 1, j)) +
           (c.AEE(i, j) * f(i, j + 2) + c.AWW(i, j) * f(i, j - 2) + c.ANN(i, j) * f(i - 2, j) +
            c.ASS(i, j) * f(i + 2, j));
}
} // namespace

TEST_CASE("inner solvers: colourings separate all stencil offsets", "[inner_solvers]") {
    const std::vector<std::pair<int, int>> far9 = {{0, 1}, {0, -1}, {0, 2}, {0, -2},
                                                   {1, 0}, {-1, 0}, {2, 0}, {-2, 0}};
    const std::vector<std::pair<int, int>> near5 = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    REQUIRE(momentum_colours(small_params("QUICK", 4)) == 5);
    REQUIRE(momentum_colours(small_params("QUICK", 2)) == 5);
    REQUIRE(momentum_colours(small_params("UDS", 4)) == 5);
    REQUIRE(momentum_colours(small_params("UDS", 2)) == 2);
    for (auto [ncol, offs] : {std::make_pair(5, far9), std::make_pair(2, near5)})
        for (int i = 2; i < 22; ++i)
            for (int j = 2; j < 22; ++j)
                for (auto [di, dj] : offs) {
                    INFO("ncol " << ncol << " cell " << i << "," << j << " offset " << di << ","
                                 << dj);
                    REQUIRE(momentum_colour(ncol, i, j) != momentum_colour(ncol, i + di, j + dj));
                }
}

TEST_CASE("inner solvers: one u Jacobi sweep equals a reference loop", "[inner_solvers]") {
    Params p = small_params("QUICK", 4);
    Lcg g;
    Coeffs c = random_coeffs(p, g);
    Field2D u(p.nrows, p.ncols), pf(p.nrows, p.ncols), scratch(p.nrows, p.ncols);
    fill(u, g, -0.5);
    fill(pf, g, -0.5);
    const Field2D u0 = u;
    Field2D ref = u;
    const double a = p.relaxation_factor_u;
    for (int i = p.NX + p.H - 1; i >= p.H; --i)
        for (int j = p.H + 1; j <= p.NY + p.H - 1; ++j)
            ref(i, j) = a * (nb(c, u0, i, j) + (pf(i, j) - pf(i, j + 1)) * p.dy) / c.AP(i, j) +
                        (1 - a) * u0(i, j);
    sweep_u_jacobi(p, c, u, pf, scratch);
    REQUIRE(same(u, ref));
}

TEST_CASE("inner solvers: one v Jacobi sweep equals a reference loop", "[inner_solvers]") {
    Params p = small_params("QUICK", 4);
    Lcg g;
    Coeffs c = random_coeffs(p, g);
    Field2D v(p.nrows, p.ncols), pf(p.nrows, p.ncols), scratch(p.nrows, p.ncols);
    fill(v, g, -0.5);
    fill(pf, g, -0.5);
    const Field2D v0 = v;
    Field2D ref = v;
    const double a = p.relaxation_factor_v;
    for (int i = p.NX + p.H - 1; i > p.H; --i)
        for (int j = p.H + 1; j <= p.NY + p.H; ++j)
            ref(i, j) = a * (nb(c, v0, i, j) + (pf(i, j) - pf(i - 1, j)) * p.dx) / c.AP(i, j) +
                        (1 - a) * v0(i, j);
    sweep_v_jacobi(p, c, v, pf, scratch);
    REQUIRE(same(v, ref));
}

TEST_CASE("inner solvers: u multicolour sweep equals a colour-by-colour reference loop",
          "[inner_solvers]") {
    for (auto [scheme, order] : {std::make_pair("QUICK", 4), std::make_pair("UDS", 2)}) {
        Params p = small_params(scheme, order);
        const int ncol = momentum_colours(p);
        Lcg g;
        Coeffs c = random_coeffs(p, g);
        if (ncol == 2)
            for (Field2D* f : {&c.AEE, &c.AWW, &c.ANN, &c.ASS})
                f->fill(0.0);
        Field2D u(p.nrows, p.ncols), pf(p.nrows, p.ncols);
        fill(u, g, -0.5);
        fill(pf, g, -0.5);
        Field2D ref = u;
        const double a = p.relaxation_factor_u;
        // Every interior cell is visited once, in its colour's pass.
        for (int col = 0; col < ncol; ++col)
            for (int i = p.NX + p.H - 1; i >= p.H; --i)
                for (int j = p.H + 1; j <= p.NY + p.H - 1; ++j)
                    if (momentum_colour(ncol, i, j) == col) {
                        const double s =
                            ncol == 2 ? c.AE(i, j) * ref(i, j + 1) + c.AW(i, j) * ref(i, j - 1) +
                                            c.AN(i, j) * ref(i - 1, j) + c.AS(i, j) * ref(i + 1, j)
                                      : nb(c, ref, i, j);
                        ref(i, j) = a * (s + (pf(i, j) - pf(i, j + 1)) * p.dy) / c.AP(i, j) +
                                    (1 - a) * ref(i, j);
                    }
        sweep_u_multicolour(p, c, u, pf);
        INFO("scheme " << scheme);
        REQUIRE(same(u, ref));
    }
}

TEST_CASE("inner solvers: p' damped Jacobi and red-black equal reference loops",
          "[inner_solvers]") {
    Params p = small_params("QUICK", 4);
    p.maxit_inner_p = 3;
    Lcg g;
    PressureWork w(p.nrows, p.ncols);
    for (Field2D* f : {&w.AEP, &w.AWP, &w.ANP, &w.ASP, &w.b})
        fill(*f, g, 0.0);
    for (int i = 0; i < p.nrows; ++i)
        for (int j = 0; j < p.ncols; ++j)
            w.APP(i, j) = w.AEP(i, j) + w.AWP(i, j) + w.ANP(i, j) + w.ASP(i, j);
    const int H = p.H, NX = p.NX, NY = p.NY;
    auto upd = [&](Field2D& pc, const Field2D& src, int i, int j) {
        pc(i, j) = w.b(i, j) / w.APP(i, j);
        pc(i, j) += (w.AEP(i, j) * src(i, j + 1) + w.AWP(i, j) * src(i, j - 1) +
                     w.ASP(i, j) * src(i + 1, j) + w.ANP(i, j) * src(i - 1, j)) /
                    w.APP(i, j);
    };

    Field2D ref(p.nrows, p.ncols);
    for (int k = 0; k < p.maxit_inner_p; ++k) {
        const Field2D old = ref;
        for (int i = NX + H - 1; i >= H; --i)
            for (int j = H + 1; j <= NY + H; ++j) {
                upd(ref, old, i, j);
                ref(i, j) = 0.8 * ref(i, j) + (1 - 0.8) * old(i, j);
            }
    }
    Field2D scratch(p.nrows, p.ncols);
    w.pc.fill(7.0); // the solver must reset p' to 0
    solve_pc_jacobi(p, w, scratch);
    REQUIRE(same(w.pc, ref));

    ref.fill(0.0);
    for (int k = 0; k < p.maxit_inner_p; ++k)
        for (int col = 0; col < 2; ++col)
            for (int i = NX + H - 1; i >= H; --i)
                for (int j = H + 1; j <= NY + H; ++j)
                    if ((i + j) % 2 == col)
                        upd(ref, ref, i, j);
    w.pc.fill(7.0);
    solve_pc_redblack(p, w);
    REQUIRE(same(w.pc, ref));
}
