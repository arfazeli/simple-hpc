// T32: Jacobi batches with alternating buffers are bit-identical to copy-per-sweep Jacobi.
#include "simple/momentum_u.hpp"
#include "simple/momentum_v.hpp"
#include "simple/pressure.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace {
Params params() {
    return make_params({{"N_cells_x", "13"}, {"N_cells_y", "17"}, {"Re", "100"},
                        {"convection_scheme", "QUICK"}, {"diffusion_order", "4"}});
}

struct Lcg {
    std::uint64_t s = 987654321;
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

Coeffs coeffs(const Params& p, Lcg& g) {
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
} // namespace

TEST_CASE("jacobi buffers: u and v batches equal repeated single sweeps", "[jacobi]") {
    const Params p = params();
    for (int n : {1, 2, 3, 4}) {
        INFO("nsweeps " << n);
        Lcg g;
        const Coeffs c = coeffs(p, g);
        Field2D u(p.nrows, p.ncols), pf(p.nrows, p.ncols), scratch(p.nrows, p.ncols);
        fill(u, g, -0.5);
        fill(pf, g, -0.5);
        scratch.fill(3.0); // contents of the scratch buffer must not matter
        Field2D ref = u, v = u, vref = u;
        for (int k = 0; k < n; ++k) { // single sweeps: a full copy before each one
            sweep_u_jacobi(p, c, ref, pf, scratch);
            sweep_v_jacobi(p, c, vref, pf, scratch);
        }
        scratch.fill(-2.0);
        sweep_u_jacobi_batch(p, c, u, pf, scratch, n);
        REQUIRE(same(u, ref));
        sweep_v_jacobi_batch(p, c, v, pf, scratch, n);
        REQUIRE(same(v, vref));
    }
}

TEST_CASE("jacobi buffers: p' damped Jacobi equals copy-per-sweep loop, odd and even counts",
          "[jacobi]") {
    Params p = params();
    const int H = p.H, NX = p.NX, NY = p.NY;
    for (int n : {7, 8}) {
        INFO("maxit_inner_p " << n);
        p.maxit_inner_p = n;
        Lcg g;
        PressureWork w(p.nrows, p.ncols);
        for (Field2D* f : {&w.AEP, &w.AWP, &w.ANP, &w.ASP, &w.b})
            fill(*f, g, 0.0);
        for (int i = 0; i < p.nrows; ++i)
            for (int j = 0; j < p.ncols; ++j)
                w.APP(i, j) = w.AEP(i, j) + w.AWP(i, j) + w.ANP(i, j) + w.ASP(i, j);
        Field2D ref(p.nrows, p.ncols);
        for (int k = 0; k < n; ++k) {
            const Field2D old = ref;
            for (int i = NX + H - 1; i >= H; --i)
                for (int j = H + 1; j <= NY + H; ++j) {
                    ref(i, j) = w.b(i, j) / w.APP(i, j);
                    ref(i, j) += (w.AEP(i, j) * old(i, j + 1) + w.AWP(i, j) * old(i, j - 1) +
                                  w.ASP(i, j) * old(i + 1, j) + w.ANP(i, j) * old(i - 1, j)) /
                                 w.APP(i, j);
                    ref(i, j) = 0.8 * ref(i, j) + (1 - 0.8) * old(i, j);
                }
        }
        Field2D scratch(p.nrows, p.ncols, 5.0);
        w.pc.fill(7.0);
        solve_pc_jacobi(p, w, scratch);
        REQUIRE(same(w.pc, ref));
    }
}
