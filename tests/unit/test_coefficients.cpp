#include "simple/coefficients.hpp"

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdio>

using Catch::Approx;

static Params mk(Convection c, int order) {
    Params p;
    p.NX = 8;
    p.NY = 8;
    p.H = 2;
    p.N_halo_x = 2;
    p.N_halo_y = 2;
    p.dx = 0.125;
    p.dy = 0.25;
    p.nu = 0.01;
    p.convection_scheme = c;
    p.diffusion_order = order;
    return p;
}

TEST_CASE("coefficients: boundary band") {
    Params p = mk(Convection::QUICK, 4);
    REQUIRE(in_boundary_band(p, 2, 5));
    REQUIRE(in_boundary_band(p, 9, 5));
    REQUIRE(in_boundary_band(p, 5, 3));
    REQUIRE(in_boundary_band(p, 5, 10));
    REQUIRE_FALSE(in_boundary_band(p, 5, 2));
    REQUIRE_FALSE(in_boundary_band(p, 5, 5));
}

TEST_CASE("coefficients: UDS") {
    Params p = mk(Convection::UDS, 2);
    auto c = convective_coefficients(p, 1.0, 2.0, 3.0, 4.0, 5, 5);
    REQUIRE(c.AE == 0.0);
    REQUIRE(c.AW == Approx(0.25 * 2.0));
    REQUIRE(c.AN == 0.0);
    REQUIRE(c.AS == Approx(0.125 * 4.0));
    REQUIRE(c.AP == Approx(1.0));
    c = convective_coefficients(p, -1.0, -2.0, -3.0, -4.0, 5, 5);
    REQUIRE(c.AE == Approx(0.25));
    REQUIRE(c.AW == 0.0);
    REQUIRE(c.AN == Approx(0.375));
    REQUIRE(c.AS == 0.0);
    REQUIRE(c.AP == Approx(0.625));
}

TEST_CASE("coefficients: QUICK interior and band fallback") {
    Params p = mk(Convection::QUICK, 4);
    auto c = convective_coefficients(p, 1.0, 1.0, 1.0, 1.0, 5, 5);
    REQUIRE(c.AE == Approx(-0.25 * 0.375));
    REQUIRE(c.AW == Approx(-0.25 * (-0.125 - 0.75)));
    REQUIRE(c.AN == Approx(-0.125 * 0.375));
    REQUIRE(c.AS == Approx(-0.125 * (-0.125 - 0.75)));
    REQUIRE(c.AEE == 0.0);
    REQUIRE(c.AWW == Approx(-0.125 * 0.25));
    REQUIRE(c.ASS == Approx(-0.125 * 0.125));
    REQUIRE(c.AP == Approx(c.AE + c.AW + c.AN + c.AS + c.AWW + c.ASS));
    c = convective_coefficients(p, -1.0, -1.0, -1.0, -1.0, 5, 5);
    REQUIRE(c.AE == Approx(-0.25 * (-0.75 - 0.125)));
    REQUIRE(c.AEE == Approx(-0.125 * 0.25));
    REQUIRE(c.ANN == Approx(-0.125 * 0.125));
    REQUIRE(c.AWW == 0.0);
    c = convective_coefficients(p, -1.0, 2.0, -3.0, 4.0, 2, 5);
    REQUIRE(c.AE == Approx(0.25));
    REQUIRE(c.AW == Approx(0.5));
    REQUIRE(c.AEE == 0.0);
    REQUIRE(c.AP == Approx(0.25 + 0.5 + 0.375 + 0.5));
}

TEST_CASE("coefficients: diffusion and sum") {
    Params p = mk(Convection::UDS, 2);
    auto d = diffusive_coefficients(p, 5, 5);
    REQUIRE(d.AE == Approx(0.25 * 0.01 / 0.125));
    REQUIRE(d.AN == Approx(0.125 * 0.01 / 0.25));
    REQUIRE(d.AP == Approx(0.01 * (2 * 0.125 / 0.25 + 2 * 0.25 / 0.125)));
    p.diffusion_order = 4;
    d = diffusive_coefficients(p, 5, 5);
    REQUIRE(d.AE == Approx(4.0 / 3 * 0.25 * 0.01 / 0.125));
    REQUIRE(d.AEE == Approx(-1.0 / 12 * 0.25 * 0.01 / 0.125));
    REQUIRE(d.ANN == Approx(-1.0 / 12 * 0.125 * 0.01 / 0.25));
    REQUIRE(d.AP == Approx(0.01 * (2.5 * 0.125 / 0.25 + 2.5 * 0.25 / 0.125)));
    d = diffusive_coefficients(p, 5, 3);
    REQUIRE(d.AE == Approx(0.25 * 0.01 / 0.125));
    REQUIRE(d.AEE == 0.0);
    d = diffusive_coefficients(p, 5, 2);
    REQUIRE(d.AEE != 0.0);
    auto s = Coef9{1, 2, 3, 4, 5, 6, 7, 8, 9} + Coef9{1, 1, 1, 1, 1, 1, 1, 1, 1};
    REQUIRE(s.AP == 10.0);
    REQUIRE(s.AE == 2.0);
}
