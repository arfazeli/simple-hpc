#include "simple/params.hpp"

#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <stdexcept>

static Params from(const std::string& s) {
    std::istringstream in(s);
    return make_params(parse_config(in));
}

TEST_CASE("params derived values for a 16x16 grid") {
    auto p = from("N_cells_x = 16\nN_cells_y = 16\nRe = 100\nconvection_scheme = UDS\n");
    REQUIRE(p.nrows == 21);
    REQUIRE(p.ncols == 21);
    REQUIRE(p.dx == 1.0 / 16);
    REQUIRE(p.dy == 1.0 / 16);
    REQUIRE(p.nu == 1.0 / 100);
    REQUIRE(p.convection_scheme == Convection::UDS);
}

TEST_CASE("params defaults match SPEC 3.1") {
    auto p = from("");
    REQUIRE(p.NX == 80);
    REQUIRE(p.NY == 80);
    REQUIRE(p.H == 2);
    REQUIRE(p.Re == 1000.0);
    REQUIRE(p.maxit == 15000);
    REQUIRE(p.diffusion_order == 4);
    REQUIRE(p.convection_scheme == Convection::QUICK);
    REQUIRE(p.output_dir == "results/run");
    REQUIRE(p.relaxation_factor_p == 0.2);
}

TEST_CASE("params enums parse") {
    auto p = from("backend = openmp\nmomentum_solver = jacobi\n"
                  "pressure_solver = cg\n");
    REQUIRE(p.backend == Backend::OpenMP);
    REQUIRE(p.momentum_solver == MomentumSolver::Jacobi);
    REQUIRE(p.pressure_solver == PressureSolver::CG);
}

TEST_CASE("params mode is an unknown key") {
    REQUIRE_THROWS_AS(from("mode = corrected\n"), std::runtime_error);
}

TEST_CASE("params unknown key throws") {
    REQUIRE_THROWS_AS(from("bogus = 1\n"), std::runtime_error);
}

TEST_CASE("params unparsable number throws") {
    REQUIRE_THROWS_AS(from("Re = abc\n"), std::runtime_error);
    REQUIRE_THROWS_AS(from("maxit = 1.5\n"), std::runtime_error);
}

TEST_CASE("params unknown enum throws") {
    REQUIRE_THROWS_AS(from("backend = fast\n"), std::runtime_error);
    REQUIRE_THROWS_AS(from("convection_scheme = CDS\n"), std::runtime_error);
}

TEST_CASE("params halo constraints") {
    REQUIRE_THROWS_AS(from("N_halo_x = 2\nN_halo_y = 3\n"), std::runtime_error);
    REQUIRE_THROWS_AS(from("N_halo_x = 1\nN_halo_y = 1\n"), std::runtime_error);
}

TEST_CASE("params diffusion_order must be 2 or 4") {
    REQUIRE_THROWS_AS(from("diffusion_order = 3\n"), std::runtime_error);
}

TEST_CASE("params Hybrid_switch True is rejected") {
    REQUIRE_THROWS_AS(from("Hybrid_switch = True\n"), std::runtime_error);
}

TEST_CASE("params non-square grid: dx = 1/NY, dy = 1/NX") {
    auto p = from("N_cells_x = 8\nN_cells_y = 16\n");
    REQUIRE(p.dx == 1.0 / 16);
    REQUIRE(p.dy == 1.0 / 8);
}

TEST_CASE("print_params prints derived values") {
    std::ostringstream os;
    print_params(os, from("N_cells_x = 32\nN_cells_y = 32\n"));
    auto s = os.str();
    REQUIRE(s.find("nrows = 37") != std::string::npos);
    REQUIRE(s.find("dx = 0.03125") != std::string::npos);
    REQUIRE(s.find("nu = ") != std::string::npos);
}
