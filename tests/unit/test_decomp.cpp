#include "simple/decomp.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

static Params grid(int nx, int ny) {
    std::istringstream in("N_cells_x = " + std::to_string(nx) +
                          "\nN_cells_y = " + std::to_string(ny) +
                          "\nmomentum_solver = gs_multicolour\npressure_solver = gs_redblack\n");
    return make_params(parse_config(in));
}

TEST_CASE("decompose: every global row has exactly one owner, halos stay in the grid") {
    for (int nranks : {1, 2, 3, 4, 7}) {
        const Params g = grid(32, 40);
        int next = 0; // first row not yet owned
        for (int r = 0; r < nranks; ++r) {
            Params p = g;
            decompose(p, r, nranks);
            REQUIRE(p.row_lo == next);
            REQUIRE(p.row_hi >= p.row_lo);
            next = p.row_hi + 1;
            if (r > 0)
                REQUIRE(p.row_hi - p.row_lo + 1 >= p.H); // >= H interior rows per rank
            REQUIRE(p.alloc_row0 == std::max(0, p.row_lo - p.H));
            REQUIRE(p.alloc_row0 + p.alloc_rows - 1 == std::min(p.nrows - 1, p.row_hi + p.H));
        }
        REQUIRE(next == g.nrows);
    }
}

TEST_CASE("decompose: one rank owns the whole grid") {
    Params p = grid(16, 16);
    decompose(p, 0, 1);
    REQUIRE(p.row_lo == 0);
    REQUIRE(p.row_hi == p.nrows - 1);
    REQUIRE(p.alloc_row0 == 0);
    REQUIRE(p.alloc_rows == p.nrows);
}

TEST_CASE("decompose: rejects too many ranks and solvers without an MPI path") {
    Params p = grid(8, 8);
    REQUIRE_THROWS_AS(decompose(p, 0, 5), std::runtime_error); // 8 rows / 5 ranks < H
    std::istringstream in("N_cells_x = 32\nN_cells_y = 32\n"); // gs_lex defaults
    Params q = make_params(parse_config(in));
    REQUIRE_NOTHROW(decompose(q, 0, 1));
    REQUIRE_THROWS_AS(decompose(q, 0, 2), std::runtime_error);
}

TEST_CASE("Field2D with row0 addresses global rows") {
    Field2D f(3, 4, 0.0, 10);
    f(10, 0) = 1.0;
    f(12, 3) = 2.0;
    REQUIRE(f.row0() == 10);
    REQUIRE(f.data()[0] == 1.0);
    REQUIRE(f.data()[11] == 2.0);
}
