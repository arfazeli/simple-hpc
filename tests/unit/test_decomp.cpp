#include "simple/decomp.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static Params grid(int nx, int ny, const std::string& extra = "") {
    std::istringstream in(
        "N_cells_x = " + std::to_string(nx) + "\nN_cells_y = " + std::to_string(ny) +
        "\nmomentum_solver = gs_multicolour\npressure_solver = gs_redblack\n" + extra);
    return make_params(parse_config(in));
}

TEST_CASE("decompose: every global cell has exactly one owner, halos stay in the grid") {
    const int shapes[][2] = {{1, 1}, {2, 1}, {3, 1}, {4, 1}, {7, 1}, {1, 2},
                             {1, 3}, {1, 4}, {2, 2}, {2, 3}, {3, 2}, {4, 2}};
    for (auto& s : shapes) {
        const int R = s[0], C = s[1];
        const Params g = grid(32, 40, "mpi_grid = " + std::to_string(R) + "x" + std::to_string(C));
        std::vector<int> owner(static_cast<std::size_t>(g.nrows) * g.ncols, 0);
        for (int r = 0; r < R * C; ++r) {
            Params p = g;
            decompose(p, r, R * C);
            REQUIRE(p.grid_rows == R);
            REQUIRE(p.grid_cols == C);
            REQUIRE(p.grid_i * C + p.grid_j == r);
            for (int i = p.row_lo; i <= p.row_hi; ++i)
                for (int j = p.col_lo; j <= p.col_hi; ++j)
                    ++owner[static_cast<std::size_t>(i) * g.ncols + j];
            if (p.grid_i >
                0) // >= H interior rows / columns, so BC rows / columns stay out of halos
                REQUIRE(p.row_hi - p.row_lo + 1 >= p.H);
            if (p.grid_j > 0)
                REQUIRE(p.col_hi - p.col_lo + 1 >= p.H);
            REQUIRE(p.alloc_row0 == std::max(0, p.row_lo - p.H));
            REQUIRE(p.alloc_row0 + p.alloc_rows - 1 == std::min(p.nrows - 1, p.row_hi + p.H));
            REQUIRE(p.alloc_col0 == std::max(0, p.col_lo - p.H));
            REQUIRE(p.alloc_col0 + p.alloc_cols - 1 == std::min(p.ncols - 1, p.col_hi + p.H));
            if (C > 1) { // the v ghost columns H and NY+H+1 are owned by the edge grid columns
                REQUIRE((p.col_lo <= p.H) == (p.grid_j == 0));
                REQUIRE((p.col_hi >= p.NY + p.H + 1) == (p.grid_j == C - 1));
            }
        }
        REQUIRE(std::all_of(owner.begin(), owner.end(), [](int n) { return n == 1; }));
    }
}

TEST_CASE("decompose: auto grid has the smallest halo, ties go to slabs") {
    auto shape = [](int nx, int ny, int n) {
        Params p = grid(nx, ny, "");
        decompose(p, 0, n);
        return std::to_string(p.grid_rows) + "x" + std::to_string(p.grid_cols);
    };
    REQUIRE(shape(64, 64, 4) == "4x1"); // 2x2 halo 32 + 32 ties 4x1's 64
    REQUIRE(shape(32, 40, 4) == "1x4"); // 32 rows per side, against 2x2's 20 + 16 and 4x1's 40
    REQUIRE(shape(64, 64, 8) == "4x2"); // 32 + 16 < 64
    REQUIRE(shape(64, 64, 9) == "3x3");
    REQUIRE(shape(8, 64, 4) == "1x4"); // short, wide grid: column split
}

TEST_CASE("decompose: one rank owns the whole grid") {
    Params p = grid(16, 16);
    decompose(p, 0, 1);
    REQUIRE(p.row_lo == 0);
    REQUIRE(p.row_hi == p.nrows - 1);
    REQUIRE(p.alloc_row0 == 0);
    REQUIRE(p.alloc_rows == p.nrows);
    REQUIRE(p.col_lo == 0);
    REQUIRE(p.col_hi == p.ncols - 1);
    REQUIRE(p.alloc_cols == p.ncols);
}

TEST_CASE("decompose: rejects too many ranks and solvers without an MPI path") {
    Params p = grid(8, 8);
    REQUIRE_THROWS_AS(decompose(p, 0, 5), std::runtime_error); // 5 = 5x1 or 1x5, both < H
    Params b = grid(8, 8, "mpi_grid = 2x2");
    REQUIRE_THROWS_AS(decompose(b, 0, 3), std::runtime_error); // 2x2 needs 4 ranks
    Params c = grid(8, 8, "mpi_grid = 1x8");
    REQUIRE_THROWS_AS(decompose(c, 0, 8), std::runtime_error); // 8 columns / 8 < H
    std::istringstream in("N_cells_x = 32\nN_cells_y = 32\n"); // gs_lex defaults
    Params q = make_params(parse_config(in));
    REQUIRE_NOTHROW(decompose(q, 0, 1));
    REQUIRE_THROWS_AS(decompose(q, 0, 2), std::runtime_error);
}

static Params mg_grid(int nx, int ny, const std::string& mpi_grid) {
    std::istringstream in(
        "N_cells_x = " + std::to_string(nx) + "\nN_cells_y = " + std::to_string(ny) +
        "\nmomentum_solver = gs_multicolour\npressure_solver = mg\nmpi_grid = " + mpi_grid + "\n");
    return make_params(parse_config(in));
}

TEST_CASE("decompose: mg splits in aligned units and picks the decomposed levels") {
    // 32x40: levels 32x40, 16x20, 8x10, 4x5. 3x1: units of 4 rows -> 12 / 12 / 8 interior rows;
    // a rank keeps >= 4 cells per side on level split-1. 2x2: split 3 is valid but gives 16x24
    // blocks (40 = 5 units of 8 over 2 columns), split 2 even 16x20 blocks for a level of 8x10
    // cells instead of 4x5: 320 + 80 < 384 + 20, so 2.
    struct Case {
        int R, C, split;
    } cases[] = {{3, 1, 2}, {2, 2, 2}, {4, 1, 2}, {1, 4, 2}, {6, 1, 1}, {2, 3, 2}};
    for (const Case& c : cases) {
        const Params g = mg_grid(32, 40, std::to_string(c.R) + "x" + std::to_string(c.C));
        std::vector<int> rows;
        for (int r = 0; r < c.R * c.C; ++r) {
            Params p = g;
            decompose(p, r, c.R * c.C);
            REQUIRE(p.mg_split == c.split);
            int I0 = 0, nI = 0, J0 = 0, nJ = 0;
            interior_block(p, r, I0, nI, J0, nJ);
            const int unit = 1 << c.split;
            REQUIRE(I0 % unit == 0);
            REQUIRE(nI % unit == 0);
            REQUIRE(J0 % unit == 0);
            REQUIRE(nJ % unit == 0);
            if (r % c.C == 0)
                rows.push_back(nI);
        }
        if (c.R == 3)
            REQUIRE(rows == std::vector<int>{12, 12, 8});
    }
    Params one = mg_grid(32, 40, "auto");
    decompose(one, 0, 1);
    REQUIRE(one.mg_split == 0);         // one rank: nothing gathered, single-cell split
    Params flat = mg_grid(6, 6, "2x1"); // one mg level: nothing to gather onto
    REQUIRE_THROWS_AS(decompose(flat, 0, 2), std::runtime_error);
}

TEST_CASE("decompose: every p' solver but gs_lex has an MPI path") {
    for (const char* ps : {"gs_redblack", "jacobi", "cg", "mg"}) {
        std::istringstream in(std::string("N_cells_x = 32\nN_cells_y = 32\n"
                                          "momentum_solver = gs_multicolour\npressure_solver = ") +
                              ps + "\n");
        Params p = make_params(parse_config(in));
        REQUIRE_NOTHROW(decompose(p, 1, 2));
    }
}

TEST_CASE("overlap_boxes: interior and strips cover the kernel box once; interior reads no halo") {
    const int shapes[][2] = {{1, 1}, {2, 1}, {3, 1}, {1, 3}, {2, 2}, {3, 3}, {4, 2}};
    for (auto& sh : shapes)
        for (int reach : {1, 2, 4}) {
            const int R = sh[0], C = sh[1];
            const Params g = grid(24, 20, "mpi_grid = " + std::to_string(R) + "x" + std::to_string(C));
            for (int r = 0; r < R * C; ++r) {
                Params p = g;
                decompose(p, r, R * C);
                const Box k{row_from(p, p.H), row_to(p, p.NX + p.H - 1), col_from(p, p.H + 1),
                            col_to(p, p.NY + p.H)};
                const Box own{p.row_lo, p.row_hi, p.col_lo, p.col_hi};
                const auto bx = overlap_boxes(p, k, own, reach);
                std::vector<int> hits(static_cast<std::size_t>(p.nrows) * p.ncols, 0);
                for (const Box& b : bx)
                    for (int i = b.ia; i <= b.ib; ++i)
                        for (int j = b.ja; j <= b.jb; ++j)
                            ++hits[static_cast<std::size_t>(i) * p.ncols + j];
                for (int i = 0; i < p.nrows; ++i)
                    for (int j = 0; j < p.ncols; ++j) {
                        const bool in = i >= k.ia && i <= k.ib && j >= k.ja && j <= k.jb;
                        REQUIRE(hits[static_cast<std::size_t>(i) * p.ncols + j] == (in ? 1 : 0));
                    }
                const Box& b = bx[0]; // reach cells beyond the interior stay owned (or non-halo)
                if (b.ia <= b.ib && b.ja <= b.jb) {
                    if (p.grid_i > 0)
                        REQUIRE(b.ia - reach >= own.ia);
                    if (p.grid_i < R - 1)
                        REQUIRE(b.ib + reach <= own.ib);
                    if (p.grid_j > 0)
                        REQUIRE(b.ja - reach >= own.ja);
                    if (p.grid_j < C - 1)
                        REQUIRE(b.jb + reach <= own.jb);
                }
                if (R * C == 1)
                    REQUIRE((b.ia == k.ia && b.ib == k.ib && b.ja == k.ja && b.jb == k.jb));
            }
        }
}

TEST_CASE("Field2D with row0 and col0 addresses global cells") {
    Field2D f(3, 4, 0.0, 10, 20);
    f(10, 20) = 1.0;
    f(12, 23) = 2.0;
    REQUIRE(f.row0() == 10);
    REQUIRE(f.col0() == 20);
    REQUIRE(f.data()[0] == 1.0);
    REQUIRE(f.data()[11] == 2.0);
}

TEST_CASE("CellSums with k values per cell: k independent fixed-order sums") {
    std::istringstream in("N_cells_x = 8\nN_cells_y = 6\n");
    const Params p = make_params(parse_config(in));
    const int H = p.H;
    CellSums cs(p, H, p.NX + H - 1, H + 1, p.NY + H, 2);
    double ref[2] = {0.0, 0.0};
    for (int i = cs.ia; i <= cs.ib; ++i) {
        double s[2] = {0.0, 0.0}, r[2] = {0.0, 0.0};
        for (int j = cs.ja; j <= cs.jb; ++j) {
            s[0] += 0.1 * i + j;
            s[1] += 1.0 / (i + j);
            r[0] += 0.1 * i + j;
            r[1] += 1.0 / (i + j);
        }
        cs.row(i, s);
        ref[0] += r[0];
        ref[1] += r[1];
    }
    double out[2];
    cs.sum(p, out);
    REQUIRE(out[0] == ref[0]);
    REQUIRE(out[1] == ref[1]);
}

TEST_CASE("CellSums on one rank: rows j ascending from 0.0, then rows in order") {
    Params p = grid(8, 8, "");
    decompose(p, 0, 1);
    CellSums cs(p, 2, 4, 3, 5);
    REQUIRE(cs.cells(2) == nullptr); // one grid column: the kernel's row sums are used
    double want = 0.0;
    for (int i = 2; i <= 4; ++i) {
        double row = 0.0;
        for (int j = 3; j <= 5; ++j)
            row += 0.1 * i + 0.01 * j;
        cs.row(i, row);
        want += row;
    }
    REQUIRE(cs.sum(p) == want); // exact
}
