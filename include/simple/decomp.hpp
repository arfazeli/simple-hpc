#pragma once
#include "simple/field2d.hpp"
#include "simple/params.hpp"

#include <cstddef>
#include <vector>

// MPI block decomposition over a Pr x Pc process grid (rank = grid_i * Pc + grid_j; Pc = 1 gives
// row slabs). The NX interior rows [H, NX+H-1] are split as evenly as possible over the Pr grid
// rows (grid row 0 at the top, next to the lid) and the NY interior columns [H+1, NY+H] over the
// Pc grid columns; the first and last parts also own the rows / columns outside the interior, so
// every global cell has exactly one owner. Arrays store the owned block plus H halo cells on each
// side (clamped to the grid) and keep global indices (Field2D row0 / col0), so the kernels only
// clamp their loops (row_from / row_to, col_from / col_to).
//
// The grid is p.grid_rows_cfg x p.grid_cols_cfg (config mpi_grid), or, when that is 0 (auto), the
// factor pair of nranks with the smallest per-rank halo; ties go to more grid rows. Throws if a
// rank would own fewer than H interior rows or columns, or if nranks > 1 with a solver that has
// no MPI path yet. Pure arithmetic: no MPI calls, so it is unit-tested in every build.
void decompose(Params& p, int rank, int nranks);

// Global rows [rlo, rhi] and columns [clo, chi] owned by rank r (the split decompose() uses).
void owned_block(const Params& p, int r, int& rlo, int& rhi, int& clo, int& chi);

// Fill f's halo from the neighbouring ranks' owned cells, H deep. East/west first (the owned rows),
// then north/south over the full stored width, so the corner halos (read by the staggered
// interpolation in assemble_u/v) arrive with the second phase. No-op on one rank.
// Call outside OpenMP parallel regions or from the master thread only (MPI_THREAD_FUNNELED).
void halo_exchange(const Params& p, Field2D& f);

// Deterministic global sum of one value per cell over a kernel's global range [ia, ib] x [ja, jb].
// A kernel row looks like this (one thread per row, j ascending):
//     double s = 0.0, *c = cs.cells(i);
//     for (j = cs.ja; j <= cs.jb; ++j) { v = ...; s += v; if (c) c[j - cs.ja] = v; }
//     cs.row(i, s);
// sum() adds every row j ascending from 0.0, then the rows in ascending global order: same bits
// for any process grid or thread count. With one grid column a rank holds whole rows, so the
// kernel's own row sums are used and cells() is nullptr. Otherwise the cells are kept and sum()
// passes row partials west to east, each grid column adding its cells in order.
// Construct and call sum() outside OpenMP parallel regions; cells() and row() are safe inside.
class CellSums {
  public:
    CellSums(const Params& p, int ia, int ib, int ja, int jb);
    double* cells(int i) {
        return d_ ? d_ + static_cast<std::size_t>(i - ia) * static_cast<std::size_t>(nc_) : nullptr;
    }
    void row(int i, double s) { rows_[static_cast<std::size_t>(i - ia)] = s; }
    double sum(const Params& p) const;
    const int ia, ib, ja, jb; // the range clamped to this rank's block

  private:
    int nc_;
    double* d_; // cells (more than one grid column only); shared buffer, kernels run one at a time
    std::vector<double> rows_;
};

// Rank 0 receives every rank's owned block of `local` into `global` (nrows x ncols, row0 = col0 =
// 0). `global` is only touched on rank 0. With one rank it is a plain copy.
void gather_field(const Params& p, const Field2D& local, Field2D& global);
