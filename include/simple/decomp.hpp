#pragma once
#include "simple/field2d.hpp"
#include "simple/params.hpp"

#include <array>
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

// Interior cells of rank r in level-0 multigrid indices (I = i - H, J = j - H - 1): rows
// [I0, I0 + nI) x columns [J0, J0 + nJ).
void interior_block(const Params& p, int r, int& I0, int& nI, int& J0, int& nJ);

// Fill f's halo from the neighbouring ranks' owned cells, H deep. East/west first (the owned rows),
// then north/south over the full stored width, so the corner halos (read by the staggered
// interpolation in assemble_u/v) arrive with the second phase. No-op on one rank.
// Call outside OpenMP parallel regions or from the master thread only (MPI_THREAD_FUNNELED).
void halo_exchange(const Params& p, Field2D& f);
// Same exchange on a raw array: x00 is the rank's first owned cell, nr x nc owned cells, rows
// `stride` apart, `depth` halo cells on each side; north/south messages carry `depth` whole stored
// rows, which start `west` cells before x00. Neighbours from the same process grid.
void halo_exchange_raw(const Params& p, double* x00, int nr, int nc, int stride, int west,
                       int depth);
// Non-blocking exchange for the sweeps, which read only straight-line neighbours: all four
// directions at once, owned cells only, corners not refreshed. halo_post starts it, halo_wait
// finishes it; one exchange in flight at a time. Master thread only, like halo_exchange.
void halo_post(const Params& p, Field2D& f);
void halo_post_raw(const Params& p, double* x00, int nr, int nc, int stride, int depth);
void halo_wait();

// Inclusive cell box (empty when ia > ib or ja > jb).
struct Box {
    int ia, ib, ja, jb;
};
// Kernel box k split for overlapping a halo exchange with compute: [0] is the interior (cells at
// least `reach` from every side of the owned box `own` that has a neighbour rank, so it reads no
// halo), [1..4] the top, bottom, left and right strips. Together they cover k exactly once.
std::array<Box, 5> overlap_boxes(const Params& p, const Box& k, const Box& own, int reach);

// How a coloured sweep refreshes halos between colours.
enum class Halo { None, Blocking, Overlap };
inline Halo halo_mode(const Params& p) {
    return p.nranks == 1 ? Halo::None : p.halo_overlap ? Halo::Overlap : Halo::Blocking;
}

// `sweeps` x `colours` passes of a coloured sweep; call from every thread of an OpenMP team.
// pass(c, box) updates colour c's cells in box under `omp for nowait`. Cells of one colour are
// independent, so splitting a pass into boxes gives the same bits.
//   None:     pass over k, barrier.
//   Blocking: pass over k, then full() (halo_exchange, corners included) after every colour.
//   Overlap:  post() the previous colour's cells, pass over the interior bx[0] while they travel,
//             halo_wait(), then the strips bx[1..4]; after the last colour full(), so the caller
//             sees the same halos (corners included) as with Blocking.
// full() and post() run on the master thread (MPI_THREAD_FUNNELED).
template <class Pass, class Full, class Post>
void colour_sweeps(Halo mode, int sweeps, int colours, const Box& k, const std::array<Box, 5>& bx,
                   const Pass& pass, const Full& full, const Post& post) {
    for (int s = 0; s < sweeps; ++s)
        for (int c = 0; c < colours; ++c) {
            if (mode == Halo::Overlap && (s > 0 || c > 0)) {
#pragma omp master
                post();
                pass(c, bx[0]);
#pragma omp master
                halo_wait();
#pragma omp barrier
                for (int m = 1; m < 5; ++m)
                    pass(c, bx[m]);
            } else {
                pass(c, k);
            }
#pragma omp barrier
            if (mode == Halo::Blocking ||
                (mode == Halo::Overlap && s == sweeps - 1 && c == colours - 1)) {
#pragma omp master
                full();
#pragma omp barrier
            }
        }
}

// Multigrid level `shift` stored whole on every rank: x00 is its interior cell (0, 0), rows
// `stride` apart. Each rank's block (interior_block >> shift) is sent to every rank, in place.
// No-op on one rank.
void allgather_level(const Params& p, int shift, double* x00, int stride);

// Deterministic global sum of one value per cell over a kernel's global range [ia, ib] x [ja, jb].
// A kernel row looks like this (one thread per row, j ascending):
//     double s = 0.0, *c = cs.cells(i);
//     for (j = cs.ja; j <= cs.jb; ++j) { v = ...; s += v; if (c) c[j - cs.ja] = v; }
//     cs.row(i, s);
// With k values per cell (k > 1), value m of cell j goes to c[m * cs.nc + j - cs.ja], the row
// passes its k sums with row(i, s[k]), and sum(p, out) fills out[k]; one collective for all k.
// sum() adds every row j ascending from 0.0, then the rows in ascending global order: same bits
// for any process grid or thread count. With one grid column a rank holds whole rows, so the
// kernel's own row sums are used and cells() is nullptr. Otherwise the cells are kept and sum()
// passes row partials west to east, each grid column adding its cells in order.
// Construct and call sum() outside OpenMP parallel regions; cells() and row() are safe inside.
class CellSums {
  public:
    CellSums(const Params& p, int ia, int ib, int ja, int jb, int k = 1);
    double* cells(int i) {
        return d_ ? d_ + static_cast<std::size_t>(i - ia) * static_cast<std::size_t>(k * nc)
                  : nullptr;
    }
    void row(int i, double s) { rows_[static_cast<std::size_t>(i - ia)] = s; }
    void row(int i, const double* s) {
        for (int m = 0; m < k; ++m)
            rows_[static_cast<std::size_t>(i - ia) * k + m] = s[m];
    }
    double sum(const Params& p) const {
        double s = 0.0;
        sum(p, &s);
        return s;
    }
    void sum(const Params& p, double* out) const;
    const int ia, ib, ja, jb; // the range clamped to this rank's block
    const int k, nc;          // values per cell, columns in the clamped range

  private:
    double* d_; // cells (more than one grid column only); shared buffer, kernels run one at a time
    std::vector<double> rows_; // k sums per row
};

// Rank 0 receives every rank's owned block of `local` into `global` (nrows x ncols, row0 = col0 =
// 0). `global` is only touched on rank 0. With one rank it is a plain copy.
void gather_field(const Params& p, const Field2D& local, Field2D& global);
