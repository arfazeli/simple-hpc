#pragma once
#include "simple/field2d.hpp"
#include "simple/params.hpp"

#include <vector>

// MPI slab decomposition by rows. The NX interior rows [H, NX+H-1] are split as evenly as possible
// over the ranks in order (rank 0 gets the top rows, next to the lid); rank 0 also owns the halo
// rows above them and the last rank the rows below, so every global row has exactly one owner.
// Arrays store the owned rows plus H halo rows on each side (clamped to the grid) and keep global
// indices (Field2D row0), so the kernels only clamp their row loops (row_from / row_to).
//
// Sets p.rank, p.nranks, p.row_lo, p.row_hi, p.alloc_row0, p.alloc_rows. Throws if a rank would
// own fewer than H interior rows, or if nranks > 1 with a solver that has no MPI path yet.
// Pure arithmetic: no MPI calls, so it is unit-tested in every build.
void decompose(Params& p, int rank, int nranks);

// Fill f's halo rows from the neighbouring ranks' owned rows (H rows each way). No-op on one rank.
// Call outside OpenMP parallel regions or from the master thread only (MPI_THREAD_FUNNELED).
void halo_exchange(const Params& p, Field2D& f);

// Global rows [lo, hi] owned by rank r of p.nranks (the same split decompose() uses).
void owned_rows(const Params& p, int r, int& lo, int& hi);

// Deterministic global sum. rows has p.nrows entries: each rank fills its owned rows with that
// row's sum (j ascending) and leaves the others at 0.0. Every rank gets all rows (Allgatherv), then
// adds them in global row order, so the result is the same bits for any rank or thread count.
double sum_rows(const Params& p, std::vector<double>& rows);

// Rank 0 receives every rank's owned rows of `local` into `global` (nrows x ncols, row0 0).
// `global` is only touched on rank 0. With one rank it is a plain copy.
void gather_rows(const Params& p, const Field2D& local, Field2D& global);
