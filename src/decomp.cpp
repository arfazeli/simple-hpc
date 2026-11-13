#include "simple/decomp.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SIMPLE_MPI
#include <mpi.h>
#endif

void owned_rows(const Params& p, int r, int& lo, int& hi) {
    // Interior rows of rank r start at H + r*base + min(r, extra); the first `extra` ranks get one
    // more row. At least H rows per rank (checked in decompose) keeps every halo inside the
    // neighbour's owned rows and the boundary-condition rows (H-1, NX+H) out of other ranks' halos.
    const int H = p.H, n = p.nranks, base = p.NX / n, extra = p.NX % n;
    auto first = [&](int k) { return H + k * base + std::min(k, extra); };
    lo = r == 0 ? 0 : first(r);
    hi = r == n - 1 ? p.nrows - 1 : first(r + 1) - 1;
}

void decompose(Params& p, int rank, int nranks) {
    if (nranks < 1 || rank < 0 || rank >= nranks)
        throw std::runtime_error("decompose: bad rank " + std::to_string(rank) + " of " +
                                 std::to_string(nranks));
    if (nranks > 1) {
        if (p.momentum_solver != MomentumSolver::GsMulticolour ||
            p.pressure_solver != PressureSolver::GsRedBlack || p.backend == Backend::GPU)
            throw std::runtime_error("more than one MPI rank needs momentum_solver = "
                                     "gs_multicolour and pressure_solver = gs_redblack on the CPU");
    }
    const int H = p.H, base = p.NX / nranks;
    if (base < H)
        throw std::runtime_error("decompose: " + std::to_string(nranks) +
                                 " ranks leave fewer than " + std::to_string(H) +
                                 " interior rows per rank (NX = " + std::to_string(p.NX) + ")");
    p.rank = rank;
    p.nranks = nranks;
    owned_rows(p, rank, p.row_lo, p.row_hi);
    p.alloc_row0 = std::max(0, p.row_lo - H);
    p.alloc_rows = std::min(p.nrows - 1, p.row_hi + H) - p.alloc_row0 + 1;
}

// Fixed order: global row 0 first. Rows a kernel does not cover hold 0.0, and x + 0.0 == x.
static double add_rows(const std::vector<double>& rows) {
    double s = 0.0;
    for (double r : rows)
        s += r;
    return s;
}

#ifdef SIMPLE_MPI

void halo_exchange(const Params& p, Field2D& f) {
    if (p.nranks == 1)
        return;
    const int H = p.H, n = H * f.cols();
    auto row = [&](int i) { return &f(i, 0); };
    MPI_Request req[4];
    int k = 0;
    if (p.rank > 0) { // neighbour above owns the rows just before row_lo
        MPI_Irecv(row(p.row_lo - H), n, MPI_DOUBLE, p.rank - 1, 0, MPI_COMM_WORLD, &req[k++]);
        MPI_Isend(row(p.row_lo), n, MPI_DOUBLE, p.rank - 1, 1, MPI_COMM_WORLD, &req[k++]);
    }
    if (p.rank < p.nranks - 1) {
        MPI_Irecv(row(p.row_hi + 1), n, MPI_DOUBLE, p.rank + 1, 1, MPI_COMM_WORLD, &req[k++]);
        MPI_Isend(row(p.row_hi - H + 1), n, MPI_DOUBLE, p.rank + 1, 0, MPI_COMM_WORLD, &req[k++]);
    }
    MPI_Waitall(k, req, MPI_STATUSES_IGNORE);
}

double sum_rows(const Params& p, std::vector<double>& rows) {
    if (p.nranks > 1) { // every rank receives all owned row sums, in place
        std::vector<int> counts(p.nranks), displs(p.nranks);
        for (int r = 0; r < p.nranks; ++r) {
            int lo = 0, hi = 0;
            owned_rows(p, r, lo, hi);
            displs[r] = lo;
            counts[r] = hi - lo + 1;
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rows.data(), counts.data(),
                       displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    return add_rows(rows);
}

void gather_rows(const Params& p, const Field2D& local, Field2D& global) {
    const int cols = local.cols();
    const int mine = (std::min(p.row_hi, p.nrows - 1) - p.row_lo + 1) * cols;
    if (p.nranks == 1) {
        std::copy(local.data(), local.data() + mine, global.data());
        return;
    }
    std::vector<int> counts(p.nranks), displs(p.nranks);
    const int lo = p.row_lo;
    MPI_Gather(&mine, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&lo, 1, MPI_INT, displs.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    for (int& d : displs)
        d *= cols;
    MPI_Gatherv(local.data() + static_cast<std::size_t>(p.row_lo - local.row0()) * cols, mine,
                MPI_DOUBLE, p.rank == 0 ? global.data() : nullptr, counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

#else // no MPI: only one rank can exist

void halo_exchange(const Params& p, Field2D&) {
    if (p.nranks != 1)
        throw std::runtime_error("nranks > 1 needs a build with -DSIMPLE_MPI=ON");
}

double sum_rows(const Params& p, std::vector<double>& rows) {
    if (p.nranks != 1)
        throw std::runtime_error("nranks > 1 needs a build with -DSIMPLE_MPI=ON");
    return add_rows(rows);
}

void gather_rows(const Params& p, const Field2D& local, Field2D& global) {
    if (p.nranks != 1)
        throw std::runtime_error("nranks > 1 needs a build with -DSIMPLE_MPI=ON");
    const int mine = (std::min(p.row_hi, p.nrows - 1) - p.row_lo + 1) * local.cols();
    std::copy(local.data(), local.data() + mine, global.data());
}

#endif
