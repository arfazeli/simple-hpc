#include "simple/decomp.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SIMPLE_MPI
#include <mpi.h>
#endif

// Part k of `parts` of an index range [0, total) whose interior is n indices starting at `first`:
// interior indices are split as evenly as possible (the first n % parts parts get one more), the
// first part also owns [0, first) and the last part everything after the interior. At least H
// interior indices per part keeps every halo inside the neighbour's owned cells and the
// boundary-condition rows (H-1, NX+H) and columns (H, NY+H+1) out of other ranks' halos.
static void split(int n, int first, int total, int parts, int k, int& lo, int& hi) {
    const int base = n / parts, extra = n % parts;
    auto start = [&](int m) { return first + m * base + std::min(m, extra); };
    lo = k == 0 ? 0 : start(k);
    hi = k == parts - 1 ? total - 1 : start(k + 1) - 1;
}

void owned_block(const Params& p, int r, int& rlo, int& rhi, int& clo, int& chi) {
    split(p.NX, p.H, p.nrows, p.grid_rows, r / p.grid_cols, rlo, rhi);
    split(p.NY, p.H + 1, p.ncols, p.grid_cols, r % p.grid_cols, clo, chi);
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
    const int H = p.H;
    auto fits = [&](int pr, int pc) { return p.NX / pr >= H && p.NY / pc >= H; };
    int pr = p.grid_rows_cfg, pc = p.grid_cols_cfg;
    if (pr == 0) { // auto: smallest halo per rank (cells per side), ties to more grid rows
        double best = 0.0;
        for (int r = 1; r <= nranks; ++r) {
            const int c = nranks / r;
            if (r * c != nranks || !fits(r, c))
                continue;
            const double halo =
                (r > 1 ? static_cast<double>(p.NY) / c : 0.0) + (c > 1 ? double(p.NX) / r : 0.0);
            if (pr == 0 || halo <= best) {
                pr = r;
                pc = c;
                best = halo;
            }
        }
        if (pr == 0)
            throw std::runtime_error("decompose: no process grid for " + std::to_string(nranks) +
                                     " ranks leaves at least " + std::to_string(H) +
                                     " interior rows and columns per rank (grid " +
                                     std::to_string(p.NX) + " x " + std::to_string(p.NY) + ")");
    } else if (pr * pc != nranks) {
        throw std::runtime_error("decompose: mpi_grid " + std::to_string(pr) + "x" +
                                 std::to_string(pc) + " needs " + std::to_string(pr * pc) +
                                 " ranks, got " + std::to_string(nranks));
    } else if (!fits(pr, pc)) {
        throw std::runtime_error("decompose: mpi_grid " + std::to_string(pr) + "x" +
                                 std::to_string(pc) + " leaves fewer than " + std::to_string(H) +
                                 " interior rows or columns per rank (grid " +
                                 std::to_string(p.NX) + " x " + std::to_string(p.NY) + ")");
    }
    p.rank = rank;
    p.nranks = nranks;
    p.grid_rows = pr;
    p.grid_cols = pc;
    p.grid_i = rank / pc;
    p.grid_j = rank % pc;
    owned_block(p, rank, p.row_lo, p.row_hi, p.col_lo, p.col_hi);
    p.alloc_row0 = std::max(0, p.row_lo - H);
    p.alloc_rows = std::min(p.nrows - 1, p.row_hi + H) - p.alloc_row0 + 1;
    p.alloc_col0 = std::max(0, p.col_lo - H);
    p.alloc_cols = std::min(p.ncols - 1, p.col_hi + H) - p.alloc_col0 + 1;
}

// ---- CellSums ----

static double* cell_buffer(std::size_t n) {
    static std::vector<double, NoInitAlloc<double>> buf; // ponytail: grows, never shrinks
    if (buf.size() < n)
        buf.resize(n);
    return buf.data();
}

CellSums::CellSums(const Params& p, int ia_, int ib_, int ja_, int jb_)
    : ia(row_from(p, ia_)), ib(row_to(p, ib_)), ja(col_from(p, ja_)), jb(col_to(p, jb_)),
      nc_(std::max(0, jb - ja + 1)), d_(nullptr), rows_(std::max(0, ib - ia + 1), 0.0) {
    if (p.grid_cols > 1)
        d_ = cell_buffer(rows_.size() * static_cast<std::size_t>(nc_));
}

// Fixed order: global row 0 first. Rows no rank filled hold 0.0, and x + 0.0 == x.
static double add_rows(const std::vector<double>& rows) {
    double s = 0.0;
    for (double r : rows)
        s += r;
    return s;
}

double CellSums::sum(const Params& p) const {
    const int nr = std::max(0, ib - ia + 1);
    std::vector<double> part(nr, 0.0); // row partials: 0.0, or what the west neighbour reached
    if (!d_) {
        part = rows_; // whole rows, summed by the kernel
    } else {
#ifdef SIMPLE_MPI
        if (p.grid_j > 0)
            MPI_Recv(part.data(), nr, MPI_DOUBLE, p.rank - 1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
#pragma omp parallel for schedule(static)
        for (int r = 0; r < nr; ++r) {
            double s = part[r];
            const double* c = d_ + static_cast<std::size_t>(r) * nc_;
            for (int j = 0; j < nc_; ++j)
                s += c[j];
            part[r] = s;
        }
    }
    std::vector<double> rows(p.nrows, 0.0);
    if (p.grid_j < p.grid_cols - 1) {
#ifdef SIMPLE_MPI
        MPI_Send(part.data(), nr, MPI_DOUBLE, p.rank + 1, 4, MPI_COMM_WORLD);
#endif
    } else {
        std::copy(part.begin(), part.end(), rows.begin() + std::max(0, ia));
    }
    if (p.nranks == 1)
        return add_rows(rows);
#ifdef SIMPLE_MPI
    // Every rank receives the full row sums held by the last rank of each grid row, in place.
    std::vector<int> counts(p.nranks, 0), displs(p.nranks, 0);
    for (int r = 0; r < p.nranks; ++r) {
        int rlo = 0, rhi = 0, clo = 0, chi = 0;
        owned_block(p, r, rlo, rhi, clo, chi);
        if (r % p.grid_cols == p.grid_cols - 1) {
            displs[r] = rlo;
            counts[r] = rhi - rlo + 1;
        }
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rows.data(), counts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
    return add_rows(rows);
#else
    throw std::runtime_error("nranks > 1 needs a build with -DSIMPLE_MPI=ON");
#endif
}

// ---- gather and halo exchange ----

static void copy_block(const Field2D& from, Field2D& to, int rlo, int rhi, int clo, int chi) {
    for (int i = rlo; i <= rhi; ++i)
        for (int j = clo; j <= chi; ++j)
            to(i, j) = from(i, j);
}

void gather_field(const Params& p, const Field2D& local, Field2D& global) {
    if (p.nranks == 1) {
        copy_block(local, global, 0, p.nrows - 1, 0, p.ncols - 1);
        return;
    }
#ifdef SIMPLE_MPI
    // Each rank sends its owned block packed row by row; rank 0 knows every block's extent.
    std::vector<double> mine;
    for (int i = p.row_lo; i <= p.row_hi; ++i)
        for (int j = p.col_lo; j <= p.col_hi; ++j)
            mine.push_back(local(i, j));
    std::vector<int> counts(p.nranks), displs(p.nranks);
    std::vector<int> box(4 * static_cast<std::size_t>(p.nranks));
    int total = 0;
    for (int r = 0; r < p.nranks; ++r) {
        int* b = &box[4 * static_cast<std::size_t>(r)];
        owned_block(p, r, b[0], b[1], b[2], b[3]);
        counts[r] = (b[1] - b[0] + 1) * (b[3] - b[2] + 1);
        displs[r] = total;
        total += counts[r];
    }
    std::vector<double> all(p.rank == 0 ? total : 0);
    MPI_Gatherv(mine.data(), static_cast<int>(mine.size()), MPI_DOUBLE, all.data(), counts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (p.rank != 0)
        return;
    for (int r = 0; r < p.nranks; ++r) {
        const int* b = &box[4 * static_cast<std::size_t>(r)];
        const double* src = all.data() + displs[r];
        for (int i = b[0]; i <= b[1]; ++i)
            for (int j = b[2]; j <= b[3]; ++j)
                global(i, j) = *src++;
    }
#else
    (void)local;
    (void)global;
    throw std::runtime_error("nranks > 1 needs a build with -DSIMPLE_MPI=ON");
#endif
}

#ifdef SIMPLE_MPI

// H columns of the owned rows, one strided datatype per field shape (all fields of a rank share
// it). ponytail: MPI_Type_vector; manual packing is the alternative to measure on the cluster.
static MPI_Datatype column_type(int count, int H, int stride) {
    static MPI_Datatype t = MPI_DATATYPE_NULL;
    static int key[3] = {-1, -1, -1};
    if (count != key[0] || H != key[1] || stride != key[2]) {
        if (t != MPI_DATATYPE_NULL)
            MPI_Type_free(&t);
        MPI_Type_vector(count, H, stride, MPI_DOUBLE, &t);
        MPI_Type_commit(&t);
        key[0] = count;
        key[1] = H;
        key[2] = stride;
    }
    return t;
}

void halo_exchange(const Params& p, Field2D& f) {
    if (p.nranks == 1)
        return;
    const int H = p.H;
    MPI_Request req[4];
    // Tags: 0 south-going, 1 north-going, 2 east-going, 3 west-going.
    if (p.grid_cols > 1) { // east/west: H columns of the owned rows
        const MPI_Datatype t = column_type(p.row_hi - p.row_lo + 1, H, f.cols());
        const int i = p.row_lo;
        int k = 0;
        if (p.grid_j > 0) {
            MPI_Irecv(&f(i, p.col_lo - H), 1, t, p.rank - 1, 2, MPI_COMM_WORLD, &req[k++]);
            MPI_Isend(&f(i, p.col_lo), 1, t, p.rank - 1, 3, MPI_COMM_WORLD, &req[k++]);
        }
        if (p.grid_j < p.grid_cols - 1) {
            MPI_Irecv(&f(i, p.col_hi + 1), 1, t, p.rank + 1, 3, MPI_COMM_WORLD, &req[k++]);
            MPI_Isend(&f(i, p.col_hi - H + 1), 1, t, p.rank + 1, 2, MPI_COMM_WORLD, &req[k++]);
        }
        MPI_Waitall(k, req, MPI_STATUSES_IGNORE);
    }
    if (p.grid_rows > 1) { // north/south: H full stored rows (contiguous), corners included
        const int n = H * f.cols(), up = p.rank - p.grid_cols, down = p.rank + p.grid_cols;
        auto row = [&](int i) { return &f(i, f.col0()); };
        int k = 0;
        if (p.grid_i > 0) { // neighbour above owns the rows just before row_lo
            MPI_Irecv(row(p.row_lo - H), n, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &req[k++]);
            MPI_Isend(row(p.row_lo), n, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &req[k++]);
        }
        if (p.grid_i < p.grid_rows - 1) {
            MPI_Irecv(row(p.row_hi + 1), n, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &req[k++]);
            MPI_Isend(row(p.row_hi - H + 1), n, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &req[k++]);
        }
        MPI_Waitall(k, req, MPI_STATUSES_IGNORE);
    }
}

#else // no MPI: only one rank can exist

void halo_exchange(const Params& p, Field2D&) {
    if (p.nranks != 1)
        throw std::runtime_error("nranks > 1 needs a build with -DSIMPLE_MPI=ON");
}

#endif
