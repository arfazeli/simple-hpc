#include "simple/decomp.hpp"

#include "simple/multigrid.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SIMPLE_MPI
#include <mpi.h>
#endif

static CommTimes g_comm;
CommTimes& comm_times() { return g_comm; }

namespace {
// Adds the scope's wall time to `acc` (one of g_comm's fields).
struct CommTimer {
    double& acc;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ~CommTimer() {
        acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};
} // namespace

// Part k of `parts` of an index range [0, total) whose interior is n indices starting at `first`:
// interior indices are split in units of `unit` (n % unit == 0) as evenly as possible (the first
// (n / unit) % parts parts get one unit more), the
// first part also owns [0, first) and the last part everything after the interior. At least H
// interior indices per part keeps every halo inside the neighbour's owned cells and the
// boundary-condition rows (H-1, NX+H) and columns (H, NY+H+1) out of other ranks' halos.
static void split(int n, int first, int total, int parts, int k, int unit, int& lo, int& hi) {
    const int base = n / unit / parts, extra = n / unit % parts;
    auto start = [&](int m) { return first + unit * (m * base + std::min(m, extra)); };
    lo = k == 0 ? 0 : start(k);
    hi = k == parts - 1 ? total - 1 : start(k + 1) - 1;
}

void owned_block(const Params& p, int r, int& rlo, int& rhi, int& clo, int& chi) {
    const int unit = 1 << p.mg_split;
    split(p.NX, p.H, p.nrows, p.grid_rows, r / p.grid_cols, unit, rlo, rhi);
    split(p.NY, p.H + 1, p.ncols, p.grid_cols, r % p.grid_cols, unit, clo, chi);
}

void interior_block(const Params& p, int r, int& I0, int& nI, int& J0, int& nJ) {
    int rlo = 0, rhi = 0, clo = 0, chi = 0;
    owned_block(p, r, rlo, rhi, clo, chi);
    I0 = std::max(rlo, p.H) - p.H;
    nI = std::min(rhi, p.NX + p.H - 1) - p.H - I0 + 1;
    J0 = std::max(clo, p.H + 1) - p.H - 1;
    nJ = std::min(chi, p.NY + p.H) - p.H - 1 - J0 + 1;
}

// Decomposed MG levels for a pr x pc grid. A level D >= 1 is valid if every rank owns at least
// one 2^D unit in each direction, at least H level-0 cells, and at least kMgAggMin cells per side
// on level D-1. Coarse units (large D) leave up to one unit of level-0 imbalance (e.g. 3072/3072/
// 2048 rows of 8192 on 3 ranks, 12.5 %, and every rank waits for the largest block); fine units
// balance the blocks but make level D, which every rank gathers and solves, bigger. So pick the
// valid D with the least per-rank work: largest block's level-0 cells + level D's cells (ties to
// the larger D). 0 if no D is valid.
// ponytail: the gathered level is weighted like level-0 cells although only the p' solve touches
// it, so this errs towards coarse units; weight it by the measured solve_p share if that matters.
constexpr int kMgAggMin = 4;
static int mg_split_levels(const Params& p, int pr, int pc) {
    const int nl = static_cast<int>(mg_level_dims(p.NX, p.NY).size());
    int best = 0;
    double best_work = 0.0;
    for (int d = 1; d < nl; ++d) {
        const int nr = p.NX >> d, nc = p.NY >> d; // level d, also the units per direction
        const int ur = nr / pr, uc = nc / pc;     // fewest units on a rank
        if (!(ur >= 1 && uc >= 1 && (ur << d) >= p.H && (uc << d) >= p.H && 2 * ur >= kMgAggMin &&
              2 * uc >= kMgAggMin))
            continue;
        const double big = double((nr + pr - 1) / pr << d) * double((nc + pc - 1) / pc << d);
        const double work = big + double(nr) * nc;
        if (best == 0 || work <= best_work) {
            best = d;
            best_work = work;
        }
    }
    return best;
}

void decompose(Params& p, int rank, int nranks) {
    if (nranks < 1 || rank < 0 || rank >= nranks)
        throw std::runtime_error("decompose: bad rank " + std::to_string(rank) + " of " +
                                 std::to_string(nranks));
    if (nranks > 1) {
        if (p.momentum_solver != MomentumSolver::GsMulticolour ||
            p.pressure_solver == PressureSolver::GsLex || p.backend == Backend::GPU)
            throw std::runtime_error("more than one MPI rank needs momentum_solver = "
                                     "gs_multicolour, a pressure_solver other than gs_lex, and "
                                     "the CPU backend");
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
    p.mg_split = 0;
    if (nranks > 1 && p.pressure_solver == PressureSolver::MG) {
        p.mg_split = mg_split_levels(p, pr, pc);
        if (p.mg_split == 0)
            throw std::runtime_error(
                "decompose: pressure_solver = mg needs two multigrid levels "
                "and " +
                std::to_string(kMgAggMin / 2) + " coarse cells per rank on each side (grid " +
                std::to_string(p.NX) + " x " + std::to_string(p.NY) + ", process grid " +
                std::to_string(pr) + "x" + std::to_string(pc) + ")");
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

CellSums::CellSums(const Params& p, int ia_, int ib_, int ja_, int jb_, int k_)
    : ia(row_from(p, ia_)), ib(row_to(p, ib_)), ja(col_from(p, ja_)), jb(col_to(p, jb_)), k(k_),
      nc(std::max(0, jb - ja + 1)), d_(nullptr),
      rows_(static_cast<std::size_t>(std::max(0, ib - ia + 1)) * k_, 0.0) {
    if (p.grid_cols > 1)
        d_ = cell_buffer(rows_.size() * static_cast<std::size_t>(nc));
}

// Fixed order: global row 0 first, value m of row r at rows[r * k + m]. Rows no rank filled hold
// 0.0, and x + 0.0 == x.
static void add_rows(const std::vector<double>& rows, int k, double* out) {
    for (int m = 0; m < k; ++m) {
        double s = 0.0;
        for (std::size_t r = static_cast<std::size_t>(m); r < rows.size(); r += k)
            s += rows[r];
        out[m] = s;
    }
}

void CellSums::sum(const Params& p, double* out) const {
    CommTimer timer{g_comm.collective};
    const int nr = std::max(0, ib - ia + 1), n = nr * k;
    std::vector<double> part(n, 0.0); // row partials: 0.0, or what the west neighbour reached
    if (!d_) {
        part = rows_; // whole rows, summed by the kernel
    } else {
#ifdef SIMPLE_MPI
        if (p.grid_j > 0)
            MPI_Recv(part.data(), n, MPI_DOUBLE, p.rank - 1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
#pragma omp parallel for schedule(static)
        for (int r = 0; r < nr; ++r)
            for (int m = 0; m < k; ++m) {
                double s = part[static_cast<std::size_t>(r) * k + m];
                const double* c = d_ + (static_cast<std::size_t>(r) * k + m) * nc;
                for (int j = 0; j < nc; ++j)
                    s += c[j];
                part[static_cast<std::size_t>(r) * k + m] = s;
            }
    }
    std::vector<double> rows(static_cast<std::size_t>(p.nrows) * k, 0.0);
    if (p.grid_j < p.grid_cols - 1) {
#ifdef SIMPLE_MPI
        MPI_Send(part.data(), n, MPI_DOUBLE, p.rank + 1, 4, MPI_COMM_WORLD);
#endif
    } else {
        std::copy(part.begin(), part.end(), rows.begin() + std::max(0, ia) * k);
    }
    if (p.nranks == 1)
        return add_rows(rows, k, out);
#ifdef SIMPLE_MPI
    // Every rank receives the full row sums held by the last rank of each grid row, in place.
    std::vector<int> counts(p.nranks, 0), displs(p.nranks, 0);
    for (int r = 0; r < p.nranks; ++r) {
        int rlo = 0, rhi = 0, clo = 0, chi = 0;
        owned_block(p, r, rlo, rhi, clo, chi);
        if (r % p.grid_cols == p.grid_cols - 1) {
            displs[r] = rlo * k;
            counts[r] = (rhi - rlo + 1) * k;
        }
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rows.data(), counts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
    add_rows(rows, k, out);
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

// `depth` columns of `count` rows, one committed datatype per shape (fields and multigrid levels
// have a few shapes each). ponytail: MPI_Type_vector, never freed; manual packing is the
// alternative to measure on the cluster.
static MPI_Datatype column_type(int count, int depth, int stride) {
    static std::map<std::array<int, 3>, MPI_Datatype> types;
    const std::array<int, 3> key{count, depth, stride};
    auto it = types.find(key);
    if (it == types.end()) {
        MPI_Datatype t;
        MPI_Type_vector(count, depth, stride, MPI_DOUBLE, &t);
        MPI_Type_commit(&t);
        it = types.emplace(key, t).first;
    }
    return it->second;
}

void halo_exchange_raw(const Params& p, double* x00, int nr, int nc, int stride, int west,
                       int depth) {
    if (p.nranks == 1)
        return;
    CommTimer timer{g_comm.halo};
    MPI_Request req[4];
    // Tags: 0 south-going, 1 north-going, 2 east-going, 3 west-going.
    if (p.grid_cols > 1) { // east/west: `depth` columns of the owned rows
        const MPI_Datatype t = column_type(nr, depth, stride);
        int k = 0;
        if (p.grid_j > 0) {
            MPI_Irecv(x00 - depth, 1, t, p.rank - 1, 2, MPI_COMM_WORLD, &req[k++]);
            MPI_Isend(x00, 1, t, p.rank - 1, 3, MPI_COMM_WORLD, &req[k++]);
        }
        if (p.grid_j < p.grid_cols - 1) {
            MPI_Irecv(x00 + nc, 1, t, p.rank + 1, 3, MPI_COMM_WORLD, &req[k++]);
            MPI_Isend(x00 + nc - depth, 1, t, p.rank + 1, 2, MPI_COMM_WORLD, &req[k++]);
        }
        MPI_Waitall(k, req, MPI_STATUSES_IGNORE);
    }
    if (p.grid_rows > 1) { // north/south: `depth` whole stored rows (contiguous), corners included
        const int n = depth * stride, up = p.rank - p.grid_cols, down = p.rank + p.grid_cols;
        auto row = [&](int i) { return x00 + static_cast<std::ptrdiff_t>(i) * stride - west; };
        int k = 0;
        if (p.grid_i > 0) { // neighbour above owns the rows just before the first owned one
            MPI_Irecv(row(-depth), n, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &req[k++]);
            MPI_Isend(row(0), n, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &req[k++]);
        }
        if (p.grid_i < p.grid_rows - 1) {
            MPI_Irecv(row(nr), n, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &req[k++]);
            MPI_Isend(row(nr - depth), n, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &req[k++]);
        }
        MPI_Waitall(k, req, MPI_STATUSES_IGNORE);
    }
}

void halo_exchange(const Params& p, Field2D& f) {
    if (p.nranks == 1)
        return;
    halo_exchange_raw(p, &f(p.row_lo, p.col_lo), p.row_hi - p.row_lo + 1, p.col_hi - p.col_lo + 1,
                      f.cols(), p.col_lo - f.col0(), p.H);
}

// One corner-free exchange in flight at a time (halo_post ... halo_wait), master thread only.
static MPI_Request g_req[8];
static int g_nreq = 0;

void halo_post_raw(const Params& p, double* x00, int nr, int nc, int stride, int depth) {
    if (p.nranks == 1)
        return;
    CommTimer timer{g_comm.halo};
    if (g_nreq != 0)
        throw std::logic_error("halo_post: an exchange is already in flight");
    // Tags as in halo_exchange_raw. Every message is owned cells only, so no send buffer overlaps
    // a receive buffer and all four directions go at once.
    auto at = [&](int i, int j) { return x00 + static_cast<std::ptrdiff_t>(i) * stride + j; };
    if (p.grid_cols > 1) {
        const MPI_Datatype t = column_type(nr, depth, stride);
        if (p.grid_j > 0) {
            MPI_Irecv(at(0, -depth), 1, t, p.rank - 1, 2, MPI_COMM_WORLD, &g_req[g_nreq++]);
            MPI_Isend(at(0, 0), 1, t, p.rank - 1, 3, MPI_COMM_WORLD, &g_req[g_nreq++]);
        }
        if (p.grid_j < p.grid_cols - 1) {
            MPI_Irecv(at(0, nc), 1, t, p.rank + 1, 3, MPI_COMM_WORLD, &g_req[g_nreq++]);
            MPI_Isend(at(0, nc - depth), 1, t, p.rank + 1, 2, MPI_COMM_WORLD, &g_req[g_nreq++]);
        }
    }
    if (p.grid_rows > 1) {
        const MPI_Datatype t = column_type(depth, nc, stride); // `depth` rows of nc owned cells
        const int up = p.rank - p.grid_cols, down = p.rank + p.grid_cols;
        if (p.grid_i > 0) {
            MPI_Irecv(at(-depth, 0), 1, t, up, 0, MPI_COMM_WORLD, &g_req[g_nreq++]);
            MPI_Isend(at(0, 0), 1, t, up, 1, MPI_COMM_WORLD, &g_req[g_nreq++]);
        }
        if (p.grid_i < p.grid_rows - 1) {
            MPI_Irecv(at(nr, 0), 1, t, down, 1, MPI_COMM_WORLD, &g_req[g_nreq++]);
            MPI_Isend(at(nr - depth, 0), 1, t, down, 0, MPI_COMM_WORLD, &g_req[g_nreq++]);
        }
    }
}

void halo_post(const Params& p, Field2D& f) {
    if (p.nranks == 1)
        return;
    halo_post_raw(p, &f(p.row_lo, p.col_lo), p.row_hi - p.row_lo + 1, p.col_hi - p.col_lo + 1,
                  f.cols(), p.H);
}

void halo_wait() {
    CommTimer timer{g_comm.halo};
    MPI_Waitall(g_nreq, g_req, MPI_STATUSES_IGNORE);
    g_nreq = 0;
}

void allgather_level(const Params& p, int shift, double* x00, int stride) {
    if (p.nranks == 1)
        return;
    CommTimer timer{g_comm.collective};
    std::vector<int> box(4 * static_cast<std::size_t>(p.nranks)), counts(p.nranks),
        displs(p.nranks);
    int total = 0;
    for (int r = 0; r < p.nranks; ++r) {
        int* b = &box[4 * static_cast<std::size_t>(r)];
        interior_block(p, r, b[0], b[1], b[2], b[3]);
        for (int m = 0; m < 4; ++m)
            b[m] >>= shift; // the split unit is 2^mg_split, so blocks stay aligned
        counts[r] = b[1] * b[3];
        displs[r] = total;
        total += counts[r];
    }
    std::vector<double> all(total);
    double* mine = all.data() + displs[p.rank];
    const int* b = &box[4 * static_cast<std::size_t>(p.rank)];
    for (int I = 0; I < b[1]; ++I)
        for (int J = 0; J < b[3]; ++J)
            *mine++ = x00[static_cast<std::ptrdiff_t>(b[0] + I) * stride + b[2] + J];
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all.data(), counts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
    for (int r = 0; r < p.nranks; ++r) {
        const int* q = &box[4 * static_cast<std::size_t>(r)];
        const double* src = all.data() + displs[r];
        for (int I = 0; I < q[1]; ++I)
            for (int J = 0; J < q[3]; ++J)
                x00[static_cast<std::ptrdiff_t>(q[0] + I) * stride + q[2] + J] = *src++;
    }
}

#else // no MPI: only one rank can exist

void halo_exchange(const Params& p, Field2D&) {
    if (p.nranks != 1)
        throw std::runtime_error("nranks > 1 needs a build with -DSIMPLE_MPI=ON");
}

void halo_exchange_raw(const Params& p, double*, int, int, int, int, int) {
    if (p.nranks != 1)
        throw std::runtime_error("nranks > 1 needs a build with -DSIMPLE_MPI=ON");
}

void halo_post(const Params& p, Field2D&) {
    if (p.nranks != 1)
        throw std::runtime_error("nranks > 1 needs a build with -DSIMPLE_MPI=ON");
}

void halo_post_raw(const Params& p, double*, int, int, int, int) {
    if (p.nranks != 1)
        throw std::runtime_error("nranks > 1 needs a build with -DSIMPLE_MPI=ON");
}

void halo_wait() {}

void allgather_level(const Params& p, int, double*, int) {
    if (p.nranks != 1)
        throw std::runtime_error("nranks > 1 needs a build with -DSIMPLE_MPI=ON");
}

#endif

std::array<Box, 5> overlap_boxes(const Params& p, const Box& k, const Box& own, int reach) {
    // Interior rows [ra, rb], columns [ca, cb]: k shrunk by `reach` from each side of `own` that
    // has a neighbour rank, clamped so the three row bands (and column bands) partition k.
    const bool n = p.grid_i > 0, s = p.grid_i < p.grid_rows - 1;
    const bool w = p.grid_j > 0, e = p.grid_j < p.grid_cols - 1;
    const int ra = std::min(n ? std::max(k.ia, own.ia + reach) : k.ia, k.ib + 1);
    const int rb = std::max(ra - 1, s ? std::min(k.ib, own.ib - reach) : k.ib);
    const int ca = std::min(w ? std::max(k.ja, own.ja + reach) : k.ja, k.jb + 1);
    const int cb = std::max(ca - 1, e ? std::min(k.jb, own.jb - reach) : k.jb);
    return {{{ra, rb, ca, cb},
             {k.ia, ra - 1, k.ja, k.jb},
             {rb + 1, k.ib, k.ja, k.jb},
             {ra, rb, k.ja, ca - 1},
             {ra, rb, cb + 1, k.jb}}};
}
