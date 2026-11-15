#include "simple/multigrid.hpp"

#include "simple/decomp.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <initializer_list>

namespace {
constexpr double kCoarseTol = 1e-10; // relative residual of the coarsest-level solve
constexpr int kCoarseMaxSweeps = 200;

bool par(const MGView& L, const MGWork& mg) { return L.nx * L.ny >= mg.parallel_min_cells; }

// Fixed-order sum of f(q) over the interior: each row summed in order (rows in parallel), then
// the row sums in order. Independent of the thread count.
template <class F> double det_sum(const MGView& L, MGWork& mg, F f) {
    double* rows = mg.rowsum.data();
    const bool pr = par(L, mg);
    (void)pr; // unused without OpenMP
#pragma omp parallel for schedule(static) if (pr)
    for (int k_ = 0; k_ < L.nx; ++k_) { // rows descending (T38)
        const int I = L.nx - 1 - k_;
        double s = 0.0;
        for (int J = 0; J < L.ny; ++J)
            s += f(I * L.stride + J);
        rows[I] = s;
    }
    double s = 0.0;
    for (int I = 0; I < L.nx; ++I)
        s += rows[I];
    return s;
}

// `sweeps` red-black Gauss-Seidel sweeps on L.x, two-statement update of SPEC 4.3 step 4. `ex`
// (if set) refreshes L.x's halo after each colour; it runs on the master thread (MPI FUNNELED).
using Exchange = std::function<void()>;
void smooth(const MGView& L, int sweeps, bool pr, const Exchange& ex = {}) {
    (void)pr;
    const int par0 = L.I0 + L.J0; // colours in global indices
    const int s = L.stride, nx = L.nx, ny = L.ny;
    double* x = L.x;
    const double *AE = L.AE, *AW = L.AW, *AN = L.AN, *AS = L.AS, *AP = L.AP, *b = L.b;
#pragma omp parallel if (pr)
    for (int k = 0; k < sweeps; ++k)
        for (int c = 0; c < 2; ++c) {
#pragma omp for schedule(static)
            for (int k_ = 0; k_ < nx; ++k_) { // rows descending (T38)
                const int I = nx - 1 - k_;
                for (int J = (I + par0 + c) & 1; J < ny; J += 2) {
                    const int q = I * s + J;
                    x[q] = b[q] / AP[q];
                    x[q] += (AE[q] * x[q + 1] + AW[q] * x[q - 1] + AS[q] * x[q + s] +
                             AN[q] * x[q - s]) /
                            AP[q];
                }
            }
            if (ex) {
#pragma omp master
                ex();
#pragma omp barrier
            }
        }
}

// L.r = L.b - A L.x
void residual(const MGView& L, bool pr) {
    (void)pr;
    const int s = L.stride, nx = L.nx, ny = L.ny;
    const double *x = L.x, *AE = L.AE, *AW = L.AW, *AN = L.AN, *AS = L.AS, *AP = L.AP, *b = L.b;
    double* r = L.r;
#pragma omp parallel for schedule(static) if (pr)
    for (int k_ = 0; k_ < nx; ++k_) { // rows descending (T38)
        const int I = nx - 1 - k_;
        for (int J = 0; J < ny; ++J) {
            const int q = I * s + J;
            r[q] = b[q] - (AP[q] * x[q] - (AE[q] * x[q + 1] + AW[q] * x[q - 1] + AS[q] * x[q + s] +
                                           AN[q] * x[q - s]));
        }
    }
}

// C.b = R F.r (sum of the 4 fine residuals), C.x = 0, on the coarse cells of F's block.
void restrict_residual(const MGView& F, const MGView& C, bool pr) {
    (void)pr;
    const int fs = F.stride, cs = C.stride, nx = F.nx / 2, ny = F.ny / 2;
    const int c0 = (F.I0 / 2 - C.I0) * cs + F.J0 / 2 - C.J0; // F's block inside C
#pragma omp parallel for schedule(static) if (pr)
    for (int k_ = 0; k_ < nx; ++k_) { // rows descending (T38)
        const int I = nx - 1 - k_;
        for (int J = 0; J < ny; ++J) {
            const int f = 2 * I * fs + 2 * J;
            C.b[c0 + I * cs + J] = F.r[f] + F.r[f + 1] + F.r[f + fs] + F.r[f + fs + 1];
            C.x[c0 + I * cs + J] = 0.0;
        }
    }
}

// F.x += omega P C.x (injection to the 4 fine cells, over-corrected by omega).
void prolong(const MGView& C, const MGView& F, double omega, bool pr) {
    (void)pr;
    const int fs = F.stride, cs = C.stride;
    const double* cx = C.x + (F.I0 / 2 - C.I0) * cs + F.J0 / 2 - C.J0; // F's block (I0, J0 even)
#pragma omp parallel for schedule(static) if (pr)
    for (int k_ = 0; k_ < F.nx; ++k_) { // rows descending (T38)
        const int i = F.nx - 1 - k_;
        for (int j = 0; j < F.ny; ++j)
            F.x[i * fs + j] += omega * cx[(i / 2) * cs + j / 2];
    }
}

// Coarsest level: remove the mean of b, red-black sweeps until ||b - A x|| < kCoarseTol ||b|| or
// kCoarseMaxSweeps, remove the mean of x. All sums fixed-order.
void coarse_solve(const MGView& L, MGWork& mg) {
    const bool pr = par(L, mg);
    const double* r = L.r;
    double* b = L.b;
    mg_remove_mean(mg, L, b);
    const double bb = det_sum(L, mg, [b](int q) { return b[q] * b[q]; });
    int sweeps = 0;
    if (bb > 0.0) {
        const double bn = std::sqrt(bb);
        while (sweeps < kCoarseMaxSweeps) {
            smooth(L, 1, pr);
            ++sweeps;
            residual(L, pr);
            const double rr = det_sum(L, mg, [r](int q) { return r[q] * r[q]; });
            if (std::sqrt(rr) < kCoarseTol * bn)
                break;
        }
    }
    mg_remove_mean(mg, L, L.x);
    mg.last_coarse_sweeps = sweeps;
}
} // namespace

std::vector<std::pair<int, int>> mg_level_dims(int nx, int ny) {
    std::vector<std::pair<int, int>> d{{nx, ny}};
    while (nx % 2 == 0 && ny % 2 == 0 && nx / 2 >= 4 && ny / 2 >= 4) {
        nx /= 2;
        ny /= 2;
        d.emplace_back(nx, ny);
    }
    return d;
}

MGWork::MGWork(const Params& p)
    : H(p.H), nrows(p.alloc_rows > 0 ? p.alloc_rows : p.nrows),
      ncols(p.alloc_cols > 0 ? p.alloc_cols : p.ncols), row0(p.alloc_row0), col0(p.alloc_col0),
      prm(p), rowsum(p.NX) {
    const auto dims = mg_level_dims(p.NX, p.NY);
    const int nl = static_cast<int>(dims.size());
    split = p.nranks > 1 ? p.mg_split : nl;
    int I0 = 0, nI = 0, J0 = 0, nJ = 0;
    interior_block(p, p.rank, I0, nI, J0, nJ);
    levels.resize(dims.size());
    for (int l = 0; l < nl; ++l) {
        MGLevel& L = levels[l];
        if (l < split) { // this rank's block (the whole level on one rank)
            L.nx = nI >> l;
            L.ny = nJ >> l;
            L.I0 = I0 >> l;
            L.J0 = J0 >> l;
        } else {
            L.nx = dims[l].first;
            L.ny = dims[l].second;
        }
        if (l == 0) {
            L.r.resize(static_cast<std::size_t>(nrows) * ncols);
            first_touch_fill(L.r.data(), nrows, ncols, 0.0);
            continue;
        }
        const std::size_t n = static_cast<std::size_t>(L.nx + 2) * (L.ny + 2);
        for (auto* a : {&L.AE, &L.AW, &L.AN, &L.AS, &L.AP, &L.b, &L.x, &L.r}) {
            a->resize(n);
            first_touch_fill(a->data(), L.nx + 2, L.ny + 2, 0.0);
        }
    }
}

MGView mg_view(PressureWork& w, MGWork& mg, int level) {
    MGLevel& L = mg.levels[level];
    MGView v;
    v.nx = L.nx;
    v.ny = L.ny;
    v.I0 = L.I0;
    v.J0 = L.J0;
    if (level == 0) {
        v.stride = mg.ncols;
        v.alloc = static_cast<std::size_t>(mg.nrows) * mg.ncols;
        v.base = static_cast<std::ptrdiff_t>(mg.H + L.I0 - mg.row0) * mg.ncols + mg.H + 1 + L.J0 -
                 mg.col0;
        v.AE = w.AEP.data() + v.base;
        v.AW = w.AWP.data() + v.base;
        v.AN = w.ANP.data() + v.base;
        v.AS = w.ASP.data() + v.base;
        v.AP = w.APP.data() + v.base;
        v.b = w.b.data() + v.base;
        v.x = w.pc.data() + v.base;
    } else {
        v.stride = L.ny + 2;
        v.alloc = static_cast<std::size_t>(L.nx + 2) * (L.ny + 2);
        v.base = v.stride + 1;
        v.AE = L.AE.data() + v.base;
        v.AW = L.AW.data() + v.base;
        v.AN = L.AN.data() + v.base;
        v.AS = L.AS.data() + v.base;
        v.AP = L.AP.data() + v.base;
        v.b = L.b.data() + v.base;
        v.x = L.x.data() + v.base;
    }
    v.r = L.r.data() + v.base;
    return v;
}

void mg_apply(const MGView& L, const double* x, double* out) {
    const int s = L.stride;
    for (int I = 0; I < L.nx; ++I)
        for (int J = 0; J < L.ny; ++J) {
            const int q = I * s + J;
            out[q] = L.AP[q] * x[q] - (L.AE[q] * x[q + 1] + L.AW[q] * x[q - 1] +
                                       L.AS[q] * x[q + s] + L.AN[q] * x[q - s]);
        }
}

void mg_build(PressureWork& w, MGWork& mg) {
    for (std::size_t l = 1; l < mg.levels.size(); ++l) {
        const MGView F = mg_view(w, mg, static_cast<int>(l - 1));
        const MGView C = mg_view(w, mg, static_cast<int>(l));
        const int fs = F.stride, cs = C.stride, nx = F.nx / 2, ny = F.ny / 2;
        const int c0 = (F.I0 / 2 - C.I0) * cs + F.J0 / 2 - C.J0; // F's block inside C
        const bool pr = par(C, mg);
        (void)pr;
#pragma omp parallel for schedule(static) if (pr)
        for (int k_ = 0; k_ < nx; ++k_) { // rows descending (T38)
            const int I = nx - 1 - k_;
            for (int J = 0; J < ny; ++J) {
                const int f = 2 * I * fs + 2 * J, c = c0 + I * cs + J;
                // Fine cells of the block: f (NW), f+1 (NE), f+fs (SW), f+fs+1 (SE).
                const double ae = F.AE[f + 1] + F.AE[f + fs + 1];
                const double aw = F.AW[f] + F.AW[f + fs];
                const double an = F.AN[f] + F.AN[f + 1];
                const double as = F.AS[f + fs] + F.AS[f + fs + 1];
                C.AE[c] = ae;
                C.AW[c] = aw;
                C.AN[c] = an;
                C.AS[c] = as;
                C.AP[c] = ae + aw + an + as;
            }
        }
        if (static_cast<int>(l) == mg.split) // first level held whole: gather every block
            for (double* a : {C.AE, C.AW, C.AN, C.AS, C.AP})
                allgather_level(mg.prm, mg.split, a, cs);
    }
}

double mg_remove_mean(MGWork& mg, const MGView& L, double* x) {
    const double m =
        det_sum(L, mg, [x](int q) { return x[q]; }) / (static_cast<double>(L.nx) * L.ny);
    const bool pr = par(L, mg);
    (void)pr;
    const int s = L.stride;
#pragma omp parallel for schedule(static) if (pr)
    for (int k_ = 0; k_ < L.nx; ++k_) { // rows descending (T38)
        const int I = L.nx - 1 - k_;
        for (int J = 0; J < L.ny; ++J)
            x[I * s + J] -= m;
    }
    return m;
}

// Refresh level l's halo of x (decomposed levels on more than one rank only).
static Exchange level_exchange(PressureWork& w, MGWork& mg, int l) {
    if (mg.prm.nranks == 1 || l >= mg.split)
        return {};
    if (l == 0)
        return [&w, &mg] { halo_exchange(mg.prm, w.pc); };
    const MGView L = mg_view(w, mg, l);
    return [L, &mg] { halo_exchange_raw(mg.prm, L.x, L.nx, L.ny, L.stride, 1, 1); };
}

void mg_vcycle(const Params& p, PressureWork& w, MGWork& mg) {
    const int nl = static_cast<int>(mg.levels.size());
    for (int l = 0; l + 1 < nl; ++l) {
        const MGView F = mg_view(w, mg, l);
        const bool pr = par(F, mg);
        smooth(F, p.mg_pre, pr, level_exchange(w, mg, l));
        residual(F, pr);
        const MGView C = mg_view(w, mg, l + 1);
        if (mg.prm.nranks > 1 && l + 1 <= mg.split) // x = 0 on the halo and other ranks' cells
            first_touch_fill(C.x - C.base, C.nx + 2, C.ny + 2, 0.0);
        restrict_residual(F, C, par(C, mg));
        if (l + 1 == mg.split)
            allgather_level(mg.prm, mg.split, C.b, C.stride);
    }
    coarse_solve(mg_view(w, mg, nl - 1), mg);
    for (int l = nl - 2; l >= 0; --l) {
        const MGView F = mg_view(w, mg, l);
        const bool pr = par(F, mg);
        prolong(mg_view(w, mg, l + 1), F, p.mg_omega, pr);
        const Exchange ex = level_exchange(w, mg, l);
        if (ex)
            ex(); // the first post-smoothing colour reads the other colour across the boundary
        smooth(F, p.mg_post, pr, ex);
    }
}

// Interior mean of a level-0 field over the whole grid (CellSums: the order of det_sum, so the
// same bits as mg_remove_mean on one rank), subtracted from this rank's cells.
static void remove_mean0(const Params& p, Field2D& f) {
    const int H = p.H;
    CellSums cs(p, H, p.NX + H - 1, H + 1, p.NY + H);
#pragma omp parallel for schedule(static)
    for (int i = cs.ib; i >= cs.ia; --i) {
        double s = 0.0, *c = cs.cells(i);
        for (int j = cs.ja; j <= cs.jb; ++j) {
            s += f(i, j);
            if (c)
                c[j - cs.ja] = f(i, j);
        }
        cs.row(i, s);
    }
    const double m = cs.sum(p) / (static_cast<double>(p.NX) * p.NY);
#pragma omp parallel for schedule(static)
    for (int i = cs.ib; i >= cs.ia; --i)
        for (int j = cs.ja; j <= cs.jb; ++j)
            f(i, j) -= m;
}

void solve_pc_mg(const Params& p, PressureWork& w, MGWork& mg) {
    mg_build(w, mg);
    w.pc.fill(0.0);
    // SPEC 4.3: project b onto the range of A (mean zero), in place. In SIMPLE the mean of b is
    // round-off; without this a non-zero mean makes the Gauss-Seidel limit non-constant-shifted.
    remove_mean0(p, w.b);
    for (int c = 0; c < p.mg_cycles; ++c)
        mg_vcycle(p, w, mg);
    remove_mean0(p, w.pc);
    halo_exchange(p, w.pc); // correct reads p' in the neighbours' halos
}
