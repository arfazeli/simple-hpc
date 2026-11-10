#include "simple/multigrid.hpp"

#include <cmath>
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

// `sweeps` red-black Gauss-Seidel sweeps on L.x, two-statement update of SPEC 4.3 step 4.
void smooth(const MGView& L, int sweeps, bool pr) {
    (void)pr;
    const int s = L.stride, nx = L.nx, ny = L.ny;
    double* x = L.x;
    const double *AE = L.AE, *AW = L.AW, *AN = L.AN, *AS = L.AS, *AP = L.AP, *b = L.b;
#pragma omp parallel if (pr)
    for (int k = 0; k < sweeps; ++k)
        for (int c = 0; c < 2; ++c) {
#pragma omp for schedule(static)
            for (int k_ = 0; k_ < nx; ++k_) { // rows descending (T38)
                const int I = nx - 1 - k_;
                for (int J = (I + c) & 1; J < ny; J += 2) {
                    const int q = I * s + J;
                    x[q] = b[q] / AP[q];
                    x[q] += (AE[q] * x[q + 1] + AW[q] * x[q - 1] + AS[q] * x[q + s] +
                             AN[q] * x[q - s]) /
                            AP[q];
                }
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

// C.b = R F.r (sum of the 4 fine residuals), C.x = 0.
void restrict_residual(const MGView& F, const MGView& C, bool pr) {
    (void)pr;
    const int fs = F.stride, cs = C.stride;
#pragma omp parallel for schedule(static) if (pr)
    for (int k_ = 0; k_ < C.nx; ++k_) { // rows descending (T38)
        const int I = C.nx - 1 - k_;
        for (int J = 0; J < C.ny; ++J) {
            const int f = 2 * I * fs + 2 * J;
            C.b[I * cs + J] = F.r[f] + F.r[f + 1] + F.r[f + fs] + F.r[f + fs + 1];
            C.x[I * cs + J] = 0.0;
        }
    }
}

// F.x += omega P C.x (injection to the 4 fine cells, over-corrected by omega).
void prolong(const MGView& C, const MGView& F, double omega, bool pr) {
    (void)pr;
    const int fs = F.stride, cs = C.stride;
#pragma omp parallel for schedule(static) if (pr)
    for (int k_ = 0; k_ < F.nx; ++k_) { // rows descending (T38)
        const int i = F.nx - 1 - k_;
        for (int j = 0; j < F.ny; ++j)
            F.x[i * fs + j] += omega * C.x[(i / 2) * cs + j / 2];
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

MGWork::MGWork(const Params& p) : H(p.H), nrows(p.nrows), ncols(p.ncols), rowsum(p.NX) {
    const auto dims = mg_level_dims(p.NX, p.NY);
    levels.resize(dims.size());
    for (std::size_t l = 0; l < dims.size(); ++l) {
        MGLevel& L = levels[l];
        L.nx = dims[l].first;
        L.ny = dims[l].second;
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
    if (level == 0) {
        v.stride = mg.ncols;
        v.alloc = static_cast<std::size_t>(mg.nrows) * mg.ncols;
        v.base = static_cast<std::ptrdiff_t>(mg.H) * mg.ncols + mg.H + 1;
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
        const int fs = F.stride, cs = C.stride;
        const bool pr = par(C, mg);
        (void)pr;
#pragma omp parallel for schedule(static) if (pr)
        for (int k_ = 0; k_ < C.nx; ++k_) { // rows descending (T38)
            const int I = C.nx - 1 - k_;
            for (int J = 0; J < C.ny; ++J) {
                const int f = 2 * I * fs + 2 * J, c = I * cs + J;
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

void mg_vcycle(const Params& p, PressureWork& w, MGWork& mg) {
    const int nl = static_cast<int>(mg.levels.size());
    for (int l = 0; l + 1 < nl; ++l) {
        const MGView F = mg_view(w, mg, l);
        const bool pr = par(F, mg);
        smooth(F, p.mg_pre, pr);
        residual(F, pr);
        const MGView C = mg_view(w, mg, l + 1);
        restrict_residual(F, C, par(C, mg));
    }
    coarse_solve(mg_view(w, mg, nl - 1), mg);
    for (int l = nl - 2; l >= 0; --l) {
        const MGView F = mg_view(w, mg, l);
        const bool pr = par(F, mg);
        prolong(mg_view(w, mg, l + 1), F, p.mg_omega, pr);
        smooth(F, p.mg_post, pr);
    }
}

void solve_pc_mg(const Params& p, PressureWork& w, MGWork& mg) {
    mg_build(w, mg);
    w.pc.fill(0.0);
    const MGView L0 = mg_view(w, mg, 0);
    // SPEC 4.3: project b onto the range of A (mean zero), in place. In SIMPLE the mean of b is
    // round-off; without this a non-zero mean makes the Gauss-Seidel limit non-constant-shifted.
    mg_remove_mean(mg, L0, L0.b);
    for (int c = 0; c < p.mg_cycles; ++c)
        mg_vcycle(p, w, mg);
    mg_remove_mean(mg, L0, L0.x);
}
