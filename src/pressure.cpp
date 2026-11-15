#include "simple/pressure.hpp"

#include "simple/decomp.hpp"

#include <algorithm>
#include <cmath>

PressureWork::PressureWork(int rows, int cols, int row0, int col0)
    : AEP(rows, cols, 0.0, row0, col0), AWP(rows, cols, 0.0, row0, col0),
      ANP(rows, cols, 0.0, row0, col0), ASP(rows, cols, 0.0, row0, col0),
      APP(rows, cols, 0.0, row0, col0), b(rows, cols, 0.0, row0, col0),
      pc(rows, cols, 0.0, row0, col0) {}

double assemble_p(const Params& p, const Field2D& APU, const Field2D& APV, const Field2D& u,
                  const Field2D& v, PressureWork& w) {
    const int H = p.H, NX = p.NX, NY = p.NY;
    const double dx = p.dx, dy = p.dy;
#pragma omp parallel for schedule(static)
    for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
        for (int j = col_from(p, H + 1); j <= col_to(p, NY + H); ++j) {
            w.AEP(i, j) = dy * dy / APU(i, j);
            w.AWP(i, j) = dy * dy / APU(i, j - 1);
            w.ANP(i, j) = dx * dx / APV(i, j);
            w.ASP(i, j) = dx * dx / APV(i + 1, j);
        }
    const int r0 = row_from(p, 0), r1 = row_to(p, p.nrows - 1);
    const int c0 = col_from(p, 0), c1 = col_to(p, p.ncols - 1);
    for (int i = r0; i <= r1; ++i) {
        if (owns_col(p, H + 1))
            w.AWP(i, H + 1) = 0.0;
        if (owns_col(p, NY + H))
            w.AEP(i, NY + H) = 0.0;
    }
    if (owns_row(p, NX + H - 1))
        for (int j = c0; j <= c1; ++j)
            w.ASP(NX + H - 1, j) = 0.0;
    if (owns_row(p, H))
        for (int j = c0; j <= c1; ++j)
            w.ANP(H, j) = 0.0;
    CellSums cs(p, H, NX + H - 1, H + 1, NY + H); // added in serial order by sum()
#pragma omp parallel for schedule(static)
    for (int i = cs.ib; i >= cs.ia; --i) {
        double res = 0.0, *c = cs.cells(i);
        for (int j = cs.ja; j <= cs.jb; ++j) {
            w.APP(i, j) = w.AEP(i, j) + w.AWP(i, j) + w.ANP(i, j) + w.ASP(i, j);
            w.b(i, j) = -dx * (-v(i + 1, j) + v(i, j)) - dy * (-u(i, j - 1) + u(i, j));
            res += std::fabs(w.b(i, j));
            if (c)
                c[j - cs.ja] = std::fabs(w.b(i, j));
        }
        cs.row(i, res);
    }
    return cs.sum(p);
}

void solve_pc_gs_lex(const Params& p, PressureWork& w) {
    const int H = p.H, NX = p.NX, NY = p.NY;
    w.pc.fill(0.0);
    for (int k = 0; k < p.maxit_inner_p; ++k)
        for (int i = row_to(p, NX + H - 1); i >= row_from(p, H); --i)
            for (int j = H + 1; j <= NY + H; ++j) {
                w.pc(i, j) = w.b(i, j) / w.APP(i, j);
                w.pc(i, j) += (w.AEP(i, j) * w.pc(i, j + 1) + w.AWP(i, j) * w.pc(i, j - 1) +
                               w.ASP(i, j) * w.pc(i + 1, j) + w.ANP(i, j) * w.pc(i - 1, j)) /
                              w.APP(i, j);
            }
}

// Two-statement update of SPEC 4.3 step 4 at (i, j), neighbours read from `src`.
static inline void pc_update(PressureWork& w, const Field2D& src, int i, int j) {
    w.pc(i, j) = w.b(i, j) / w.APP(i, j);
    w.pc(i, j) += (w.AEP(i, j) * src(i, j + 1) + w.AWP(i, j) * src(i, j - 1) +
                   w.ASP(i, j) * src(i + 1, j) + w.ANP(i, j) * src(i - 1, j)) /
                  w.APP(i, j);
}

void solve_pc_jacobi(const Params& p, PressureWork& w, Field2D& old) {
    const int H = p.H, NX = p.NX, r0 = w.pc.row0(), r1 = w.pc.row0() + w.pc.rows(),
              c0 = w.pc.col0(), c1 = c0 + w.pc.cols(), jmin = col_from(p, H + 1),
              jmax = col_to(p, p.NY + H);
    const int n = p.maxit_inner_p;
    w.pc.fill(0.0);
    // T32: one copy, then the two buffers alternate. Physical halo cells are never written, so
    // both stay 0 there. The first destination is chosen so that the last pass writes w.pc.
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int i = r0; i < r1; ++i)
            for (int j = c0; j < c1; ++j)
                old(i, j) = w.pc(i, j);
        for (int k = 0; k < n; ++k) {
            const bool to_pc = (n - k) % 2 == 1; // pass k writes w.pc iff an odd count remains
            const Field2D& src = to_pc ? old : w.pc;
            Field2D& dst = to_pc ? w.pc : old;
#pragma omp for schedule(static)
            for (int i = row_to(p, NX + H - 1); i >= row_from(p, H); --i)
                for (int j = jmin; j <= jmax; ++j) {
                    dst(i, j) = w.b(i, j) / w.APP(i, j);
                    dst(i, j) += (w.AEP(i, j) * src(i, j + 1) + w.AWP(i, j) * src(i, j - 1) +
                                  w.ASP(i, j) * src(i + 1, j) + w.ANP(i, j) * src(i - 1, j)) /
                                 w.APP(i, j);
                    dst(i, j) = kPcJacobiWeight * dst(i, j) + (1 - kPcJacobiWeight) * src(i, j);
                }
            // MPI: the next pass (and correct, after the last) reads dst in the neighbours' halos.
            if (p.nranks > 1) {
#pragma omp master
                halo_exchange(p, dst);
#pragma omp barrier
            }
        }
    }
}

void solve_pc_redblack(const Params& p, PressureWork& w) {
    const int H = p.H, NX = p.NX, jmin = col_from(p, H + 1), jmax = col_to(p, p.NY + H);
    w.pc.fill(0.0);
#pragma omp parallel
    for (int k = 0; k < p.maxit_inner_p; ++k)
        for (int c = 0; c < 2; ++c) {
#pragma omp for schedule(static)
            for (int i = row_to(p, NX + H - 1); i >= row_from(p, H); --i) {
                int j = H + 1 + ((i + H + 1 + c) & 1);    // first cell of colour c in the row
                j += (std::max(0, jmin - j) + 1) / 2 * 2; // first one in this rank's columns
                for (; j <= jmax; j += 2)
                    pc_update(w, w.pc, i, j);
            }
            // MPI: the other colour (and correct) reads these cells in the neighbours' halos.
            // The omp for ended with a barrier; the master exchanges, then everyone waits for it.
            if (p.nranks > 1) {
#pragma omp master
                halo_exchange(p, w.pc);
#pragma omp barrier
            }
        }
}

void correct(const Params& p, const PressureWork& w, const Field2D& APU, const Field2D& APV,
             Field2D& u, Field2D& v, Field2D& p_field) {
    const int H = p.H, NX = p.NX, NY = p.NY;
    const double dy = p.dy, ap = p.relaxation_factor_p;
    // Velocity corrections use the face length (dy for u, dx for v) and no relaxation factor.
    // One team for the three loops (T33); they write different arrays and only read w.pc.
#pragma omp parallel
    {
#pragma omp for schedule(static) nowait
        for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
            for (int j = col_from(p, H + 1); j <= col_to(p, NY + H - 1); ++j)
                u(i, j) += dy * (w.pc(i, j) - w.pc(i, j + 1)) / APU(i, j);
#pragma omp for schedule(static) nowait
        for (int i = row_from(p, H + 1); i <= row_to(p, NX + H - 1); ++i)
            for (int j = col_from(p, H + 1); j <= col_to(p, NY + H); ++j)
                v(i, j) += p.dx * (w.pc(i, j) - w.pc(i - 1, j)) / APV(i, j);
#pragma omp for schedule(static)
        for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
            for (int j = col_from(p, H + 1); j <= col_to(p, NY + H); ++j)
                p_field(i, j) += ap * w.pc(i, j);
    }
}

// ---- Conjugate gradient (pressure_solver = cg) ----

CGWork::CGWork(int rows, int cols, int row0, int col0)
    : r(rows, cols, 0.0, row0, col0), z(rows, cols, 0.0, row0, col0),
      d(rows, cols, 0.0, row0, col0), q(rows, cols, 0.0, row0, col0) {}

void apply_A_p(const Params& p, const PressureWork& w, const Field2D& x, Field2D& out) {
    const int H = p.H, NX = p.NX, jmin = col_from(p, H + 1), jmax = col_to(p, p.NY + H);
#pragma omp parallel for schedule(static)
    for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
        for (int j = jmin; j <= jmax; ++j)
            out(i, j) =
                w.APP(i, j) * x(i, j) - (w.AEP(i, j) * x(i, j + 1) + w.AWP(i, j) * x(i, j - 1) +
                                         w.ASP(i, j) * x(i + 1, j) + w.ANP(i, j) * x(i - 1, j));
}

// Sums of K per-cell values over the interior in the fixed order of CellSums (rows j ascending,
// then rows ascending): the same bits for any thread count or process grid. f(i, j, v) writes the
// K values of cell (i, j) into v.
template <int K, class F> static void cg_sums(const Params& p, F f, double* out) {
    const int H = p.H;
    CellSums cs(p, H, p.NX + H - 1, H + 1, p.NY + H, K);
    const int ia = cs.ia, ib = cs.ib, ja = cs.ja, jb = cs.jb, nc = cs.nc;
    // Rows ascending like the rest of CG (not T38's descending order) and bounds in locals: rows
    // descending, or bounds read through `cs`, made the solve ~50 % slower under Apple clang and
    // ~10 % under GCC 15 (256^2, one thread).
#pragma omp parallel for schedule(static)
    for (int i = ia; i <= ib; ++i) {
        double s[K] = {}, v[K];
        double* c = cs.cells(i);
        if (c) // more than one grid column: keep the cells for the west-to-east chain
            for (int j = ja; j <= jb; ++j) {
                f(i, j, v);
                for (int m = 0; m < K; ++m) {
                    s[m] += v[m];
                    c[m * nc + j - ja] = v[m];
                }
            }
        else
            for (int j = ja; j <= jb; ++j) {
                f(i, j, v);
                for (int m = 0; m < K; ++m)
                    s[m] += v[m];
            }
        cs.row(i, s);
    }
    cs.sum(p, out);
}

int solve_pc_cg(const Params& p, PressureWork& w, CGWork& cg) {
    const int H = p.H, NX = p.NX, NY = p.NY, jmin = col_from(p, H + 1), jmax = col_to(p, NY + H);
    const double n = static_cast<double>(NX) * NY;
    Field2D &r = cg.r, &z = cg.z, &d = cg.d, &q = cg.q, &x = w.pc;
    if (cg.history)
        cg.history->clear();
    cg.last_iterations = 0;
    cg.last_rel_residual = 0.0;
    cg.last_mean_removed = 0.0;
    x.fill(0.0);

    // Project b onto the range of A (mean zero); r0 = b - A*0 = b.
    double sb = 0.0;
    cg_sums<1>(p, [&](int i, int j, double* v) { v[0] = w.b(i, j); }, &sb);
    const double mb = sb / n;
    // z = r / APP (Jacobi); sums for ||r||^2, sum(z), r.z and sum(r) in the same pass.
    double s4[4];
    cg_sums<4>(
        p,
        [&](int i, int j, double* v) {
            const double ri = w.b(i, j) - mb;
            r(i, j) = ri;
            const double zi = ri / w.APP(i, j);
            z(i, j) = zi;
            v[0] = ri * ri;
            v[1] = zi;
            v[2] = ri * zi;
            v[3] = ri;
        },
        s4);
    double rr = s4[0], sz = s4[1], rz = s4[2], sr = s4[3];
    if (cg.history)
        cg.history->push_back(rr > 0.0 ? 1.0 : 0.0);
    if (rr == 0.0)
        return 0; // ||b|| == 0: p' = 0
    const double nb = std::sqrt(rr);
    // Preconditioned residual projected onto mean zero: z <- z - mean(z), so r.z -= mean(z)*sum(r).
    double mz = sz / n;
    rz -= mz * sr;
#pragma omp parallel for schedule(static)
    for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
        for (int j = jmin; j <= jmax; ++j)
            d(i, j) = z(i, j) - mz;

    int k = 0;
    double rel = 1.0;
    while (k < p.pressure_maxit) {
        halo_exchange(p, d); // the matvec reads d in the neighbours' halos
        double dq = 0.0;
        cg_sums<1>(
            p,
            [&](int i, int j, double* v) {
                const double qi =
                    w.APP(i, j) * d(i, j) - (w.AEP(i, j) * d(i, j + 1) + w.AWP(i, j) * d(i, j - 1) +
                                             w.ASP(i, j) * d(i + 1, j) + w.ANP(i, j) * d(i - 1, j));
                q(i, j) = qi;
                v[0] = d(i, j) * qi;
            },
            &dq);
        if (!(dq > 0.0))
            break; // breakdown (d in the null space or round-off): stop with the current x
        const double alpha = rz / dq;
        cg_sums<4>(
            p,
            [&](int i, int j, double* v) {
                x(i, j) += alpha * d(i, j);
                const double ri = r(i, j) - alpha * q(i, j);
                r(i, j) = ri;
                const double zi = ri / w.APP(i, j);
                z(i, j) = zi;
                v[0] = ri * ri;
                v[1] = zi;
                v[2] = ri * zi;
                v[3] = ri;
            },
            s4);
        rr = s4[0];
        sz = s4[1];
        double rzn = s4[2];
        sr = s4[3];
        ++k;
        rel = std::sqrt(rr) / nb;
        if (cg.history)
            cg.history->push_back(rel);
        if (rel <= p.pressure_tolerance)
            break;
        mz = sz / n;
        rzn -= mz * sr;
        const double beta = rzn / rz;
        rz = rzn;
#pragma omp parallel for schedule(static)
        for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
            for (int j = jmin; j <= jmax; ++j)
                d(i, j) = (z(i, j) - mz) + beta * d(i, j);
    }

    // Remove the mean of the solution (it is zero up to round-off).
    double sx = 0.0;
    cg_sums<1>(p, [&](int i, int j, double* v) { v[0] = x(i, j); }, &sx);
    const double mx = sx / n;
#pragma omp parallel for schedule(static)
    for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
        for (int j = jmin; j <= jmax; ++j)
            x(i, j) -= mx;
    halo_exchange(p, x); // correct reads p' in the neighbours' halos
    cg.last_iterations = k;
    cg.last_rel_residual = rel;
    cg.last_mean_removed = mx;
    return k;
}
