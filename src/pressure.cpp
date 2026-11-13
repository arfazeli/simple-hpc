#include "simple/pressure.hpp"

#include "simple/decomp.hpp"

#include <cmath>

PressureWork::PressureWork(int rows, int cols, int row0)
    : AEP(rows, cols, 0.0, row0), AWP(rows, cols, 0.0, row0), ANP(rows, cols, 0.0, row0),
      ASP(rows, cols, 0.0, row0), APP(rows, cols, 0.0, row0), b(rows, cols, 0.0, row0),
      pc(rows, cols, 0.0, row0) {}

double assemble_p(const Params& p, const Field2D& APU, const Field2D& APV, const Field2D& u,
                  const Field2D& v, PressureWork& w) {
    const int H = p.H, NX = p.NX, NY = p.NY;
    const double dx = p.dx, dy = p.dy;
#pragma omp parallel for schedule(static)
    for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
        for (int j = H + 1; j <= NY + H; ++j) {
            w.AEP(i, j) = dy * dy / APU(i, j);
            w.AWP(i, j) = dy * dy / APU(i, j - 1);
            w.ANP(i, j) = dx * dx / APV(i, j);
            w.ASP(i, j) = dx * dx / APV(i + 1, j);
        }
    for (int i = row_from(p, 0); i <= row_to(p, p.nrows - 1); ++i) {
        w.AWP(i, H + 1) = 0.0;
        w.AEP(i, NY + H) = 0.0;
    }
    if (owns_row(p, NX + H - 1))
        for (int j = 0; j < p.ncols; ++j)
            w.ASP(NX + H - 1, j) = 0.0;
    if (owns_row(p, H))
        for (int j = 0; j < p.ncols; ++j)
            w.ANP(H, j) = 0.0;
    std::vector<double> rows(p.nrows, 0.0); // row sums, added in row order by sum_rows
#pragma omp parallel for schedule(static)
    for (int i = row_to(p, NX + H - 1); i >= row_from(p, H); --i) {
        double res = 0.0;
        for (int j = H + 1; j <= NY + H; ++j) {
            w.APP(i, j) = w.AEP(i, j) + w.AWP(i, j) + w.ANP(i, j) + w.ASP(i, j);
            w.b(i, j) = -dx * (-v(i + 1, j) + v(i, j)) - dy * (-u(i, j - 1) + u(i, j));
            res += std::fabs(w.b(i, j));
        }
        rows[i] = res;
    }
    return sum_rows(p, rows);
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
    const int H = p.H, NX = p.NX, NY = p.NY, r0 = w.pc.row0(), r1 = w.pc.row0() + w.pc.rows(),
              cols = w.pc.cols();
    const int n = p.maxit_inner_p;
    w.pc.fill(0.0);
    // T32: one copy, then the two buffers alternate. Halo cells are never written, so both stay 0
    // there. The first destination is chosen so that the last pass writes w.pc.
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int i = r0; i < r1; ++i)
            for (int j = 0; j < cols; ++j)
                old(i, j) = w.pc(i, j);
        for (int k = 0; k < n; ++k) {
            const bool to_pc = (n - k) % 2 == 1; // pass k writes w.pc iff an odd count remains
            const Field2D& src = to_pc ? old : w.pc;
            Field2D& dst = to_pc ? w.pc : old;
#pragma omp for schedule(static)
            for (int i = row_to(p, NX + H - 1); i >= row_from(p, H); --i)
                for (int j = H + 1; j <= NY + H; ++j) {
                    dst(i, j) = w.b(i, j) / w.APP(i, j);
                    dst(i, j) += (w.AEP(i, j) * src(i, j + 1) + w.AWP(i, j) * src(i, j - 1) +
                                  w.ASP(i, j) * src(i + 1, j) + w.ANP(i, j) * src(i - 1, j)) /
                                 w.APP(i, j);
                    dst(i, j) = kPcJacobiWeight * dst(i, j) + (1 - kPcJacobiWeight) * src(i, j);
                }
        }
    }
}

void solve_pc_redblack(const Params& p, PressureWork& w) {
    const int H = p.H, NX = p.NX, NY = p.NY;
    w.pc.fill(0.0);
#pragma omp parallel
    for (int k = 0; k < p.maxit_inner_p; ++k)
        for (int c = 0; c < 2; ++c) {
#pragma omp for schedule(static)
            for (int i = row_to(p, NX + H - 1); i >= row_from(p, H); --i)
                for (int j = H + 1 + ((i + H + 1 + c) & 1); j <= NY + H; j += 2)
                    pc_update(w, w.pc, i, j);
            // MPI: the other colour (and correct) reads these cells in the neighbours' halo rows.
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
            for (int j = H + 1; j <= NY + H - 1; ++j)
                u(i, j) += dy * (w.pc(i, j) - w.pc(i, j + 1)) / APU(i, j);
#pragma omp for schedule(static) nowait
        for (int i = row_from(p, H + 1); i <= row_to(p, NX + H - 1); ++i)
            for (int j = H + 1; j <= NY + H; ++j)
                v(i, j) += p.dx * (w.pc(i, j) - w.pc(i - 1, j)) / APV(i, j);
#pragma omp for schedule(static)
        for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
            for (int j = H + 1; j <= NY + H; ++j)
                p_field(i, j) += ap * w.pc(i, j);
    }
}

// ---- Conjugate gradient (pressure_solver = cg) ----

CGWork::CGWork(int rows, int cols, int row0)
    : r(rows, cols, 0.0, row0), z(rows, cols, 0.0, row0), d(rows, cols, 0.0, row0),
      q(rows, cols, 0.0, row0) {}

void apply_A_p(const Params& p, const PressureWork& w, const Field2D& x, Field2D& out) {
    const int H = p.H, NX = p.NX, NY = p.NY;
#pragma omp parallel for schedule(static)
    for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
        for (int j = H + 1; j <= NY + H; ++j)
            out(i, j) =
                w.APP(i, j) * x(i, j) - (w.AEP(i, j) * x(i, j + 1) + w.AWP(i, j) * x(i, j - 1) +
                                         w.ASP(i, j) * x(i + 1, j) + w.ANP(i, j) * x(i - 1, j));
}

int solve_pc_cg(const Params& p, PressureWork& w, CGWork& cg) {
    const int H = p.H, NX = p.NX, NY = p.NY;
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
#pragma omp parallel for schedule(static) reduction(+ : sb)
    for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
        for (int j = H + 1; j <= NY + H; ++j)
            sb += w.b(i, j);
    const double mb = sb / n;
    // z = r / APP (Jacobi); sums for ||r||^2, sum(z), r.z and sum(r) in the same pass.
    double rr = 0.0, sz = 0.0, rz = 0.0, sr = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : rr, sz, rz, sr)
    for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
        for (int j = H + 1; j <= NY + H; ++j) {
            const double ri = w.b(i, j) - mb;
            r(i, j) = ri;
            const double zi = ri / w.APP(i, j);
            z(i, j) = zi;
            rr += ri * ri;
            sz += zi;
            rz += ri * zi;
            sr += ri;
        }
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
        for (int j = H + 1; j <= NY + H; ++j)
            d(i, j) = z(i, j) - mz;

    int k = 0;
    double rel = 1.0;
    while (k < p.pressure_maxit) {
        double dq = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : dq)
        for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
            for (int j = H + 1; j <= NY + H; ++j) {
                const double qi =
                    w.APP(i, j) * d(i, j) - (w.AEP(i, j) * d(i, j + 1) + w.AWP(i, j) * d(i, j - 1) +
                                             w.ASP(i, j) * d(i + 1, j) + w.ANP(i, j) * d(i - 1, j));
                q(i, j) = qi;
                dq += d(i, j) * qi;
            }
        if (!(dq > 0.0))
            break; // breakdown (d in the null space or round-off): stop with the current x
        const double alpha = rz / dq;
        rr = sz = sr = 0.0;
        double rzn = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : rr, sz, rzn, sr)
        for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
            for (int j = H + 1; j <= NY + H; ++j) {
                x(i, j) += alpha * d(i, j);
                const double ri = r(i, j) - alpha * q(i, j);
                r(i, j) = ri;
                const double zi = ri / w.APP(i, j);
                z(i, j) = zi;
                rr += ri * ri;
                sz += zi;
                rzn += ri * zi;
                sr += ri;
            }
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
            for (int j = H + 1; j <= NY + H; ++j)
                d(i, j) = (z(i, j) - mz) + beta * d(i, j);
    }

    // Remove the mean of the solution (it is zero up to round-off).
    double sx = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : sx)
    for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
        for (int j = H + 1; j <= NY + H; ++j)
            sx += x(i, j);
    const double mx = sx / n;
#pragma omp parallel for schedule(static)
    for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
        for (int j = H + 1; j <= NY + H; ++j)
            x(i, j) -= mx;
    cg.last_iterations = k;
    cg.last_rel_residual = rel;
    cg.last_mean_removed = mx;
    return k;
}
