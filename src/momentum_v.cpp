#include "simple/momentum_v.hpp"

#include "simple/decomp.hpp"
#include "simple/momentum_u.hpp"

#include <cmath>
#include <vector>

void assemble_v(const Params& p, const Field2D& u, const Field2D& v, Coeffs& cv) {
    const int H = p.H, NX = p.NX, NY = p.NY;
#pragma omp parallel for schedule(static)
    for (int i = row_to(p, NX + H + 1); i >= row_from(p, H); --i)
        for (int j = H; j <= NY + H + 1; ++j) {
            const double ue = (u(i - 1, j) + u(i, j)) / 2;
            const double uw = (u(i, j - 1) + u(i - 1, j - 1)) / 2;
            const double vn = (v(i, j) + v(i - 1, j)) / 2;
            const double vs = (v(i, j) + v(i + 1, j)) / 2;
            cv.set(i, j,
                   convective_coefficients(p, ue, uw, vn, vs, i, j) +
                       diffusive_coefficients(p, i, j));
        }
}

void apply_bc_v(const Params& p, Field2D& v) {
    const int H = p.H, NX = p.NX, NY = p.NY;
    // O(N) loops run serially: a team start costs more than the work (T33).
    for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
        v(i, H) = -v(i, H + 1);
    // East ghost column mirrors the last interior column.
    for (int i = row_from(p, H); i <= row_to(p, NX + H - 1); ++i)
        v(i, NY + H + 1) = -v(i, NY + H);
}

double residual_v(const Params& p, const Coeffs& cv, const Field2D& v, const Field2D& pf) {
    const int H = p.H, NX = p.NX, NY = p.NY;
    std::vector<double> rows(p.nrows, 0.0); // row sums, added in row order by sum_rows
#pragma omp parallel for schedule(static)
    for (int i = row_to(p, NX + H - 1); i >= row_from(p, H + 1); --i) {
        double res = 0.0;
        for (int j = H + 1; j <= NY + H; ++j) {
            const double R = (cv.AE(i, j) * v(i, j + 1) + cv.AW(i, j) * v(i, j - 1) +
                              cv.AN(i, j) * v(i - 1, j) + cv.AS(i, j) * v(i + 1, j)) +
                             (cv.AEE(i, j) * v(i, j + 2) + cv.AWW(i, j) * v(i, j - 2) +
                              cv.ANN(i, j) * v(i - 2, j) + cv.ASS(i, j) * v(i + 2, j)) +
                             (pf(i, j) - pf(i - 1, j)) * p.dx - cv.AP(i, j) * v(i, j);
            res += std::abs(R);
        }
        rows[i] = res;
    }
    return sum_rows(p, rows);
}

void sweep_v_gs_lex(const Params& p, const Coeffs& cv, Field2D& v, const Field2D& pf) {
    const int H = p.H, NX = p.NX, NY = p.NY;
    const double av = p.relaxation_factor_v;
    for (int i = row_to(p, NX + H - 1); i >= row_from(p, H + 1); --i)
        for (int j = H + 1; j <= NY + H; ++j)
            v(i, j) = av *
                          ((cv.AE(i, j) * v(i, j + 1) + cv.AW(i, j) * v(i, j - 1) +
                            cv.AN(i, j) * v(i - 1, j) + cv.AS(i, j) * v(i + 1, j)) +
                           (cv.AEE(i, j) * v(i, j + 2) + cv.AWW(i, j) * v(i, j - 2) +
                            cv.ANN(i, j) * v(i - 2, j) + cv.ASS(i, j) * v(i + 2, j)) +
                           (pf(i, j) - pf(i - 1, j)) * p.dx) /
                          cv.AP(i, j) +
                      (1 - av) * v(i, j);
}

// Same neighbour sums as sweep_v_gs_lex, evaluated on `f`.
static inline double nb_sum_v(const Coeffs& cv, const Field2D& f, int i, int j) {
    return (cv.AE(i, j) * f(i, j + 1) + cv.AW(i, j) * f(i, j - 1) + cv.AN(i, j) * f(i - 1, j) +
            cv.AS(i, j) * f(i + 1, j)) +
           (cv.AEE(i, j) * f(i, j + 2) + cv.AWW(i, j) * f(i, j - 2) + cv.ANN(i, j) * f(i - 2, j) +
            cv.ASS(i, j) * f(i + 2, j));
}

static inline double near_sum_v(const Coeffs& cv, const Field2D& f, int i, int j) {
    return cv.AE(i, j) * f(i, j + 1) + cv.AW(i, j) * f(i, j - 1) + cv.AN(i, j) * f(i - 1, j) +
           cv.AS(i, j) * f(i + 1, j);
}

void sweep_v_jacobi_batch(const Params& p, const Coeffs& cv, Field2D& v, const Field2D& pf,
                          Field2D& old, int nsweeps) {
    if (nsweeps <= 0)
        return;
    const int H = p.H, NX = p.NX, NY = p.NY, r0 = v.row0(), r1 = v.row0() + v.rows(),
              cols = v.cols();
    const double av = p.relaxation_factor_v;
    // One team for the whole batch (T33). One full copy (halo included); passes never write the
    // halo, so both buffers keep it. Buffers then alternate (T32); each thread derives src/dst
    // from k, and the first destination is chosen so that the last pass writes v.
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int i = r0; i < r1; ++i)
            for (int j = 0; j < cols; ++j)
                old(i, j) = v(i, j);
        for (int k = 0; k < nsweeps; ++k) {
            const bool to_v = (nsweeps - k) % 2 == 1;
            const Field2D& src = to_v ? old : v;
            Field2D& dst = to_v ? v : old;
#pragma omp for schedule(static)
            for (int i = row_to(p, NX + H - 1); i >= row_from(p, H + 1); --i)
                for (int j = H + 1; j <= NY + H; ++j)
                    dst(i, j) =
                        av * (nb_sum_v(cv, src, i, j) + (pf(i, j) - pf(i - 1, j)) * p.dx) /
                            cv.AP(i, j) +
                        (1 - av) * src(i, j);
        }
    }
}

void sweep_v_jacobi(const Params& p, const Coeffs& cv, Field2D& v, const Field2D& pf,
                    Field2D& old) {
    sweep_v_jacobi_batch(p, cv, v, pf, old, 1);
}

void sweep_v_multicolour_batch(const Params& p, const Coeffs& cv, Field2D& v, const Field2D& pf,
                               int nsweeps) {
    const int H = p.H, NX = p.NX, NY = p.NY, ncol = momentum_colours(p);
    const double av = p.relaxation_factor_v;
#pragma omp parallel // one team for all sweeps and colours (T33)
    for (int k = 0; k < nsweeps; ++k)
        for (int c = 0; c < ncol; ++c) {
#pragma omp for schedule(static)
            for (int i = row_to(p, NX + H - 1); i >= row_from(p, H + 1); --i) {
                int j0 = H + 1;
                while (momentum_colour(ncol, i, j0) != c)
                    ++j0;
                for (int j = j0; j <= NY + H; j += ncol) {
                    const double nb =
                        ncol == 2 ? near_sum_v(cv, v, i, j) : nb_sum_v(cv, v, i, j);
                    v(i, j) = av * (nb + (pf(i, j) - pf(i - 1, j)) * p.dx) / cv.AP(i, j) +
                              (1 - av) * v(i, j);
                }
            }
            // MPI: the next colour reads this colour's cells in the neighbours' halo rows. The
            // omp for ended with a barrier; the master exchanges, then everyone waits for it.
            if (p.nranks > 1) {
#pragma omp master
                halo_exchange(p, v);
#pragma omp barrier
            }
        }
}

void sweep_v_multicolour(const Params& p, const Coeffs& cv, Field2D& v, const Field2D& pf) {
    sweep_v_multicolour_batch(p, cv, v, pf, 1);
}

void sweep_v(const Params& p, const Coeffs& cv, Field2D& v, const Field2D& pf, Field2D& scratch) {
    switch (p.momentum_solver) {
    case MomentumSolver::Jacobi:
        sweep_v_jacobi(p, cv, v, pf, scratch);
        break;
    case MomentumSolver::GsMulticolour:
        sweep_v_multicolour(p, cv, v, pf);
        break;
    default:
        sweep_v_gs_lex(p, cv, v, pf);
    }
}

void sweep_v_batch(const Params& p, const Coeffs& cv, Field2D& v, const Field2D& pf,
                   Field2D& scratch, int nsweeps) {
    if (p.momentum_solver == MomentumSolver::Jacobi)
        sweep_v_jacobi_batch(p, cv, v, pf, scratch, nsweeps);
    else if (p.momentum_solver == MomentumSolver::GsMulticolour)
        sweep_v_multicolour_batch(p, cv, v, pf, nsweeps);
    else
        for (int k = 0; k < nsweeps; ++k)
            sweep_v_gs_lex(p, cv, v, pf);
}

double calculate_v(const Params& p, const Field2D& u, Field2D& v, const Field2D& pf, Coeffs& cv) {
    assemble_v(p, u, v, cv);
    apply_bc_v(p, v);
    const double res = residual_v(p, cv, v, pf);
    for (int k = 0; k < p.maxit_inner_v; ++k)
        sweep_v_gs_lex(p, cv, v, pf);
    return res;
}
