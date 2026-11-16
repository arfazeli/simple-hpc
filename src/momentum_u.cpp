#include "simple/momentum_u.hpp"

#include "simple/decomp.hpp"

#include <cmath>
#include <vector>

void assemble_u(const Params& p, const Field2D& u, const Field2D& v, Coeffs& cu) {
    const int H = p.H;
#pragma omp parallel for schedule(static)
    for (int i = row_to(p, p.NX + H - 1); i >= row_from(p, H); --i) {
        for (int j = col_from(p, H); j <= col_to(p, p.NY + H); ++j) {
            double ue = (u(i, j) + u(i, j + 1)) / 2;
            double uw = (u(i, j) + u(i, j - 1)) / 2;
            double vn = (v(i, j) + v(i, j + 1)) / 2;
            double vs = (v(i + 1, j) + v(i + 1, j + 1)) / 2;
            cu.set(i, j,
                   convective_coefficients(p, ue, uw, vn, vs, i, j) +
                       diffusive_coefficients(p, i, j));
        }
    }
}

void apply_bc_u(const Params& p, Field2D& u) {
    const int H = p.H;
    // O(N) loops run serially: a team start costs more than the work (T33).
    if (owns_row(p, H - 1)) // the lid row and the first interior row belong to grid row 0
        for (int j = col_from(p, H + 1); j <= col_to(p, p.NY + H); ++j)
            u(H - 1, j) = 2 - u(H, j);
    // South ghost row u[NX+H, j] mirrors the first interior row.
    if (owns_row(p, p.NX + H))
        for (int j = col_from(p, H + 1); j <= col_to(p, p.NY + H); ++j)
            u(p.NX + H, j) = -u(p.NX + H - 1, j);
}

static inline double nb_sum(const Coeffs& c, const Field2D& u, int i, int j) {
    return (c.AE(i, j) * u(i, j + 1) + c.AW(i, j) * u(i, j - 1) + c.AN(i, j) * u(i - 1, j) +
            c.AS(i, j) * u(i + 1, j)) +
           (c.AEE(i, j) * u(i, j + 2) + c.AWW(i, j) * u(i, j - 2) + c.ANN(i, j) * u(i - 2, j) +
            c.ASS(i, j) * u(i + 2, j));
}

double residual_u(const Params& p, const Coeffs& cu, const Field2D& u, const Field2D& pf) {
    const int H = p.H;
    CellSums cs(p, H, p.NX + H - 1, H + 1, p.NY + H - 1); // added in serial order by sum()
#pragma omp parallel for schedule(static)
    for (int i = cs.ib; i >= cs.ia; --i) {
        double res = 0.0, *c = cs.cells(i);
        for (int j = cs.ja; j <= cs.jb; ++j) {
            double R =
                nb_sum(cu, u, i, j) + (pf(i, j) - pf(i, j + 1)) * p.dy - cu.AP(i, j) * u(i, j);
            res += std::fabs(R);
            if (c)
                c[j - cs.ja] = std::fabs(R);
        }
        cs.row(i, res);
    }
    return cs.sum(p);
}

void sweep_u_gs_lex(const Params& p, const Coeffs& cu, Field2D& u, const Field2D& pf) {
    const int H = p.H;
    const double a = p.relaxation_factor_u;
    for (int i = row_to(p, p.NX + H - 1); i >= row_from(p, H); --i)
        for (int j = H + 1; j <= p.NY + H - 1; ++j)
            u(i, j) = a * (nb_sum(cu, u, i, j) + (pf(i, j) - pf(i, j + 1)) * p.dy) / cu.AP(i, j) +
                      (1 - a) * u(i, j);
}

int momentum_colours(const Params& p) {
    return (p.convection_scheme == Convection::UDS && p.diffusion_order == 2) ? 2 : 5;
}

// First column j >= jmin of colour c in row i.
static inline int first_j(int ncol, int c, int i, int jmin) {
    int j = jmin;
    while (momentum_colour(ncol, i, j) != c)
        ++j;
    return j;
}

// 5-point neighbour sum for the 2-colour case (UDS, order 2: the far coefficients are 0 and the
// far cells may belong to the same colour, so they must not be read).
static inline double near_sum(const Coeffs& c, const Field2D& u, int i, int j) {
    return c.AE(i, j) * u(i, j + 1) + c.AW(i, j) * u(i, j - 1) + c.AN(i, j) * u(i - 1, j) +
           c.AS(i, j) * u(i + 1, j);
}

void sweep_u_jacobi_batch(const Params& p, const Coeffs& cu, Field2D& u, const Field2D& pf,
                          Field2D& old, int nsweeps) {
    if (nsweeps <= 0)
        return;
    const int H = p.H, r0 = u.row0(), r1 = u.row0() + u.rows(), c0 = u.col0(), c1 = c0 + u.cols();
    const double a = p.relaxation_factor_u;
    // One team for the whole batch (T33). One full copy (halo included); passes never write the
    // halo, so both buffers keep it. Buffers then alternate (T32); each thread derives src/dst
    // from k, and the first destination is chosen so that the last pass writes u.
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int i = r0; i < r1; ++i)
            for (int j = c0; j < c1; ++j)
                old(i, j) = u(i, j);
        for (int k = 0; k < nsweeps; ++k) {
            const bool to_u = (nsweeps - k) % 2 == 1;
            const Field2D& src = to_u ? old : u;
            Field2D& dst = to_u ? u : old;
#pragma omp for schedule(static)
            for (int i = row_to(p, p.NX + H - 1); i >= row_from(p, H); --i)
                for (int j = H + 1; j <= p.NY + H - 1; ++j)
                    dst(i, j) = a * (nb_sum(cu, src, i, j) + (pf(i, j) - pf(i, j + 1)) * p.dy) /
                                    cu.AP(i, j) +
                                (1 - a) * src(i, j);
        }
    }
}

void sweep_u_jacobi(const Params& p, const Coeffs& cu, Field2D& u, const Field2D& pf,
                    Field2D& old) {
    sweep_u_jacobi_batch(p, cu, u, pf, old, 1);
}

void sweep_u_multicolour_batch(const Params& p, const Coeffs& cu, Field2D& u, const Field2D& pf,
                               int nsweeps) {
    const int H = p.H, ncol = momentum_colours(p);
    const double a = p.relaxation_factor_u;
    const Box k{row_from(p, H), row_to(p, p.NX + H - 1), col_from(p, H + 1), col_to(p, p.NY + H - 1)};
    // MPI: the next colour reads this colour's cells in the neighbours' halos (colour_sweeps).
    const auto bx = overlap_boxes(p, k, {p.row_lo, p.row_hi, p.col_lo, p.col_hi}, H);
    auto pass = [&](int c, const Box& b) {
#pragma omp for schedule(static) nowait
        for (int i = b.ib; i >= b.ia; --i)
            for (int j = first_j(ncol, c, i, b.ja); j <= b.jb; j += ncol) {
                const double nb = ncol == 2 ? near_sum(cu, u, i, j) : nb_sum(cu, u, i, j);
                u(i, j) =
                    a * (nb + (pf(i, j) - pf(i, j + 1)) * p.dy) / cu.AP(i, j) + (1 - a) * u(i, j);
            }
    };
#pragma omp parallel // one team for all sweeps and colours (T33)
    colour_sweeps(
        halo_mode(p), nsweeps, ncol, k, bx, pass, [&] { halo_exchange(p, u); },
        [&] { halo_post(p, u); });
}

void sweep_u_multicolour(const Params& p, const Coeffs& cu, Field2D& u, const Field2D& pf) {
    sweep_u_multicolour_batch(p, cu, u, pf, 1);
}

void sweep_u(const Params& p, const Coeffs& cu, Field2D& u, const Field2D& pf, Field2D& scratch) {
    switch (p.momentum_solver) {
    case MomentumSolver::Jacobi:
        sweep_u_jacobi(p, cu, u, pf, scratch);
        break;
    case MomentumSolver::GsMulticolour:
        sweep_u_multicolour(p, cu, u, pf);
        break;
    default:
        sweep_u_gs_lex(p, cu, u, pf);
    }
}

void sweep_u_batch(const Params& p, const Coeffs& cu, Field2D& u, const Field2D& pf,
                   Field2D& scratch, int nsweeps) {
    if (p.momentum_solver == MomentumSolver::Jacobi)
        sweep_u_jacobi_batch(p, cu, u, pf, scratch, nsweeps);
    else if (p.momentum_solver == MomentumSolver::GsMulticolour)
        sweep_u_multicolour_batch(p, cu, u, pf, nsweeps);
    else
        for (int k = 0; k < nsweeps; ++k)
            sweep_u_gs_lex(p, cu, u, pf);
}

double calculate_u(const Params& p, Field2D& u, const Field2D& v, const Field2D& pf, Coeffs& cu) {
    assemble_u(p, u, v, cu);
    apply_bc_u(p, u);
    double res = residual_u(p, cu, u, pf);
    for (int k = 0; k < p.maxit_inner_u; ++k)
        sweep_u_gs_lex(p, cu, u, pf);
    return res;
}
