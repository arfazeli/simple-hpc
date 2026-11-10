#pragma once
#include "simple/coefficients.hpp"
#include "simple/field2d.hpp"
#include "simple/params.hpp"

// u-momentum (SPEC 3.5).
void assemble_u(const Params& p, const Field2D& u, const Field2D& v, Coeffs& cu);
void apply_bc_u(const Params& p, Field2D& u);
double residual_u(const Params& p, const Coeffs& cu, const Field2D& u, const Field2D& p_field);
void sweep_u_gs_lex(const Params& p, const Coeffs& cu, Field2D& u, const Field2D& p_field);
// Parallel variants (T18). Jacobi reads a copy of u taken into `old` at the start of the sweep.
// Multicolour GS updates one colour at a time in place; colour = (i + 3j) mod 5, or (i + j) mod 2
// when the stencil has 5 points (UDS and diffusion_order 2). See momentum_colour.
int momentum_colours(const Params& p);
inline int momentum_colour(int ncol, int i, int j) {
    return ncol == 2 ? (i + j) % 2 : (i + 3 * j) % 5;
}
void sweep_u_jacobi(const Params& p, const Coeffs& cu, Field2D& u, const Field2D& p_field,
                    Field2D& old);
// Jacobi batch: nsweeps passes with one full copy into `old`, then alternating buffers (T32).
// Bit-identical to calling sweep_u_jacobi nsweeps times.
void sweep_u_jacobi_batch(const Params& p, const Coeffs& cu, Field2D& u, const Field2D& p_field,
                          Field2D& old, int nsweeps);
void sweep_u_multicolour(const Params& p, const Coeffs& cu, Field2D& u, const Field2D& p_field);
// nsweeps multicolour sweeps inside one OpenMP team (T33).
void sweep_u_multicolour_batch(const Params& p, const Coeffs& cu, Field2D& u,
                               const Field2D& p_field, int nsweeps);
// One sweep with the solver selected by p.momentum_solver.
void sweep_u(const Params& p, const Coeffs& cu, Field2D& u, const Field2D& p_field,
             Field2D& scratch);
double calculate_u(const Params& p, Field2D& u, const Field2D& v, const Field2D& p_field,
                   Coeffs& cu);
// nsweeps sweeps with the solver selected by p.momentum_solver (Jacobi runs as one batch).
void sweep_u_batch(const Params& p, const Coeffs& cu, Field2D& u, const Field2D& p_field,
                   Field2D& scratch, int nsweeps);
