#pragma once
#include "simple/coefficients.hpp"
#include "simple/field2d.hpp"
#include "simple/params.hpp"

void assemble_v(const Params& p, const Field2D& u, const Field2D& v, Coeffs& cv);
void apply_bc_v(const Params& p, Field2D& v);
double residual_v(const Params& p, const Coeffs& cv, const Field2D& v, const Field2D& p_field);
void sweep_v_gs_lex(const Params& p, const Coeffs& cv, Field2D& v, const Field2D& p_field);
// Parallel variants (T18), same colouring as u (momentum_colour in momentum_u.hpp).
void sweep_v_jacobi(const Params& p, const Coeffs& cv, Field2D& v, const Field2D& p_field,
                    Field2D& old);
// Jacobi batch: nsweeps passes with one full copy into `old`, then alternating buffers (T32).
// Bit-identical to calling sweep_v_jacobi nsweeps times.
void sweep_v_jacobi_batch(const Params& p, const Coeffs& cv, Field2D& v, const Field2D& p_field,
                          Field2D& old, int nsweeps);
void sweep_v_multicolour(const Params& p, const Coeffs& cv, Field2D& v, const Field2D& p_field);
// nsweeps multicolour sweeps inside one OpenMP team (T33).
void sweep_v_multicolour_batch(const Params& p, const Coeffs& cv, Field2D& v,
                               const Field2D& p_field, int nsweeps);
// One sweep with the solver selected by p.momentum_solver.
void sweep_v(const Params& p, const Coeffs& cv, Field2D& v, const Field2D& p_field,
             Field2D& scratch);
double calculate_v(const Params& p, const Field2D& u, Field2D& v, const Field2D& p_field,
                   Coeffs& cv);
// nsweeps sweeps with the solver selected by p.momentum_solver (Jacobi runs as one batch).
void sweep_v_batch(const Params& p, const Coeffs& cv, Field2D& v, const Field2D& p_field,
                   Field2D& scratch, int nsweeps);
