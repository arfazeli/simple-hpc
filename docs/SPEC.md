# simple-hpc numerical specification

Numerics of the solver in `src/`: SIMPLE on a staggered grid for the 2D lid-driven cavity (unit square, unit lid speed).

## 1. Parameters

Config files are `key = value` lines. Unknown keys are an error.

| Key | Default | Notes |
|---|---|---|
| `N_cells_x`, `N_cells_y` | 80, 80 | `N_cells_x` (`NX`) counts **rows** (first index `i`), `N_cells_y` (`NY`) counts **columns** (second index `j`) |
| `N_halo_x`, `N_halo_y` (`H`) | 2, 2 | Must be equal and >= 2; needed by QUICK and 4th-order diffusion |
| `Re` | 1000 | `nu = 1/Re` |
| `maxit_inner_u`, `maxit_inner_v`, `maxit_inner_p` | 4, 4, 8 | Fixed inner sweep counts |
| `error_tolerance` | 1e-4 | Compared with the (possibly normalised) residuals, see 5 |
| `maxit` | 15000 | Maximum outer iterations |
| `relaxation_factor_u`, `_v`, `_p` | 0.3, 0.3, 0.2 | Under-relaxation of the u sweep, v sweep and pressure update |
| `convection_scheme` | `QUICK` | `UDS` or `QUICK` |
| `diffusion_order` | 4 | 2 or 4 |
| `Hybrid_switch` | `False` | `True` is rejected |
| `residual_normalisation` | `absolute` | `absolute` or `running_max` |
| `backend` | `serial` | `serial`, `openmp`, `gpu` |
| `momentum_solver` | `gs_lex` | `gs_lex`, `jacobi`, `gs_multicolour` |
| `pressure_solver` | `gs_lex` | `gs_lex`, `jacobi`, `gs_redblack`, `cg`, `mg` |
| `output_dir` | `results/run` | |

Spacing: `dx = 1/NY` (column spacing), `dy = 1/NX` (row spacing), so non-square grids are consistent.

## 2. Arrays and indexing

- Every field (`u`, `v`, `p`, coefficient arrays, `p'`) has shape `(NX + 2H + 1, NY + 2H + 1)`, stored as one contiguous row-major array; `(i, j)` maps to `i * ncols + j`.
- `i` is the row index, increasing **downward** (north to south). `j` is the column index, increasing west to east.
- Interior pressure cells: `i ∈ [H, NX+H-1]`, `j ∈ [H+1, NY+H]`. Row `i = H` is the top (lid) row.
- Staggering:
  - `u[i, j]` sits on the **east face** of pressure cell `(i, j)`, between `p[i, j]` and `p[i, j+1]`.
  - `v[i, j]` sits on the **north face** of pressure cell `(i, j)`, between `p[i, j]` (below) and `p[i-1, j]` (above). Positive `v` points north.
- Walls: west wall u-face is `u[i, H]`; east wall u-face is `u[i, NY+H]`. Neither is updated, so both stay 0. Top wall v-face is `v[H, j]`; bottom wall v-face is `v[NX+H, j]`. Neither is updated.
- Initial condition: `u = v = p = 0` everywhere.

## 3. Coefficients

`convective + diffusive` coefficients are nine values `[AE, AW, AN, AS, AEE, AWW, ANN, ASS, AP]` for a control volume, given face velocities `ue, uw` (east and west) and `vn, vs` (north and south) and the cell indices `(i, j)`. The two nine-vectors are computed separately and added element-wise, including `AP`.

**Boundary band.** A cell is in the boundary band if `j == H+1` or `j == NY+H` or `i == NX+H-1` or `i == H`. In the band, QUICK falls back to UDS and 4th-order diffusion falls back to 2nd order.

**Convection, UDS** (also used in the band):

```
AE = dy * max(0, -ue)     AW = dy * max(0,  uw)
AN = dx * max(0, -vn)     AS = dx * max(0,  vs)
AEE = AWW = ANN = ASS = 0
AP = AE + AW + AN + AS
```

**Convection, QUICK** (outside the band):

```
AE  = -dy * (-0.75*max(0,-ue) + 0.375*max(0,ue) - 0.125*max(0,-uw))
AW  = -dy * (-0.125*max(0,ue) + 0.375*max(0,-uw) - 0.75*max(0,uw))
AN  = -dx * (-0.75*max(0,-vn) + 0.375*max(0,vn) - 0.125*max(0,-vs))
AS  = -dx * (-0.125*max(0,vn) + 0.375*max(0,-vn) - 0.75*max(0,vs))
AEE = -0.125 * dy * max(0, -ue)     AWW = -0.125 * dy * max(0, uw)
ANN = -0.125 * dx * max(0, -vn)     ASS = -0.125 * dx * max(0, vs)
AP  = AE + AW + AN + AS + AEE + AWW + ANN + ASS
```

**Diffusion, order 2** (also used in the band):

```
AE = AW = dy*nu/dx      AN = AS = dx*nu/dy      AEE = AWW = ANN = ASS = 0
AP = nu * (2*dx/dy + 2*dy/dx)
```

**Diffusion, order 4** (outside the band):

```
AE = AW = (4/3)*dy*nu/dx        AN = AS = (4/3)*dx*nu/dy
AEE = AWW = -(1/12)*dy*nu/dx    ANN = ASS = -(1/12)*dx*nu/dy
AP = nu * (2.5*dx/dy + 2.5*dy/dx)
```

## 4. One outer iteration

1. `calculate_u` (4.1).
2. `calculate_v` (4.2), using the `u` just updated.
3. Pressure-correction system assembled from the two `AP` arrays and the current `u`, `v` (4.3); `p'` solved; `u`, `v`, `p` corrected.
4. Append `[iter, res_u, res_v, res_p]` to the residual log.
5. Stopping test (5).

### 4.1 u-momentum

1. **Assemble** for `i` from `NX+H-1` down to `H` and `j` from `H` to `NY+H` (wall faces included):
   - `ue = (u[i,j] + u[i,j+1]) / 2`, `uw = (u[i,j] + u[i,j-1]) / 2`
   - `vn = (v[i,j] + v[i,j+1]) / 2`, `vs = (v[i+1,j] + v[i+1,j+1]) / 2`
   - coefficients = convection + diffusion. `APU` is the `AP` array and is used by the pressure equation.
2. **Ghost rows** (after assembly):
   - North lid: `u[H-1, j] = 2 - u[H, j]` for `j ∈ [H+1, NY+H]`.
   - South wall: `u[NX+H, j] = -u[NX+H-1, j]` for `j ∈ [H+1, NY+H]`.
3. **Residual**, before any sweep: `res_u = Σ |R(i,j)|` over `i ∈ [H, NX+H-1]`, `j ∈ [H+1, NY+H-1]`, with
   `R = (AE*u[i,j+1] + AW*u[i,j-1] + AN*u[i-1,j] + AS*u[i+1,j]) + (AEE*u[i,j+2] + AWW*u[i,j-2] + ANN*u[i-2,j] + ASS*u[i+2,j]) + (p[i,j] - p[i,j+1])*dy - AP*u[i,j]`.
4. **Sweeps** (`maxit_inner_u` times), in place, same ranges, `i` descending, `j` ascending (`gs_lex`):
   `u[i,j] = alpha_u * (near4 + far4 + (p[i,j] - p[i,j+1])*dy) / AP + (1 - alpha_u) * u[i,j]`, where `near4` and `far4` are the two bracketed neighbour sums of the residual.

### 4.2 v-momentum

1. **Assemble** for `i` from `NX+H+1` down to `H`, `j` from `H` to `NY+H+1`:
   - `ue = (u[i-1,j] + u[i,j]) / 2`, `uw = (u[i,j-1] + u[i-1,j-1]) / 2`
   - `vn = (v[i,j] + v[i-1,j]) / 2`, `vs = (v[i,j] + v[i+1,j]) / 2`
2. **Ghost columns**, for `i ∈ [H, NX+H-1]`:
   - West: `v[i, H] = -v[i, H+1]`.
   - East: `v[i, NY+H+1] = -v[i, NY+H]`.
3. **Residual** over `i ∈ [H+1, NX+H-1]`, `j ∈ [H+1, NY+H]`, same form as 4.1 on `v` with pressure term `(p[i,j] - p[i-1,j]) * dx`.
4. **Sweeps** (`maxit_inner_v` times), same ranges, `i` descending, `j` ascending, with `alpha_v` and pressure term `(p[i,j] - p[i-1,j]) * dx`.

### 4.3 Pressure correction

1. For `i ∈ [H, NX+H-1]`, `j ∈ [H+1, NY+H]`:
   - `AEP = dy² / APU[i,j]`, `AWP = dy² / APU[i,j-1]`
   - `ANP = dx² / APV[i,j]`, `ASP = dx² / APV[i+1,j]`
2. Zero the wall links: `AWP[:, H+1] = 0`, `AEP[:, NY+H] = 0`, `ASP[NX+H-1, :] = 0`, `ANP[H, :] = 0`.
3. `APP = AEP + AWP + ANP + ASP`. Mass source `b[i,j] = -dx*(-v[i+1,j] + v[i,j]) - dy*(-u[i,j-1] + u[i,j])`. `res_p = Σ |b|`.
4. `p'` starts at 0 every outer iteration. Run `maxit_inner_p` Gauss–Seidel sweeps in place (`i` descending, `j` ascending), as two statements:
   `p'[i,j] = b/APP;  p'[i,j] += (AEP*p'[i,j+1] + AWP*p'[i,j-1] + ASP*p'[i+1,j] + ANP*p'[i-1,j]) / APP`
5. Corrections, with **no velocity relaxation**:
   - `u[i,j] += dy * (p'[i,j] - p'[i,j+1]) / APU[i,j]` for `i ∈ [H, NX+H-1]`, `j ∈ [H+1, NY+H-1]`
   - `v[i,j] += dx * (p'[i,j] - p'[i-1,j]) / APV[i,j]` for `i ∈ [H+1, NX+H-1]`, `j ∈ [H+1, NY+H]`
   - `p[i,j] += alpha_p * p'[i,j]` for interior cells

The p' system is symmetric and has only Neumann (zero-flux) boundaries, so it is singular up to a constant; solvers that need it (`cg`, `mg`) must remove the mean of the right-hand side and of the solution.

### 4.4 Inner solvers

The sweeps in 4.1, 4.2 and step 4 of 4.3 are the default `gs_lex` (lexicographic Gauss–Seidel, serial). Alternatives, selected by `momentum_solver` and `pressure_solver`, run in parallel and give bit-identical results for any thread count:

- `jacobi`: every cell is updated from the previous sweep's values (double-buffered), with the same under-relaxation as the Gauss–Seidel sweep. For p' the update is **damped**: `p' = 0.8 * p'_jacobi + 0.2 * p'_old`. Undamped Jacobi has eigenvalue −1 on the checkerboard mode of the 5-point Neumann operator, so with an even sweep count that part of `b` never reaches `p'` and `res_p` stalls.
- `gs_multicolour` (momentum): Gauss–Seidel by colour, one parallel pass per colour. QUICK or 4th-order diffusion reach offsets (0,±1), (0,±2), (±1,0), (±2,0), so 5 colours `(i + 3j) mod 5` are used; with UDS and order 2 (5-point stencil), 2 colours `(i + j) mod 2`.
- `gs_redblack` (p'): 2 colours `(i + j) mod 2`.
- `mg` (p'): geometric multigrid V-cycles. Remove the mean of `b`, set `p' = 0`, run `mg_cycles` V-cycles (default 1), remove the mean of `p'`. Level 0 is the NX × NY interior; halve both dimensions while both are even and the result is at least 4 × 4.
  - Coarse operator: Galerkin with piecewise-constant transfer. Coarse cell `(I,J)` covers fine `(2I..2I+1) × (2J..2J+1)`; each coarse link is the sum of the 2 fine links crossing that face, and `AP = AE + AW + AN + AS`. It is rebuilt every outer iteration.
  - Transfers: restriction sums the 4 fine residuals. Prolongation adds `mg_omega` × the coarse value (default 1.8) to the 4 fine cells. These coarse links are about 2× too stiff in 2D, so the plain correction is about half the size needed; with an exact coarse solve, `x += omega P A_c^+ R r` does not increase the A-norm error for `0 < omega ≤ 2`.
  - Smoother: red–black Gauss–Seidel in local colours `(I + J) mod 2`, `mg_pre` = `mg_post` = 2 sweeps.
  - Coarsest level: remove the mean of `b`, sweep until `||r|| < 1e-10 ||b||` or 200 sweeps, remove the mean of `x`.
  - Determinism: all sums run in a fixed order (row sums, then rows serially).

## 5. Stopping rule

After each outer iteration, with residuals `(res_u, res_v, res_p)`:

- `absolute`: test the raw sums.
- `running_max`: each residual is divided by its maximum over all iterations so far (0 if that maximum is 0).

The run stops, converged, when `max` of the three tested values `<= error_tolerance` and `iter > 1`. Otherwise it stops after `maxit` iterations, not converged. Residuals are sums of absolute values, so the `absolute` tolerance depends on the grid.

## 6. Outputs

In `output_dir`:

- Full arrays with halos: `u.csv`, `v.csv`, `p.csv`.
- Cell-centred velocities of shape `(NX, NY)`:
  - `ux_centre[r, c] = (u[H+r, H+c] + u[H+r, H+c+1]) / 2`
  - `uy_centre[r, c] = (v[H+r, H+1+c] + v[H+r+1, H+1+c]) / 2`
- Residual log `alog.csv` with columns `Iteration, Residual_u, Residual_v, Residual_p`.
- `run_info.json` (config, backend, git hash, total time) and timing output.

## 7. Floating-point reproducibility

Results are bit-reproducible only if expressions keep the operation order given above: residual and sweep sums are `(near4) + (far4) + pressure term`, evaluated left to right; sums accumulate in loop order (`i` descending, `j` ascending) into a `double` starting at `0.0`; coefficient arrays are allocated once and entries outside the assembly ranges stay 0.

`simple_core` builds with `-ffp-contract=off`. Apple clang fuses `a*b + c` by default, which shifts results by about 1e-12 on tiny values and would break the snapshot tests.
