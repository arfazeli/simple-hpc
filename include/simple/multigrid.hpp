#pragma once
#include "simple/field2d.hpp"
#include "simple/params.hpp"
#include "simple/pressure.hpp"

#include <cstddef>
#include <utility>
#include <vector>

// Geometric multigrid V-cycle for p' (pressure_solver = mg), SPEC 4.4.
//
// Level 0 is the interior p' grid (NX x NY cells). Coarsen by 2 in each direction while both
// dimensions are even and the coarse grid is at least 4 x 4. Coarse operator: Galerkin with
// piecewise-constant transfer ("additive correction"): coarse cell (I, J) covers fine cells
// (2I, 2I+1) x (2J, 2J+1); each coarse link is the sum of the two fine links crossing that coarse
// face, links inside the 2x2 block are dropped, AP = AE + AW + AN + AS. Restriction sums the 4
// fine residuals, prolongation adds mg_omega times the coarse correction to the 4 fine cells
// (R = P^T, and A_c = R A_f P exactly because fine rows sum to zero). Over-correction: in 2D these
// coarse links are ~2x a rediscretised operator, so the plain correction is about half the size
// needed and the V-cycle factor degrades with depth (0.80 at 64^2 with omega = 1, 0.26 with 1.8).
// Parallel loops split rows I in descending order, like the other kernels (T38), so each thread
// works on the rows whose pages it first-touched. Per-cell results and fixed-order sums are
// unchanged.
// Smoother: red-black Gauss-Seidel with the two-statement update of SPEC 4.3 step 4, colour (I + J)
// mod 2 in global level indices, colour 0 first. Every kernel is per-cell independent or a
// fixed-order sum, so results do not depend on the thread count.

// MPI (T44): levels 0..D-1 (D = Params::mg_split) hold this rank's block only; the split unit
// 2^D keeps every rank boundary between 2x2 blocks, so restriction, prolongation and the coarse
// build stay local, and the smoother exchanges halos after each colour and after prolongation.
// Level D's right-hand side (and operator, in mg_build) is gathered onto every rank, which then
// runs levels D.. and the coarse solve redundantly with the serial code: the same bits on any grid.
// With one rank D = number of levels (nothing gathered).

// Level dimensions (rows, cols), finest first.
std::vector<std::pair<int, int>> mg_level_dims(int nx, int ny);

// Storage of one level. Coarse levels (l > 0) own padded (nx+2) x (ny+2) arrays with a zero halo
// (interior (1..nx, 1..ny)). Level 0 owns only the residual r, laid out like the Field2D arrays
// (nrows x ncols); its coefficients, b and x are PressureWork's AEP.., b and pc.
struct MGLevel {
    int nx = 0, ny = 0; // cells stored as interior: this rank's block on decomposed levels
    int I0 = 0, J0 = 0; // global level index of interior cell (0, 0)
    // No-init storage, first-touched in parallel by MGWork's constructor (T38).
    std::vector<double, NoInitAlloc<double>> AE, AW, AN, AS, AP, b, x, r;
};

// Raw view of a level: element (I, J), 0 <= I < nx, 0 <= J < ny, is ptr[I * stride + J], where
// each ptr points at interior cell (0, 0) of an allocation of `alloc` doubles starting `base`
// doubles earlier. Neighbours (I+-1, J) and (I, J+-1) of any interior cell are inside it.
struct MGView {
    int nx = 0, ny = 0, stride = 0;
    int I0 = 0, J0 = 0; // global level index of element (0, 0)
    std::size_t alloc = 0;
    std::ptrdiff_t base = 0;
    double *AE = nullptr, *AW = nullptr, *AN = nullptr, *AS = nullptr, *AP = nullptr;
    double *b = nullptr, *x = nullptr, *r = nullptr;
};

struct MGWork {
    std::vector<MGLevel> levels;
    int H = 0, nrows = 0, ncols = 0; // level-0 storage: PressureWork's arrays (rank's block)
    int row0 = 0, col0 = 0;          // their first stored global row / column
    int split = 0;                   // levels 0..split-1 are decomposed (all of them on one rank)
    Params prm;                      // decomposition, for the halo exchanges and the gather
    // Levels with fewer cells run their kernels on one thread (same results, less overhead).
    int parallel_min_cells = 4096;
    int last_coarse_sweeps = 0; // red-black sweeps of the last coarsest-level solve
    std::vector<double> rowsum; // per-row partial sums (fixed-order reductions)
    MGWork() = default;
    explicit MGWork(const Params& p); // allocates every level once
};

MGView mg_view(PressureWork& w, MGWork& mg, int level);
// out = A x over the interior of level L (x and out laid out like L's arrays).
void mg_apply(const MGView& L, const double* x, double* out);
// Rebuild the coarse operators from the fine p' coefficients in w (after assemble_p).
void mg_build(PressureWork& w, MGWork& mg);
// One V-cycle on w.pc (the current iterate) for the right-hand side w.b. mg_build must have run.
// If level 0 is also the coarsest level, the coarse solve removes the mean of w.b in place.
void mg_vcycle(const Params& p, PressureWork& w, MGWork& mg);
// Subtract the interior mean of x on level L (fixed-order sum: per-row partial sums, then rows
// in order). Returns the mean removed. One rank, or a level held whole on every rank.
double mg_remove_mean(MGWork& mg, const MGView& L, double* x);
// pressure_solver = mg: mg_build; p' = 0; mg_cycles V-cycles; remove the mean of p' (global
// fixed-order sums on any process grid); refresh p' halos for correct().
void solve_pc_mg(const Params& p, PressureWork& w, MGWork& mg);
