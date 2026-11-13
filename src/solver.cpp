#include "simple/solver.hpp"

#include "simple/decomp.hpp"
#include "simple/momentum_u.hpp"
#include "simple/momentum_v.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>

// Every array holds this rank's slab: rows [alloc_row0, alloc_row0 + R) with global indices
// (the whole grid when there is one rank).
static int slab_rows(const Params& p) { return p.alloc_rows > 0 ? p.alloc_rows : p.nrows; }

State::State(const Params& p)
    : u(slab_rows(p), p.ncols, 0.0, p.alloc_row0), v(slab_rows(p), p.ncols, 0.0, p.alloc_row0),
      p(slab_rows(p), p.ncols, 0.0, p.alloc_row0), cu(slab_rows(p), p.ncols, p.alloc_row0),
      cv(slab_rows(p), p.ncols, p.alloc_row0), pw(slab_rows(p), p.ncols, p.alloc_row0),
      scratch(slab_rows(p), p.ncols, 0.0, p.alloc_row0), cg(slab_rows(p), p.ncols, p.alloc_row0),
      mg(p.pressure_solver == PressureSolver::MG ? MGWork(p) : MGWork()) {}

Residuals outer_iteration(const Params& p, State& s, PhaseTimers& t) {
    Residuals r{};
    // Same kernels, same order as calculate_u / calculate_v / assemble_p+solve+correct.
    {
        {
            ScopedTimer st(t, Phase::AssembleU);
            assemble_u(p, s.u, s.v, s.cu);
        }
        {
            ScopedTimer st(t, Phase::BcU);
            apply_bc_u(p, s.u);
        }
        {
            ScopedTimer st(t, Phase::ResidualU);
            r.u = residual_u(p, s.cu, s.u, s.p);
        }
        ScopedTimer st(t, Phase::SweepU);
        sweep_u_batch(p, s.cu, s.u, s.p, s.scratch, p.maxit_inner_u);
    }
    {
        {
            ScopedTimer st(t, Phase::AssembleV);
            assemble_v(p, s.u, s.v, s.cv);
            halo_exchange(p, s.cv.AP); // assemble_p reads APV(i + 1, j)
        }
        {
            ScopedTimer st(t, Phase::BcV);
            apply_bc_v(p, s.v);
        }
        {
            ScopedTimer st(t, Phase::ResidualV);
            r.v = residual_v(p, s.cv, s.v, s.p);
        }
        ScopedTimer st(t, Phase::SweepV);
        sweep_v_batch(p, s.cv, s.v, s.p, s.scratch, p.maxit_inner_v);
    }
    {
        ScopedTimer st(t, Phase::AssembleP);
        r.p = assemble_p(p, s.cu.AP, s.cv.AP, s.u, s.v, s.pw);
    }
    {
        ScopedTimer st(t, Phase::SolveP);
        if (p.pressure_solver == PressureSolver::Jacobi)
            solve_pc_jacobi(p, s.pw, s.scratch);
        else if (p.pressure_solver == PressureSolver::GsRedBlack)
            solve_pc_redblack(p, s.pw);
        else if (p.pressure_solver == PressureSolver::CG)
            solve_pc_cg(p, s.pw, s.cg);
        else if (p.pressure_solver == PressureSolver::MG)
            solve_pc_mg(p, s.pw, s.mg);
        else
            solve_pc_gs_lex(p, s.pw);
    }
    {
        ScopedTimer st(t, Phase::Correct);
        correct(p, s.pw, s.cu.AP, s.cv.AP, s.u, s.v, s.p);
        // MPI: the next iteration's assembly reads u, v (i +- 1) and p (i - 1) across slabs. The
        // sweeps refresh u, v and p' halos after every colour; the boundary-condition rows are
        // never in another rank's halo (decompose keeps >= H rows per rank).
        halo_exchange(p, s.u);
        halo_exchange(p, s.v);
        halo_exchange(p, s.p);
    }
    return r;
}

RunResult run_simple(const Params& p, State& s) {
    long long cg_iterations = 0;
    RunResult out = run_outer_loop(p, [&](PhaseTimers& t) {
        Residuals r = outer_iteration(p, s, t);
        cg_iterations += s.cg.last_iterations;
        return r;
    });
    out.cg_iterations = cg_iterations;
    return out;
}

RunResult run_outer_loop(const Params& p, const std::function<Residuals(PhaseTimers&)>& step) {
    RunResult out;
    auto t0 = std::chrono::steady_clock::now();
    double mu = 0.0, mv = 0.0, mp = 0.0;
    for (int iter = 1; iter <= p.maxit; ++iter) {
        Residuals r = step(out.timers);
        out.log.push_back({static_cast<double>(iter), r.u, r.v, r.p});
        out.iterations = iter;
        if (p.rank == 0 && (iter == 1 || iter % 100 == 0))
            std::printf("iter %d  res_u %.6e  res_v %.6e  res_p %.6e\n", iter, r.u, r.v, r.p);
        double tu = r.u, tv = r.v, tp = r.p;
        if (p.residual_normalisation == ResidualNorm::RunningMax) {
            mu = std::max(mu, r.u);
            mv = std::max(mv, r.v);
            mp = std::max(mp, r.p);
            tu = mu > 0.0 ? r.u / mu : 0.0;
            tv = mv > 0.0 ? r.v / mv : 0.0;
            tp = mp > 0.0 ? r.p / mp : 0.0;
        }
        if (std::max({tu, tv, tp}) <= p.error_tolerance && iter > 1) {
            out.converged = true;
            break;
        }
    }
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}
