#include "simple/solver.hpp"

#include "simple/momentum_u.hpp"
#include "simple/momentum_v.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>

State::State(const Params& p)
    : u(p.nrows, p.ncols), v(p.nrows, p.ncols), p(p.nrows, p.ncols), cu(p.nrows, p.ncols),
      cv(p.nrows, p.ncols), pw(p.nrows, p.ncols), scratch(p.nrows, p.ncols),
      cg(p.nrows, p.ncols),
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
        if (iter == 1 || iter % 100 == 0)
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
