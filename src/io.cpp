#include "simple/io.hpp"

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

void write_csv(const std::string& path, const Field2D& f) {
    std::FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp)
        throw std::runtime_error("cannot open for writing: " + path);
    for (int i = 0; i < f.rows(); ++i) {
        for (int j = 0; j < f.cols(); ++j)
            std::fprintf(fp, j ? ",%.18e" : "%.18e", f(i, j));
        std::fputc('\n', fp);
    }
    std::fclose(fp);
}

void cell_centre(const Params& p, const Field2D& u, const Field2D& v, Field2D& ux, Field2D& uy) {
    const int H = p.H;
    ux = Field2D(p.NX, p.NY);
    uy = Field2D(p.NX, p.NY);
    for (int r = 0; r < p.NX; ++r)
        for (int c = 0; c < p.NY; ++c) {
            ux(r, c) = (u(H + r, H + c) + u(H + r, H + c + 1)) / 2;
            uy(r, c) = (v(H + r, H + 1 + c) + v(H + r + 1, H + 1 + c)) / 2;
        }
}

static std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\')
            o += '\\';
        o += c;
    }
    return o;
}

void write_outputs(const Params& p, const Field2D& u, const Field2D& v, const Field2D& pf,
                   const RunResult& r, const std::string& cfg_path) {
    namespace fs = std::filesystem;
    PhaseTimers timers = r.timers;
    const fs::path d(p.output_dir);
    {
    ScopedTimer out_timer(timers, Phase::Output);
    fs::create_directories(p.output_dir);
    write_csv((d / "u.csv").string(), u);
    write_csv((d / "v.csv").string(), v);
    write_csv((d / "p.csv").string(), pf);
    Field2D ux, uy;
    cell_centre(p, u, v, ux, uy);
    write_csv((d / "ux_centre.csv").string(), ux);
    write_csv((d / "uy_centre.csv").string(), uy);

    std::FILE* al = std::fopen((d / "alog.csv").string().c_str(), "w");
    if (!al)
        throw std::runtime_error("cannot write alog.csv");
    std::fputs("Iteration,Residual_u,Residual_v,Residual_p\n", al);
    for (auto& row : r.log)
        std::fprintf(al, "%d,%.18e,%.18e,%.18e\n", static_cast<int>(row[0]), row[1], row[2],
                     row[3]);
    std::fclose(al);
    }

    {
        // total = sum of all phases. Output happens after r.seconds was measured, so
        // unaccounted = seconds - (total - output).
        double total = 0.0;
        for (double x : timers.seconds)
            total += x;
        const double timed = total - timers.seconds[static_cast<std::size_t>(Phase::Output)];
        std::FILE* tf = std::fopen((d / "timing.csv").string().c_str(), "w");
        if (!tf)
            throw std::runtime_error("cannot write timing.csv");
        std::fputs("phase,seconds,fraction\n", tf);
        const double denom = total > 0.0 ? total : 1.0;
        for (std::size_t i = 0; i < timers.seconds.size(); ++i)
            std::fprintf(tf, "%s,%.9f,%.6f\n", phase_name(static_cast<Phase>(i)),
                         timers.seconds[i], timers.seconds[i] / denom);
        std::fprintf(tf, "total,%.9f,1.000000\n", total);
        std::fprintf(tf, "unaccounted,%.9f,%.6f\n", r.seconds - timed,
                     r.seconds > 0.0 ? (r.seconds - timed) / r.seconds : 0.0);
        std::fclose(tf);
    }

    const char* conv = p.convection_scheme == Convection::UDS ? "UDS" : "QUICK";
    const char* be = p.backend == Backend::Serial   ? "serial"
                     : p.backend == Backend::OpenMP ? "openmp"
                                                    : "gpu";
    const char* ms = p.momentum_solver == MomentumSolver::GsLex    ? "gs_lex"
                     : p.momentum_solver == MomentumSolver::Jacobi ? "jacobi"
                                                                   : "gs_multicolour";
    const char* ps = p.pressure_solver == PressureSolver::GsLex        ? "gs_lex"
                     : p.pressure_solver == PressureSolver::Jacobi     ? "jacobi"
                     : p.pressure_solver == PressureSolver::GsRedBlack ? "gs_redblack"
                     : p.pressure_solver == PressureSolver::CG         ? "cg"
                                                                       : "mg";
    std::FILE* fp = std::fopen((d / "run_info.json").string().c_str(), "w");
    if (!fp)
        throw std::runtime_error("cannot write run_info.json");
    std::fprintf(fp, "{\n  \"config\": \"%s\",\n  \"git_hash\": \"%s\",\n", esc(cfg_path).c_str(),
                 SIMPLE_GIT_HASH);
    std::fprintf(fp, "  \"compiler\": \"%s\",\n", SIMPLE_COMPILER);
#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    std::fprintf(fp, "  \"backend\": \"%s\",\n  \"threads\": %d,\n", be, nthreads);
    std::fprintf(fp, "  \"momentum_solver\": \"%s\",\n  \"pressure_solver\": \"%s\",\n", ms, ps);
    std::fprintf(fp, "  \"N_cells_x\": %d,\n  \"N_cells_y\": %d,\n  \"N_halo_x\": %d,\n", p.NX,
                 p.NY, p.N_halo_x);
    std::fprintf(fp, "  \"N_halo_y\": %d,\n  \"Re\": %.17g,\n", p.N_halo_y, p.Re);
    std::fprintf(fp, "  \"maxit_inner_u\": %d,\n  \"maxit_inner_v\": %d,\n", p.maxit_inner_u,
                 p.maxit_inner_v);
    std::fprintf(fp, "  \"maxit_inner_p\": %d,\n  \"error_tolerance\": %.17g,\n", p.maxit_inner_p,
                 p.error_tolerance);
    std::fprintf(fp, "  \"maxit\": %d,\n", p.maxit);
    std::fprintf(fp, "  \"relaxation_factor_u\": %.17g,\n  \"relaxation_factor_v\": %.17g,\n",
                 p.relaxation_factor_u, p.relaxation_factor_v);
    std::fprintf(fp, "  \"relaxation_factor_p\": %.17g,\n", p.relaxation_factor_p);
    std::fprintf(fp, "  \"convection_scheme\": \"%s\",\n  \"diffusion_order\": %d,\n", conv,
                 p.diffusion_order);
    std::fprintf(fp, "  \"Hybrid_switch\": %s,\n", p.Hybrid_switch ? "true" : "false");
    std::fprintf(fp, "  \"residual_normalisation\": \"%s\",\n",
                 p.residual_normalisation == ResidualNorm::Absolute ? "absolute" : "running_max");
    std::fprintf(fp, "  \"output_dir\": \"%s\",\n", esc(p.output_dir).c_str());
    std::fprintf(fp, "  \"pressure_tolerance\": %.17g,\n  \"pressure_maxit\": %d,\n",
                 p.pressure_tolerance, p.pressure_maxit);
    std::fprintf(fp, "  \"mg_cycles\": %d,\n  \"mg_pre\": %d,\n  \"mg_post\": %d,\n", p.mg_cycles,
                 p.mg_pre, p.mg_post);
    std::fprintf(fp, "  \"mg_omega\": %.17g,\n", p.mg_omega);
    std::fprintf(fp, "  \"cg_iterations_total\": %lld,\n  \"cg_iterations_mean\": %.6f,\n",
                 r.cg_iterations,
                 r.iterations > 0 ? static_cast<double>(r.cg_iterations) / r.iterations : 0.0);
    std::fprintf(fp, "  \"iterations\": %d,\n  \"converged\": %s,\n  \"seconds\": %.6f,\n  \"seconds_per_iteration\": %.9f\n}\n",
                 r.iterations, r.converged ? "true" : "false", r.seconds,
                 r.iterations > 0 ? r.seconds / r.iterations : 0.0);
    std::fclose(fp);
}
