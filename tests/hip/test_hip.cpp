// T23: HIP backend against the CPU kernels. Built only with -DSIMPLE_HIP=ON; needs a GPU.
#include "simple/momentum_u.hpp"
#include "simple/momentum_v.hpp"
#include "simple/solver_hip.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>

namespace {

ConfigMap config(const std::string& rel, const std::string& scheme, const std::string& order,
                 int maxit) {
    ConfigMap c = load_config(std::string(SIMPLE_SOURCE_DIR) + "/" + rel);
    c["momentum_solver"] = "gs_multicolour";
    c["pressure_solver"] = "gs_redblack";
    c["convection_scheme"] = scheme;
    c["diffusion_order"] = order;
    c["maxit"] = std::to_string(maxit);
    c["error_tolerance"] = "0";
    return c;
}

// "" if bit-identical, else where the first difference is and the max relative difference.
std::string diff(const Field2D& gpu, const Field2D& cpu) {
    const std::size_t n = static_cast<std::size_t>(cpu.rows()) * cpu.cols();
    if (std::memcmp(gpu.data(), cpu.data(), n * sizeof(double)) == 0)
        return "";
    std::ostringstream os;
    double m = 0.0;
    int fi = -1, fj = -1;
    for (int i = 0; i < cpu.rows(); ++i)
        for (int j = 0; j < cpu.cols(); ++j) {
            const double a = gpu(i, j), b = cpu(i, j);
            if (std::memcmp(&a, &b, sizeof a) != 0 && fi < 0)
                fi = i, fj = j;
            m = std::max(m, std::fabs(a - b) / std::max(std::fabs(b), 1e-300));
        }
    os.precision(17);
    os << "first diff at (" << fi << "," << fj << "): gpu " << gpu(fi, fj) << " cpu "
       << cpu(fi, fj) << "; max rel diff " << m;
    return os.str();
}

#define SAME(gpu, cpu)                                                                          \
    do {                                                                                        \
        const std::string d_ = diff(gpu, cpu);                                                  \
        INFO(#cpu << ": " << d_);                                                               \
        CHECK(d_.empty());                                                                      \
    } while (0)

void same_coeffs(const Coeffs& g, const Coeffs& c) {
    SAME(g.AE, c.AE);
    SAME(g.AW, c.AW);
    SAME(g.AN, c.AN);
    SAME(g.AS, c.AS);
    SAME(g.AEE, c.AEE);
    SAME(g.AWW, c.AWW);
    SAME(g.ANN, c.ANN);
    SAME(g.ASS, c.ASS);
    SAME(g.AP, c.AP);
}

bool rel_close(double a, double b) { return std::fabs(a - b) <= 1e-12 * std::max(std::fabs(b), 1e-300); }

void per_phase(const std::string& scheme, const std::string& order) {
    // Non-trivial starting fields: 3 CPU outer iterations on the 32x40 snapshot case.
    const Params p = make_params(config("configs/snapshot/converged_32x40.cfg", scheme, order, 3));
    State cpu(p);
    run_simple(p, cpu);
    HipSolver gpu(p, cpu);
    State g(p);

    assemble_u(p, cpu.u, cpu.v, cpu.cu);
    gpu.assemble_u();
    gpu.download(g);
    INFO("phase assemble_u");
    same_coeffs(g.cu, cpu.cu);

    apply_bc_u(p, cpu.u);
    gpu.bc_u();
    gpu.download(g);
    INFO("phase bc_u");
    SAME(g.u, cpu.u);

    {
        const double rc = residual_u(p, cpu.cu, cpu.u, cpu.p), rg = gpu.residual_u();
        INFO("residual_u gpu " << rg << " cpu " << rc);
        CHECK(rel_close(rg, rc));
    }

    sweep_u_batch(p, cpu.cu, cpu.u, cpu.p, cpu.scratch, p.maxit_inner_u);
    gpu.sweep_u(p.maxit_inner_u);
    gpu.download(g);
    INFO("phase sweep_u");
    SAME(g.u, cpu.u);

    assemble_v(p, cpu.u, cpu.v, cpu.cv);
    gpu.assemble_v();
    gpu.download(g);
    INFO("phase assemble_v");
    same_coeffs(g.cv, cpu.cv);

    apply_bc_v(p, cpu.v);
    gpu.bc_v();
    gpu.download(g);
    INFO("phase bc_v");
    SAME(g.v, cpu.v);

    {
        const double rc = residual_v(p, cpu.cv, cpu.v, cpu.p), rg = gpu.residual_v();
        INFO("residual_v gpu " << rg << " cpu " << rc);
        CHECK(rel_close(rg, rc));
    }

    sweep_v_batch(p, cpu.cv, cpu.v, cpu.p, cpu.scratch, p.maxit_inner_v);
    gpu.sweep_v(p.maxit_inner_v);
    gpu.download(g);
    INFO("phase sweep_v");
    SAME(g.v, cpu.v);

    {
        const double rc = assemble_p(p, cpu.cu.AP, cpu.cv.AP, cpu.u, cpu.v, cpu.pw);
        const double rg = gpu.assemble_p();
        gpu.download(g);
        INFO("phase assemble_p; res_p gpu " << rg << " cpu " << rc);
        CHECK(rel_close(rg, rc));
        SAME(g.pw.AEP, cpu.pw.AEP);
        SAME(g.pw.AWP, cpu.pw.AWP);
        SAME(g.pw.ANP, cpu.pw.ANP);
        SAME(g.pw.ASP, cpu.pw.ASP);
        SAME(g.pw.APP, cpu.pw.APP);
        SAME(g.pw.b, cpu.pw.b);
    }

    solve_pc_redblack(p, cpu.pw);
    gpu.solve_p();
    gpu.download(g);
    INFO("phase solve_p");
    SAME(g.pw.pc, cpu.pw.pc);

    correct(p, cpu.pw, cpu.cu.AP, cpu.cv.AP, cpu.u, cpu.v, cpu.p);
    gpu.correct();
    gpu.download(g);
    INFO("phase correct");
    SAME(g.u, cpu.u);
    SAME(g.v, cpu.v);
    SAME(g.p, cpu.p);
}

} // namespace

TEST_CASE("HIP per-phase kernels match the CPU bit for bit (QUICK, order 4: 5 colours)",
          "[hip]") {
    per_phase("QUICK", "4");
}

TEST_CASE("HIP per-phase kernels match the CPU bit for bit (UDS, order 2: 2 colours)", "[hip]") {
    per_phase("UDS", "2");
}

TEST_CASE("HIP fixed-work run matches the CPU (200 outer iterations, 32x40)", "[hip]") {
    const Params p = make_params(config("configs/snapshot/converged_32x40.cfg", "QUICK", "4", 200));
    State cpu(p), gpu(p);
    run_simple(p, cpu);
    run_simple_hip(p, gpu);
    SAME(gpu.u, cpu.u);
    SAME(gpu.v, cpu.v);
    SAME(gpu.p, cpu.p);
}

TEST_CASE("HIP runs are deterministic (identical residual history)", "[hip]") {
    const Params p = make_params(config("configs/snapshot/short_16.cfg", "QUICK", "4", 50));
    State a(p), b(p);
    const RunResult ra = run_simple_hip(p, a), rb = run_simple_hip(p, b);
    REQUIRE(ra.log.size() == rb.log.size());
    CHECK(std::memcmp(ra.log.data(), rb.log.data(), ra.log.size() * sizeof ra.log[0]) == 0);
    SAME(a.u, b.u);
}
