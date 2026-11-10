// HIP backend (T23). Each kernel is a one-thread-per-cell copy of a CPU loop body; expressions are
// copied verbatim from src/coefficients.cpp, momentum_u.cpp, momentum_v.cpp and pressure.cpp so
// the per-cell rounding is identical (built with -ffp-contract=off, like simple_core).
#include "simple/solver_hip.hpp"

#include "simple/momentum_u.hpp" // momentum_colours

#include <hip/hip_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP_CHECK(x)                                                                            \
    do {                                                                                        \
        hipError_t e_ = (x);                                                                    \
        if (e_ != hipSuccess)                                                                   \
            throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(e_) +      \
                                     " in " #x);                                                \
    } while (0)

namespace {

// ponytail: fixed 64x4 blocks, one launch per colour with idle threads on other colours; T24 tunes.
constexpr int BX = 64, BY = 4, NT = BX * BY;

struct DF { // device view of a Field2D: same row-major layout, index i*cols + j
    double* d;
    int cols;
    __device__ double& operator()(int i, int j) const {
        return d[static_cast<std::size_t>(i) * cols + j];
    }
};

struct DC {
    DF AE, AW, AN, AS, AEE, AWW, ANN, ASS, AP;
};

struct DP { // the Params the kernels read
    int NX, NY, H, Hx, Hy, nrows, ncols, ncol, uds, order;
    double dx, dy, nu, au, av, ap;
};

struct Range { // inclusive cell range; x threads run along j (contiguous)
    int i0, i1, j0, j1;
};

dim3 grid_of(const Range& r) {
    return dim3((r.j1 - r.j0 + 1 + BX - 1) / BX, (r.i1 - r.i0 + 1 + BY - 1) / BY);
}

__device__ inline bool cell(const Range& r, int& i, int& j) {
    j = r.j0 + static_cast<int>(blockIdx.x * BX + threadIdx.x);
    i = r.i0 + static_cast<int>(blockIdx.y * BY + threadIdx.y);
    return i <= r.i1 && j <= r.j1;
}

// ---- coefficients (src/coefficients.cpp) ----

// == std::max(0.0, a) including signed zeros: std::max returns (0.0 < a) ? a : 0.0.
__device__ inline double mx(double a) { return (0.0 < a) ? a : 0.0; }

__device__ inline bool in_band(const DP& p, int i, int j) {
    return j == p.Hy + 1 || j == p.NY + p.Hy || i == p.NX + p.Hx - 1 || i == p.Hx;
}

struct C9 {
    double AE, AW, AN, AS, AEE, AWW, ANN, ASS, AP;
};

__device__ inline C9 convective(const DP& p, double ue, double uw, double vn, double vs, int i,
                                int j) {
    const double dx = p.dx, dy = p.dy;
    C9 c;
    if (p.uds || in_band(p, i, j)) {
        c.AE = dy * mx(-ue);
        c.AW = dy * mx(uw);
        c.AN = dx * mx(-vn);
        c.AS = dx * mx(vs);
        c.AEE = c.AWW = c.ANN = c.ASS = 0.0;
        c.AP = c.AS + c.AW + c.AN + c.AE + c.ASS + c.AWW + c.ANN + c.AEE;
    } else {
        c.AE = -dy * (-0.75 * mx(-ue) + 0.375 * mx(ue) - 0.125 * mx(-uw));
        c.AW = -dy * (-0.125 * mx(ue) + 0.375 * mx(-uw) - 0.75 * mx(uw));
        c.AN = -dx * (-0.75 * mx(-vn) + 0.375 * mx(vn) - 0.125 * mx(-vs));
        c.AS = -dx * (-0.125 * mx(vn) + 0.375 * mx(-vn) - 0.75 * mx(vs));
        c.AEE = -0.125 * dy * mx(-ue);
        c.AWW = -0.125 * dy * mx(uw);
        c.ANN = -0.125 * dx * mx(-vn);
        c.ASS = -0.125 * dx * mx(vs);
        c.AP = c.AS + c.AW + c.AN + c.AE + c.ASS + c.AWW + c.ANN + c.AEE;
    }
    return c;
}

__device__ inline C9 diffusive(const DP& p, int i, int j) {
    const double dx = p.dx, dy = p.dy, nu = p.nu;
    C9 c;
    if (p.order == 2 || in_band(p, i, j)) {
        c.AE = dy * nu / dx;
        c.AW = dy * nu / dx;
        c.AN = dx * nu / dy;
        c.AS = dx * nu / dy;
        c.AEE = c.AWW = c.ANN = c.ASS = 0.0;
        c.AP = -nu * (-2 * dx / dy - 2 * dy / dx);
    } else {
        c.AE = 4.0 / 3 * dy * nu / dx;
        c.AW = 4.0 / 3 * dy * nu / dx;
        c.AN = 4.0 / 3 * dx * nu / dy;
        c.AS = 4.0 / 3 * dx * nu / dy;
        c.AEE = -1.0 / 12 * dy * nu / dx;
        c.AWW = -1.0 / 12 * dy * nu / dx;
        c.ANN = -1.0 / 12 * dx * nu / dy;
        c.ASS = -1.0 / 12 * dx * nu / dy;
        c.AP = -nu * (-2.5 * dx / dy - 2.5 * dy / dx);
    }
    return c;
}

__device__ inline void set_coef(const DC& c, int i, int j, const C9& a, const C9& b) {
    c.AE(i, j) = a.AE + b.AE;
    c.AW(i, j) = a.AW + b.AW;
    c.AN(i, j) = a.AN + b.AN;
    c.AS(i, j) = a.AS + b.AS;
    c.AEE(i, j) = a.AEE + b.AEE;
    c.AWW(i, j) = a.AWW + b.AWW;
    c.ANN(i, j) = a.ANN + b.ANN;
    c.ASS(i, j) = a.ASS + b.ASS;
    c.AP(i, j) = a.AP + b.AP;
}

__device__ inline double nb_sum(const DC& c, const DF& f, int i, int j) {
    return (c.AE(i, j) * f(i, j + 1) + c.AW(i, j) * f(i, j - 1) + c.AN(i, j) * f(i - 1, j) +
            c.AS(i, j) * f(i + 1, j)) +
           (c.AEE(i, j) * f(i, j + 2) + c.AWW(i, j) * f(i, j - 2) + c.ANN(i, j) * f(i - 2, j) +
            c.ASS(i, j) * f(i + 2, j));
}

__device__ inline double near_sum(const DC& c, const DF& f, int i, int j) {
    return c.AE(i, j) * f(i, j + 1) + c.AW(i, j) * f(i, j - 1) + c.AN(i, j) * f(i - 1, j) +
           c.AS(i, j) * f(i + 1, j);
}

__device__ inline int colour(int ncol, int i, int j) {
    return ncol == 2 ? (i + j) % 2 : (i + 3 * j) % 5;
}

// Block sum in a fixed tree, written to part[block]; deterministic for a fixed launch shape.
__device__ inline void block_sum(double val, double* part) {
    __shared__ double sh[NT];
    const int t = threadIdx.y * BX + threadIdx.x;
    sh[t] = val;
    __syncthreads();
    for (int s = NT / 2; s > 0; s >>= 1) {
        if (t < s)
            sh[t] += sh[t + s];
        __syncthreads();
    }
    if (t == 0)
        part[blockIdx.y * gridDim.x + blockIdx.x] = sh[0];
}

// ---- kernels ----

__global__ void k_assemble_u(DP p, DF u, DF v, DC c, Range r) {
    int i, j;
    if (!cell(r, i, j))
        return;
    double ue = (u(i, j) + u(i, j + 1)) / 2;
    double uw = (u(i, j) + u(i, j - 1)) / 2;
    double vn = (v(i, j) + v(i, j + 1)) / 2;
    double vs = (v(i + 1, j) + v(i + 1, j + 1)) / 2;
    set_coef(c, i, j, convective(p, ue, uw, vn, vs, i, j), diffusive(p, i, j));
}

__global__ void k_assemble_v(DP p, DF u, DF v, DC c, Range r) {
    int i, j;
    if (!cell(r, i, j))
        return;
    const double ue = (u(i - 1, j) + u(i, j)) / 2;
    const double uw = (u(i, j - 1) + u(i - 1, j - 1)) / 2;
    const double vn = (v(i, j) + v(i - 1, j)) / 2;
    const double vs = (v(i, j) + v(i + 1, j)) / 2;
    set_coef(c, i, j, convective(p, ue, uw, vn, vs, i, j), diffusive(p, i, j));
}

__global__ void k_bc_u(DP p, DF u) {
    const int H = p.H, j = H + 1 + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (j > p.NY + H)
        return;
    u(H - 1, j) = 2 - u(H, j);
    u(p.NX + H, j) = -u(p.NX + H - 1, j);
}

__global__ void k_bc_v(DP p, DF v) {
    const int H = p.H, i = H + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= p.NX + H)
        return;
    v(i, H) = -v(i, H + 1);
    v(i, p.NY + H + 1) = -v(i, p.NY + H);
}

__global__ void k_residual_u(DP p, DC c, DF u, DF pf, Range r, double* part) {
    int i, j;
    double val = 0.0;
    if (cell(r, i, j)) {
        double R = nb_sum(c, u, i, j) + (pf(i, j) - pf(i, j + 1)) * p.dy - c.AP(i, j) * u(i, j);
        val = fabs(R);
    }
    block_sum(val, part);
}

__global__ void k_residual_v(DP p, DC c, DF v, DF pf, Range r, double* part) {
    int i, j;
    double val = 0.0;
    if (cell(r, i, j)) {
        const double R = nb_sum(c, v, i, j) + (pf(i, j) - pf(i - 1, j)) * p.dx - c.AP(i, j) * v(i, j);
        val = fabs(R);
    }
    block_sum(val, part);
}

__global__ void k_sweep_u(DP p, DC c, DF u, DF pf, Range r, int col) {
    int i, j;
    if (!cell(r, i, j) || colour(p.ncol, i, j) != col)
        return;
    const double a = p.au;
    const double nb = p.ncol == 2 ? near_sum(c, u, i, j) : nb_sum(c, u, i, j);
    u(i, j) = a * (nb + (pf(i, j) - pf(i, j + 1)) * p.dy) / c.AP(i, j) + (1 - a) * u(i, j);
}

__global__ void k_sweep_v(DP p, DC c, DF v, DF pf, Range r, int col) {
    int i, j;
    if (!cell(r, i, j) || colour(p.ncol, i, j) != col)
        return;
    const double av = p.av;
    const double nb = p.ncol == 2 ? near_sum(c, v, i, j) : nb_sum(c, v, i, j);
    v(i, j) = av * (nb + (pf(i, j) - pf(i - 1, j)) * p.dx) / c.AP(i, j) + (1 - av) * v(i, j);
}

struct DPW {
    DF AEP, AWP, ANP, ASP, APP, b, pc;
};

__global__ void k_p_links(DP p, DF APU, DF APV, DPW w, Range r) {
    int i, j;
    if (!cell(r, i, j))
        return;
    const double dx = p.dx, dy = p.dy;
    w.AEP(i, j) = dy * dy / APU(i, j);
    w.AWP(i, j) = dy * dy / APU(i, j - 1);
    w.ANP(i, j) = dx * dx / APV(i, j);
    w.ASP(i, j) = dx * dx / APV(i + 1, j);
}

__global__ void k_p_zero(DP p, DPW w) {
    const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x), H = p.H;
    if (k < p.nrows) {
        w.AWP(k, H + 1) = 0.0;
        w.AEP(k, p.NY + H) = 0.0;
    }
    if (k < p.ncols) {
        w.ASP(p.NX + H - 1, k) = 0.0;
        w.ANP(H, k) = 0.0;
    }
}

__global__ void k_p_rhs(DP p, DF u, DF v, DPW w, Range r, double* part) {
    int i, j;
    double val = 0.0;
    if (cell(r, i, j)) {
        const double dx = p.dx, dy = p.dy;
        w.APP(i, j) = w.AEP(i, j) + w.AWP(i, j) + w.ANP(i, j) + w.ASP(i, j);
        w.b(i, j) = -dx * (-v(i + 1, j) + v(i, j)) - dy * (-u(i, j - 1) + u(i, j));
        val = fabs(w.b(i, j));
    }
    block_sum(val, part);
}

__global__ void k_pc_redblack(DPW w, Range r, int col) {
    int i, j;
    if (!cell(r, i, j) || ((i + j) & 1) != col)
        return;
    double t = w.b(i, j) / w.APP(i, j);
    t += (w.AEP(i, j) * w.pc(i, j + 1) + w.AWP(i, j) * w.pc(i, j - 1) +
          w.ASP(i, j) * w.pc(i + 1, j) + w.ANP(i, j) * w.pc(i - 1, j)) /
         w.APP(i, j);
    w.pc(i, j) = t;
}

__global__ void k_correct(DP p, DPW w, DF APU, DF APV, DF u, DF v, DF pf, Range r) {
    int i, j;
    if (!cell(r, i, j))
        return;
    const int H = p.H;
    if (j <= p.NY + H - 1)
        u(i, j) += p.dy * (w.pc(i, j) - w.pc(i, j + 1)) / APU(i, j);
    if (i >= H + 1)
        v(i, j) += p.dx * (w.pc(i, j) - w.pc(i - 1, j)) / APV(i, j);
    pf(i, j) += p.ap * w.pc(i, j);
}

// Sum of n partials by one block in a fixed order.
__global__ void k_finish(const double* part, int n, double* out) {
    double acc = 0.0;
    const int t = threadIdx.y * BX + threadIdx.x;
    for (int k = t; k < n; k += NT)
        acc += part[k];
    __shared__ double sh[NT];
    sh[t] = acc;
    __syncthreads();
    for (int s = NT / 2; s > 0; s >>= 1) {
        if (t < s)
            sh[t] += sh[t + s];
        __syncthreads();
    }
    if (t == 0)
        out[0] = sh[0];
}

} // namespace

// ---- host side ----

struct HipSolver::Impl {
    Params prm;
    DP p{};
    int rows = 0, cols = 0;
    std::size_t bytes = 0;
    std::vector<double*> owned;
    DF u{}, v{}, pf{};
    DC cu{}, cv{};
    DPW w{};
    double* part = nullptr; // per-block partial sums
    double* out = nullptr;  // one scalar
    Range ru{}, rv_asm{}, ru_asm{}, rres_u{}, rres_v{}, rp{};

    DF upload(const Field2D& f) {
        double* d = nullptr;
        HIP_CHECK(hipMalloc(&d, bytes));
        owned.push_back(d);
        HIP_CHECK(hipMemcpy(d, f.data(), bytes, hipMemcpyHostToDevice));
        return DF{d, cols};
    }
    void get(const DF& d, Field2D& f) const {
        HIP_CHECK(hipMemcpy(f.data(), d.d, bytes, hipMemcpyDeviceToHost));
    }
    DC upload(const Coeffs& c) {
        return DC{upload(c.AE),  upload(c.AW),  upload(c.AN),  upload(c.AS), upload(c.AEE),
                  upload(c.AWW), upload(c.ANN), upload(c.ASS), upload(c.AP)};
    }
    void get(const DC& d, Coeffs& c) const {
        get(d.AE, c.AE), get(d.AW, c.AW), get(d.AN, c.AN), get(d.AS, c.AS), get(d.AEE, c.AEE);
        get(d.AWW, c.AWW), get(d.ANN, c.ANN), get(d.ASS, c.ASS), get(d.AP, c.AP);
    }
    double finish(const Range& r) {
        const dim3 g = grid_of(r);
        const int n = static_cast<int>(g.x * g.y);
        k_finish<<<1, dim3(BX, BY)>>>(part, n, out);
        HIP_CHECK(hipGetLastError());
        double h = 0.0;
        HIP_CHECK(hipMemcpy(&h, out, sizeof h, hipMemcpyDeviceToHost));
        return h;
    }
};

static void check_launch() { HIP_CHECK(hipGetLastError()); }

HipSolver::HipSolver(const Params& prm, const State& s) : d_(std::make_unique<Impl>()) {
    if (prm.momentum_solver != MomentumSolver::GsMulticolour ||
        prm.pressure_solver != PressureSolver::GsRedBlack)
        throw std::runtime_error(
            "backend = gpu supports momentum_solver = gs_multicolour with "
            "pressure_solver = gs_redblack only");
    Impl& d = *d_;
    d.prm = prm;
    d.rows = s.u.rows();
    d.cols = s.u.cols();
    d.bytes = static_cast<std::size_t>(d.rows) * d.cols * sizeof(double);
    const int H = prm.H, NX = prm.NX, NY = prm.NY;
    d.p = DP{NX,      NY,      H,
             prm.N_halo_x, prm.N_halo_y, prm.nrows,
             prm.ncols,    momentum_colours(prm), prm.convection_scheme == Convection::UDS,
             prm.diffusion_order, prm.dx, prm.dy,
             prm.nu,      prm.relaxation_factor_u, prm.relaxation_factor_v,
             prm.relaxation_factor_p};
    // Loop ranges of the CPU kernels, as inclusive (i, j) boxes.
    d.ru_asm = Range{H, NX + H - 1, H, NY + H};          // assemble_u
    d.rv_asm = Range{H, NX + H + 1, H, NY + H + 1};      // assemble_v
    d.ru = Range{H, NX + H - 1, H + 1, NY + H - 1};      // residual_u, sweep u
    d.rres_v = Range{H + 1, NX + H - 1, H + 1, NY + H};  // residual_v, sweep v
    d.rres_u = d.ru;
    d.rp = Range{H, NX + H - 1, H + 1, NY + H};          // assemble_p, p' sweep, correct

    d.u = d.upload(s.u);
    d.v = d.upload(s.v);
    d.pf = d.upload(s.p);
    d.cu = d.upload(s.cu);
    d.cv = d.upload(s.cv);
    d.w = DPW{d.upload(s.pw.AEP), d.upload(s.pw.AWP), d.upload(s.pw.ANP), d.upload(s.pw.ASP),
              d.upload(s.pw.APP), d.upload(s.pw.b),   d.upload(s.pw.pc)};
    const dim3 gmax = grid_of(Range{0, d.rows - 1, 0, d.cols - 1});
    HIP_CHECK(hipMalloc(&d.part, sizeof(double) * gmax.x * gmax.y));
    HIP_CHECK(hipMalloc(&d.out, sizeof(double)));
}

HipSolver::~HipSolver() {
    for (double* q : d_->owned)
        (void)hipFree(q);
    (void)hipFree(d_->part);
    (void)hipFree(d_->out);
}

void HipSolver::assemble_u() {
    Impl& d = *d_;
    k_assemble_u<<<grid_of(d.ru_asm), dim3(BX, BY)>>>(d.p, d.u, d.v, d.cu, d.ru_asm);
    check_launch();
}

void HipSolver::bc_u() {
    Impl& d = *d_;
    k_bc_u<<<(d.prm.NY + NT - 1) / NT, NT>>>(d.p, d.u);
    check_launch();
}

double HipSolver::residual_u() {
    Impl& d = *d_;
    k_residual_u<<<grid_of(d.rres_u), dim3(BX, BY)>>>(d.p, d.cu, d.u, d.pf, d.rres_u, d.part);
    check_launch();
    return d.finish(d.rres_u);
}

void HipSolver::sweep_u(int nsweeps) {
    Impl& d = *d_;
    for (int k = 0; k < nsweeps; ++k)
        for (int c = 0; c < d.p.ncol; ++c) {
            k_sweep_u<<<grid_of(d.ru), dim3(BX, BY)>>>(d.p, d.cu, d.u, d.pf, d.ru, c);
            check_launch();
        }
}

void HipSolver::assemble_v() {
    Impl& d = *d_;
    k_assemble_v<<<grid_of(d.rv_asm), dim3(BX, BY)>>>(d.p, d.u, d.v, d.cv, d.rv_asm);
    check_launch();
}

void HipSolver::bc_v() {
    Impl& d = *d_;
    k_bc_v<<<(d.prm.NX + NT - 1) / NT, NT>>>(d.p, d.v);
    check_launch();
}

double HipSolver::residual_v() {
    Impl& d = *d_;
    k_residual_v<<<grid_of(d.rres_v), dim3(BX, BY)>>>(d.p, d.cv, d.v, d.pf, d.rres_v, d.part);
    check_launch();
    return d.finish(d.rres_v);
}

void HipSolver::sweep_v(int nsweeps) {
    Impl& d = *d_;
    for (int k = 0; k < nsweeps; ++k)
        for (int c = 0; c < d.p.ncol; ++c) {
            k_sweep_v<<<grid_of(d.rres_v), dim3(BX, BY)>>>(d.p, d.cv, d.v, d.pf, d.rres_v, c);
            check_launch();
        }
}

double HipSolver::assemble_p() {
    Impl& d = *d_;
    k_p_links<<<grid_of(d.rp), dim3(BX, BY)>>>(d.p, d.cu.AP, d.cv.AP, d.w, d.rp);
    check_launch();
    const int n = d.rows > d.cols ? d.rows : d.cols;
    k_p_zero<<<(n + NT - 1) / NT, NT>>>(d.p, d.w);
    check_launch();
    k_p_rhs<<<grid_of(d.rp), dim3(BX, BY)>>>(d.p, d.u, d.v, d.w, d.rp, d.part);
    check_launch();
    return d.finish(d.rp);
}

void HipSolver::solve_p() {
    Impl& d = *d_;
    HIP_CHECK(hipMemset(d.w.pc.d, 0, d.bytes)); // CPU: w.pc.fill(0.0)
    for (int k = 0; k < d.prm.maxit_inner_p; ++k)
        for (int c = 0; c < 2; ++c) {
            k_pc_redblack<<<grid_of(d.rp), dim3(BX, BY)>>>(d.w, d.rp, c);
            check_launch();
        }
}

void HipSolver::correct() {
    Impl& d = *d_;
    k_correct<<<grid_of(d.rp), dim3(BX, BY)>>>(d.p, d.w, d.cu.AP, d.cv.AP, d.u, d.v, d.pf, d.rp);
    check_launch();
}

Residuals HipSolver::outer_iteration(PhaseTimers& t) {
    // ponytail: a device sync per phase so the CPU phase timers stay meaningful; costs a little
    // launch overlap. T24 replaces this with hipEvents.
    auto phase = [&](Phase ph, auto&& f) {
        ScopedTimer st(t, ph);
        f();
        HIP_CHECK(hipDeviceSynchronize());
    };
    Residuals r{};
    const int nu = d_->prm.maxit_inner_u, nv = d_->prm.maxit_inner_v;
    phase(Phase::AssembleU, [&] { assemble_u(); });
    phase(Phase::BcU, [&] { bc_u(); });
    phase(Phase::ResidualU, [&] { r.u = residual_u(); });
    phase(Phase::SweepU, [&] { sweep_u(nu); });
    phase(Phase::AssembleV, [&] { assemble_v(); });
    phase(Phase::BcV, [&] { bc_v(); });
    phase(Phase::ResidualV, [&] { r.v = residual_v(); });
    phase(Phase::SweepV, [&] { sweep_v(nv); });
    phase(Phase::AssembleP, [&] { r.p = assemble_p(); });
    phase(Phase::SolveP, [&] { solve_p(); });
    phase(Phase::Correct, [&] { correct(); });
    return r;
}

void HipSolver::download(State& s) const {
    const Impl& d = *d_;
    HIP_CHECK(hipDeviceSynchronize());
    d.get(d.u, s.u);
    d.get(d.v, s.v);
    d.get(d.pf, s.p);
    d.get(d.cu, s.cu);
    d.get(d.cv, s.cv);
    d.get(d.w.AEP, s.pw.AEP), d.get(d.w.AWP, s.pw.AWP), d.get(d.w.ANP, s.pw.ANP);
    d.get(d.w.ASP, s.pw.ASP), d.get(d.w.APP, s.pw.APP), d.get(d.w.b, s.pw.b);
    d.get(d.w.pc, s.pw.pc);
}

RunResult run_simple_hip(const Params& p, State& s) {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("gpu: %s (%s)\n", prop.name, prop.gcnArchName);
    HipSolver g(p, s);
    RunResult out = run_outer_loop(p, [&](PhaseTimers& t) { return g.outer_iteration(t); });
    g.download(s);
    return out;
}
