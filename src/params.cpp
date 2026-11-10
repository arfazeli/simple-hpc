#include "simple/params.hpp"

#include <cstddef>
#include <initializer_list>
#include <iomanip>
#include <set>
#include <stdexcept>
#include <utility>

namespace {
int to_int(const std::string& k, const std::string& v) {
    try {
        std::size_t pos = 0;
        int r = std::stoi(v, &pos);
        if (pos != v.size())
            throw std::invalid_argument("");
        return r;
    } catch (const std::exception&) {
        throw std::runtime_error("cannot parse integer for '" + k + "': " + v);
    }
}
double to_double(const std::string& k, const std::string& v) {
    try {
        std::size_t pos = 0;
        double r = std::stod(v, &pos);
        if (pos != v.size())
            throw std::invalid_argument("");
        return r;
    } catch (const std::exception&) {
        throw std::runtime_error("cannot parse number for '" + k + "': " + v);
    }
}
bool to_bool(const std::string& k, const std::string& v) {
    if (v == "True")
        return true;
    if (v == "False")
        return false;
    throw std::runtime_error("invalid boolean for '" + k + "' (use True/False): " + v);
}
template <class E>
E to_enum(const std::string& k, const std::string& v,
          std::initializer_list<std::pair<const char*, E>> opts) {
    for (auto& o : opts)
        if (v == o.first)
            return o.second;
    throw std::runtime_error("unknown value for '" + k + "': " + v);
}
} // namespace

Params make_params(const ConfigMap& cfg) {
    static const std::set<std::string> known = {
        "N_cells_x",        "N_cells_y",       "N_halo_x",        "N_halo_y",
        "Re",               "maxit_inner_u",   "maxit_inner_v",   "maxit_inner_p",
        "error_tolerance",  "maxit",           "relaxation_factor_u",
        "relaxation_factor_v", "relaxation_factor_p", "convection_scheme",
        "diffusion_order",  "Hybrid_switch",   "backend",
        "momentum_solver",  "pressure_solver", "output_dir", "residual_normalisation",
        "pressure_tolerance", "pressure_maxit", "mg_cycles", "mg_pre", "mg_post", "mg_omega"};
    for (auto& kv : cfg)
        if (!known.count(kv.first))
            throw std::runtime_error("unknown config key: " + kv.first);

    Params p;
    for (auto& kv : cfg) {
        const std::string& k = kv.first;
        const std::string& v = kv.second;
        if (k == "N_cells_x")
            p.NX = to_int(k, v);
        else if (k == "N_cells_y")
            p.NY = to_int(k, v);
        else if (k == "N_halo_x")
            p.N_halo_x = to_int(k, v);
        else if (k == "N_halo_y")
            p.N_halo_y = to_int(k, v);
        else if (k == "Re")
            p.Re = to_double(k, v);
        else if (k == "maxit_inner_u")
            p.maxit_inner_u = to_int(k, v);
        else if (k == "maxit_inner_v")
            p.maxit_inner_v = to_int(k, v);
        else if (k == "maxit_inner_p")
            p.maxit_inner_p = to_int(k, v);
        else if (k == "error_tolerance")
            p.error_tolerance = to_double(k, v);
        else if (k == "maxit")
            p.maxit = to_int(k, v);
        else if (k == "relaxation_factor_u")
            p.relaxation_factor_u = to_double(k, v);
        else if (k == "relaxation_factor_v")
            p.relaxation_factor_v = to_double(k, v);
        else if (k == "relaxation_factor_p")
            p.relaxation_factor_p = to_double(k, v);
        else if (k == "convection_scheme")
            p.convection_scheme =
                to_enum<Convection>(k, v, {{"UDS", Convection::UDS}, {"QUICK", Convection::QUICK}});
        else if (k == "diffusion_order")
            p.diffusion_order = to_int(k, v);
        else if (k == "Hybrid_switch")
            p.Hybrid_switch = to_bool(k, v);
        else if (k == "backend")
            p.backend = to_enum<Backend>(
                k, v,
                {{"serial", Backend::Serial}, {"openmp", Backend::OpenMP}, {"gpu", Backend::GPU}});
        else if (k == "momentum_solver")
            p.momentum_solver = to_enum<MomentumSolver>(
                k, v,
                {{"gs_lex", MomentumSolver::GsLex},
                 {"jacobi", MomentumSolver::Jacobi},
                 {"gs_multicolour", MomentumSolver::GsMulticolour}});
        else if (k == "pressure_solver")
            p.pressure_solver = to_enum<PressureSolver>(
                k, v,
                {{"gs_lex", PressureSolver::GsLex},
                 {"jacobi", PressureSolver::Jacobi},
                 {"gs_redblack", PressureSolver::GsRedBlack},
                 {"cg", PressureSolver::CG},
                 {"mg", PressureSolver::MG}});
        else if (k == "residual_normalisation")
            p.residual_normalisation = to_enum<ResidualNorm>(
                k, v,
                {{"absolute", ResidualNorm::Absolute}, {"running_max", ResidualNorm::RunningMax}});
        else if (k == "output_dir")
            p.output_dir = v;
        else if (k == "pressure_tolerance")
            p.pressure_tolerance = to_double(k, v);
        else if (k == "pressure_maxit")
            p.pressure_maxit = to_int(k, v);
        else if (k == "mg_cycles")
            p.mg_cycles = to_int(k, v);
        else if (k == "mg_pre")
            p.mg_pre = to_int(k, v);
        else if (k == "mg_post")
            p.mg_post = to_int(k, v);
        else if (k == "mg_omega")
            p.mg_omega = to_double(k, v);
    }

    if (p.N_halo_x != p.N_halo_y)
        throw std::runtime_error("N_halo_x must equal N_halo_y");
    p.H = p.N_halo_x;
    if (p.H < 2)
        throw std::runtime_error("halo width H must be >= 2");
    if (p.diffusion_order != 2 && p.diffusion_order != 4)
        throw std::runtime_error("diffusion_order must be 2 or 4");
    if (p.NX < 1 || p.NY < 1)
        throw std::runtime_error("N_cells_x and N_cells_y must be positive");
    if (p.Hybrid_switch)
        throw std::runtime_error("Hybrid_switch = True is not supported");
    if (!(p.pressure_tolerance > 0.0))
        throw std::runtime_error("pressure_tolerance must be positive");
    if (p.pressure_maxit < 1)
        throw std::runtime_error("pressure_maxit must be >= 1");
    if (p.mg_cycles < 1)
        throw std::runtime_error("mg_cycles must be >= 1");
    if (p.mg_pre < 0 || p.mg_post < 0)
        throw std::runtime_error("mg_pre and mg_post must be >= 0");
    if (!(p.mg_omega > 0.0 && p.mg_omega <= 2.0))
        throw std::runtime_error("mg_omega must be in (0, 2]");

    p.nrows = p.NX + 2 * p.H + 1;
    p.ncols = p.NY + 2 * p.H + 1;
    p.dx = 1.0 / p.NY; // column spacing
    p.dy = 1.0 / p.NX; // row spacing
    p.nu = 1.0 / p.Re;
    return p;
}

void print_params(std::ostream& os, const Params& p) {
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
    os << std::setprecision(10);
    os << "NX = " << p.NX << "\nNY = " << p.NY << "\nH = " << p.H << "\n"
       << "nrows = " << p.nrows << "\nncols = " << p.ncols << "\n"
       << "dx = " << p.dx << "\ndy = " << p.dy << "\nRe = " << p.Re << "\nnu = " << p.nu << "\n"
       << "maxit = " << p.maxit << "\nmaxit_inner_u = " << p.maxit_inner_u
       << "\nmaxit_inner_v = " << p.maxit_inner_v << "\nmaxit_inner_p = " << p.maxit_inner_p
       << "\nerror_tolerance = " << p.error_tolerance
       << "\nrelaxation_factor_u = " << p.relaxation_factor_u
       << "\nrelaxation_factor_v = " << p.relaxation_factor_v
       << "\nrelaxation_factor_p = " << p.relaxation_factor_p << "\nconvection_scheme = " << conv
       << "\ndiffusion_order = " << p.diffusion_order
       << "\nHybrid_switch = " << (p.Hybrid_switch ? "True" : "False") << "\nbackend = " << be << "\nmomentum_solver = " << ms << "\npressure_solver = " << ps
       << "\nresidual_normalisation = "
       << (p.residual_normalisation == ResidualNorm::Absolute ? "absolute" : "running_max")
       << "\noutput_dir = " << p.output_dir << "\npressure_tolerance = " << p.pressure_tolerance
       << "\npressure_maxit = " << p.pressure_maxit << "\nmg_cycles = " << p.mg_cycles
       << "\nmg_pre = " << p.mg_pre << "\nmg_post = " << p.mg_post
       << "\nmg_omega = " << p.mg_omega << "\n";
}
