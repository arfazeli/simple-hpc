// MPI regression: a run on N ranks gives u, v, p byte-identical to the one-rank run, the same
// residual history digit for digit, and stops at the same iteration, for every process grid
// shape R x C with R * C = N (slabs N x 1, columns 1 x N, and blocks such as 2 x 2), for every
// p' solver with an MPI path (gs_redblack, jacobi, cg, mg).
// Run as `mpiexec -n N simple_mpi_tests`; exit code 0 on success on every rank.
#include "simple/decomp.hpp"
#include "simple/solver.hpp"

#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <sstream>
#include <string>

namespace {

Params config(const std::string& extra, const std::string& ps) {
    std::istringstream in("N_cells_x = 32\nN_cells_y = 40\nmomentum_solver = gs_multicolour\n"
                          "pressure_solver = " +
                          ps + "\n" + extra);
    return make_params(parse_config(in));
}

bool same(const Field2D& a, const Field2D& b) {
    return std::memcmp(a.data(), b.data(), sizeof(double) * a.rows() * a.cols()) == 0;
}

// Returns true on rank 0 if fields, residual log and iteration count match the one-rank run (true
// on other ranks).
bool check(const std::string& name, const std::string& extra, int rank, int nranks,
           const std::string& ps = "gs_redblack") {
    const Params global = config(extra, ps);
    Params p = global;
    decompose(p, rank, nranks);
    State s(p);
    // Thread every mg level, so hybrid runs exercise the funnelled exchanges.
    s.mg.parallel_min_cells = 0;
    const RunResult r = run_simple(p, s);

    const int rows = rank == 0 ? global.nrows : 0;
    Field2D u(rows, global.ncols), v(rows, global.ncols), pf(rows, global.ncols);
    gather_field(p, s.u, u);
    gather_field(p, s.v, v);
    gather_field(p, s.p, pf);
    if (rank != 0)
        return true;

    State ref(global);
    const RunResult rr = run_simple(global, ref);
    const bool fields = same(u, ref.u) && same(v, ref.v) && same(pf, ref.p);
    const bool log = r.iterations == rr.iterations && r.log == rr.log; // exact == on doubles
    std::printf("%s on %dx%d ranks (mg split %d): %d iterations; fields %s; residual log %s\n",
                name.c_str(), p.grid_rows, p.grid_cols, p.mg_split, r.iterations,
                fields ? "byte-identical" : "DIFFER", log ? "identical" : "DIFFERS");
    return fields && log;
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0, rank = 0, nranks = 1;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    // Fixed work (tolerance never reached), both colourings.
    const std::string fixed = "Re = 1000\nmaxit = 300\nerror_tolerance = 1e-300\n";
    int ok = 1;
    for (int r = 1; r <= nranks; ++r) {
        if (nranks % r != 0)
            continue;
        const std::string grid =
            "mpi_grid = " + std::to_string(r) + "x" + std::to_string(nranks / r) + "\n";
        ok &=
            check("QUICK/4 (5 colours), 300 its",
                  grid + fixed + "convection_scheme = QUICK\ndiffusion_order = 4\n", rank, nranks);
        ok &= check("UDS/2 (2 colours), 300 its",
                    grid + fixed + "convection_scheme = UDS\ndiffusion_order = 2\n", rank, nranks);
        for (const char* ps : {"jacobi", "cg", "mg"})
            ok &= check(std::string(ps) + ", 300 its", grid + fixed, rank, nranks, ps);
        ok &= check("mg, Re 400 to convergence",
                    grid + "Re = 400\nmaxit = 20000\nerror_tolerance = 1e-4\n"
                           "residual_normalisation = running_max\n",
                    rank, nranks, "mg");
        // Run to convergence: the stopping test sees the same residual bits on every grid.
        ok &= check("Re 400 to convergence",
                    grid + "Re = 400\nmaxit = 20000\nerror_tolerance = 1e-4\n"
                           "residual_normalisation = running_max\n",
                    rank, nranks);
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ok ? 0 : 1;
}
