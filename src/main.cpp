#include "simple/decomp.hpp"
#include "simple/io.hpp"
#include "simple/params.hpp"
#include "simple/solver.hpp"

#ifdef SIMPLE_HIP
#include "simple/solver_hip.hpp"
#endif

#include <iostream>
#include <stdexcept>
#include <string>

#ifdef SIMPLE_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv) {
    int rank = 0, nranks = 1;
#ifdef SIMPLE_MPI
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif
    int rc = 0;
    if (argc != 2) {
        if (rank == 0)
            std::cerr << "usage: simple <config>\n";
        rc = 1;
    } else {
        try {
            const Params global = make_params(load_config(argv[1]));
            Params p = global;
            decompose(p, rank, nranks);
            if (rank == 0)
                print_params(std::cout, p);
            State s(p);
            RunResult r;
            if (p.backend == Backend::GPU) {
#ifdef SIMPLE_HIP
                r = run_simple_hip(p, s);
#else
                throw std::runtime_error("backend = gpu needs a build with -DSIMPLE_HIP=ON");
#endif
            } else {
                r = run_simple(p, s);
            }
            r.ranks = nranks;
#ifdef SIMPLE_MPI
            if (nranks > 1) {
                const CommTimes& ct = comm_times();
                double mine[2] = {ct.halo, ct.collective}, sum[2], max[2];
                MPI_Reduce(mine, sum, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                MPI_Reduce(mine, max, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                r.comm = {sum[0] / nranks, max[0], sum[1] / nranks, max[1]};
            }
#endif
            if (nranks == 1) {
                write_outputs(p, s.u, s.v, s.p, r, argv[1]);
            } else { // rank 0 gathers the three output fields (only rank 0 allocates them)
                const int rows = rank == 0 && p.write_fields ? global.nrows : 0;
                Field2D u(rows, global.ncols), v(rows, global.ncols), pf(rows, global.ncols);
                if (p.write_fields) {
                    gather_field(p, s.u, u);
                    gather_field(p, s.v, v);
                    gather_field(p, s.p, pf);
                }
                if (rank == 0)
                    write_outputs(global, u, v, pf, r, argv[1]);
            }
            if (rank == 0)
                std::cout << (r.converged ? "converged" : "not converged") << " after "
                          << r.iterations << " iterations in " << r.seconds << " s"
                          << (nranks > 1 ? " on " + std::to_string(nranks) + " ranks" : "") << "\n";
        } catch (const std::exception& e) {
            std::cerr << "error (rank " << rank << "): " << e.what() << "\n";
#ifdef SIMPLE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#endif
            rc = 1;
        }
    }
#ifdef SIMPLE_MPI
    MPI_Finalize();
#endif
    return rc;
}
