#include "simple/io.hpp"
#include "simple/params.hpp"
#include "simple/solver.hpp"

#ifdef SIMPLE_HIP
#include "simple/solver_hip.hpp"
#endif

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: simple <config>\n";
        return 1;
    }
    try {
        Params p = make_params(load_config(argv[1]));
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
        write_outputs(p, s, r, argv[1]);
        std::cout << (r.converged ? "converged" : "not converged") << " after " << r.iterations
                  << " iterations in " << r.seconds << " s\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
