#include "simple/config.hpp"
#include "simple/solver.hpp"
#include "support/compare.hpp"
#include "support/csv.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
#ifdef _OPENMP
void check_threads(const std::string& name, int threads) {
    const std::string root = SIMPLE_SOURCE_DIR;
    Params p = make_params(load_config(root + "/configs/snapshot/" + name + ".cfg"));
    State s(p);
    omp_set_num_threads(threads);
    RunResult r = run_simple(p, s);

    const std::string dir = root + "/tests/snapshots/" + name + "/";
    REQUIRE(max_rel_diff(s.u, load_csv(dir + "u.csv")) == 0.0);
    REQUIRE(max_rel_diff(s.v, load_csv(dir + "v.csv")) == 0.0);
    REQUIRE(max_rel_diff(s.p, load_csv(dir + "p.csv")) == 0.0);

    auto alog = load_csv_rows(dir + "alog.csv", true);
    REQUIRE(static_cast<int>(alog.size()) == r.iterations);
    for (std::size_t n = 0; n < alog.size(); ++n)
        for (int k = 1; k < 4; ++k) {
            const double ref = alog[n][k];
            REQUIRE(std::fabs(r.log[n][k] - ref) <= 1e-12 * std::fabs(ref));
        }
}
#endif
} // namespace

TEST_CASE("threads: snapshots identical for 1, 2, 4, 8 threads", "[threads]") {
#ifdef _OPENMP
    const int saved = omp_get_max_threads();
    for (int t : {1, 2, 4, 8}) {
        INFO("threads = " << t);
        check_threads("short_16", t);
        check_threads("converged_32x40", t);
    }
    omp_set_num_threads(saved);
#else
    SKIP("built without OpenMP");
#endif
}
