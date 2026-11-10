#include "simple/config.hpp"
#include "simple/solver.hpp"
#include "support/compare.hpp"
#include "support/csv.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <string>

namespace {
void check_snapshot(const std::string& name) {
    const std::string root = SIMPLE_SOURCE_DIR;
    Params p = make_params(load_config(root + "/configs/snapshot/" + name + ".cfg"));
    State s(p);
    RunResult r = run_simple(p, s);

    const std::string dir = root + "/tests/snapshots/" + name + "/";
    const double du = max_rel_diff(s.u, load_csv(dir + "u.csv"));
    const double dv = max_rel_diff(s.v, load_csv(dir + "v.csv"));
    const double dp = max_rel_diff(s.p, load_csv(dir + "p.csv"));
    std::printf("snapshot %s: max rel diff u %.3e v %.3e p %.3e\n", name.c_str(), du, dv, dp);
    REQUIRE(du == 0.0);
    REQUIRE(dv == 0.0);
    REQUIRE(dp == 0.0);

    auto alog = load_csv_rows(dir + "alog.csv", true);
    REQUIRE(static_cast<int>(alog.size()) == r.iterations);
    const auto& last = r.log.back();
    for (int k = 0; k < 4; ++k)
        REQUIRE(alog.back()[k] == last[k]);
}
} // namespace

TEST_CASE("snapshot: short_16 matches bit for bit", "[snapshot]") {
    check_snapshot("short_16");
}

TEST_CASE("snapshot: converged_32x40 matches bit for bit", "[snapshot]") {
    check_snapshot("converged_32x40");
}
