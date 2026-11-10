#include "simple/timer.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <set>
#include <string>
#include <thread>

TEST_CASE("ScopedTimer records elapsed time", "[timer]") {
    PhaseTimers t;
    {
        ScopedTimer st(t, Phase::SolveP);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const double s = t.seconds[static_cast<std::size_t>(Phase::SolveP)];
    CHECK(s >= 0.004);
    CHECK(s <= 0.050);
    CHECK(t.seconds[static_cast<std::size_t>(Phase::Correct)] == 0.0);
}

TEST_CASE("phase_name covers every phase", "[timer]") {
    std::set<std::string> names;
    for (int i = 0; i < static_cast<int>(Phase::Count); ++i) {
        const char* n = phase_name(static_cast<Phase>(i));
        CHECK(std::strcmp(n, "?") != 0);
        names.insert(n);
    }
    CHECK(names.size() == static_cast<std::size_t>(Phase::Count));
}
