#include "simple/config.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <sstream>
#include <stdexcept>

static ConfigMap parse(const std::string& s) {
    std::istringstream in(s);
    return parse_config(in);
}

TEST_CASE("config parses key = value with whitespace") {
    auto m = parse("  a =  1  \nb=two\n");
    REQUIRE(m.size() == 2);
    REQUIRE(m["a"] == "1");
    REQUIRE(m["b"] == "two");
}

TEST_CASE("config ignores comments and blank lines") {
    auto m = parse("# header\n\na = 1 # trailing\n   \n# x = 2\n");
    REQUIRE(m.size() == 1);
    REQUIRE(m["a"] == "1");
}

TEST_CASE("config duplicate key throws with line number") {
    REQUIRE_THROWS_WITH(parse("a = 1\nb = 2\na = 3\n"),
                        Catch::Matchers::ContainsSubstring("line 3"));
}

TEST_CASE("config malformed line throws with line number") {
    REQUIRE_THROWS_WITH(parse("a = 1\nnonsense\n"), Catch::Matchers::ContainsSubstring("line 2"));
    REQUIRE_THROWS_AS(parse("= 1\n"), std::runtime_error);
}

TEST_CASE("load_config reads a config file and rejects a missing file") {
    auto m = load_config(std::string(SIMPLE_SOURCE_DIR) + "/configs/cavity_re100_32.cfg");
    REQUIRE(m.at("N_cells_x") == "32");
    REQUIRE_THROWS_AS(load_config("/nonexistent/x.cfg"), std::runtime_error);
}
