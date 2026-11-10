#include "simple/field2d.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Field2D shape and init") {
    Field2D f(3, 4, 2.5);
    REQUIRE(f.rows() == 3);
    REQUIRE(f.cols() == 4);
    REQUIRE(f(2, 3) == 2.5);
    Field2D e;
    REQUIRE(e.rows() == 0);
}

TEST_CASE("Field2D row-major indexing") {
    Field2D f(3, 4);
    f(1, 2) = 7.0;
    REQUIRE(f.data()[1 * 4 + 2] == 7.0);
    const Field2D& c = f;
    REQUIRE(c(1, 2) == 7.0);
    REQUIRE(c.data() == f.data());
}

TEST_CASE("Field2D fill") {
    Field2D f(2, 2, 1.0);
    f.fill(3.0);
    for (int k = 0; k < 4; ++k)
        REQUIRE(f.data()[k] == 3.0);
}
