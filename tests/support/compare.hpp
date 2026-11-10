#pragma once
#include "simple/field2d.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

inline double max_rel_diff(const Field2D& a, const Field2D& ref, double abs_floor = 1e-14) {
    if (a.rows() != ref.rows() || a.cols() != ref.cols())
        throw std::runtime_error("max_rel_diff: shape mismatch");
    double m = 0.0;
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j)
            m = std::max(m, std::fabs(a(i, j) - ref(i, j)) /
                                std::max(std::fabs(ref(i, j)), abs_floor));
    return m;
}
