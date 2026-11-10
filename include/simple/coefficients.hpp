#pragma once
#include "simple/field2d.hpp"
#include "simple/params.hpp"

struct Coef9 {
    double AE, AW, AN, AS, AEE, AWW, ANN, ASS, AP;
};

Coef9 operator+(const Coef9& a, const Coef9& b);

// True for j == H+1, j == NY+H, i == NX+H-1, i == H (SPEC 3.3).
bool in_boundary_band(const Params& p, int i, int j);

Coef9 convective_coefficients(const Params& p, double ue, double uw, double vn, double vs, int i,
                              int j);
Coef9 diffusive_coefficients(const Params& p, int i, int j);

struct Coeffs {
    Field2D AE, AW, AN, AS, AEE, AWW, ANN, ASS, AP;
    Coeffs() = default;
    Coeffs(int rows, int cols);
    void set(int i, int j, const Coef9& c);
};
