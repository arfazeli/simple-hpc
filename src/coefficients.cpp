#include "simple/coefficients.hpp"

#include <algorithm>

Coef9 operator+(const Coef9& a, const Coef9& b) {
    return {a.AE + b.AE,   a.AW + b.AW,   a.AN + b.AN,   a.AS + b.AS, a.AEE + b.AEE,
            a.AWW + b.AWW, a.ANN + b.ANN, a.ASS + b.ASS, a.AP + b.AP};
}

bool in_boundary_band(const Params& p, int i, int j) {
    return j == p.N_halo_y + 1 || j == p.NY + p.N_halo_y || i == p.NX + p.N_halo_x - 1 ||
           i == p.N_halo_x;
}

Coef9 convective_coefficients(const Params& p, double ue, double uw, double vn, double vs, int i,
                              int j) {
    const double dx = p.dx, dy = p.dy;
    auto mx = [](double a) { return std::max(0.0, a); };
    Coef9 c;
    if (p.convection_scheme == Convection::UDS || in_boundary_band(p, i, j)) {
        c.AE = dy * mx(-ue);
        c.AW = dy * mx(uw);
        c.AN = dx * mx(-vn);
        c.AS = dx * mx(vs);
        c.AEE = c.AWW = c.ANN = c.ASS = 0.0;
        c.AP = c.AS + c.AW + c.AN + c.AE + c.ASS + c.AWW + c.ANN + c.AEE;
    } else {
        c.AE = -dy * (-0.75 * mx(-ue) + 0.375 * mx(ue) - 0.125 * mx(-uw));
        c.AW = -dy * (-0.125 * mx(ue) + 0.375 * mx(-uw) - 0.75 * mx(uw));
        c.AN = -dx * (-0.75 * mx(-vn) + 0.375 * mx(vn) - 0.125 * mx(-vs));
        c.AS = -dx * (-0.125 * mx(vn) + 0.375 * mx(-vn) - 0.75 * mx(vs));
        c.AEE = -0.125 * dy * mx(-ue);
        c.AWW = -0.125 * dy * mx(uw);
        c.ANN = -0.125 * dx * mx(-vn);
        c.ASS = -0.125 * dx * mx(vs);
        c.AP = c.AS + c.AW + c.AN + c.AE + c.ASS + c.AWW + c.ANN + c.AEE;
    }
    return c;
}

Coef9 diffusive_coefficients(const Params& p, int i, int j) {
    const double dx = p.dx, dy = p.dy, nu = p.nu;
    Coef9 c;
    if (p.diffusion_order == 2 || in_boundary_band(p, i, j)) {
        c.AE = dy * nu / dx;
        c.AW = dy * nu / dx;
        c.AN = dx * nu / dy;
        c.AS = dx * nu / dy;
        c.AEE = c.AWW = c.ANN = c.ASS = 0.0;
        c.AP = -nu * (-2 * dx / dy - 2 * dy / dx);
    } else {
        c.AE = 4.0 / 3 * dy * nu / dx;
        c.AW = 4.0 / 3 * dy * nu / dx;
        c.AN = 4.0 / 3 * dx * nu / dy;
        c.AS = 4.0 / 3 * dx * nu / dy;
        c.AEE = -1.0 / 12 * dy * nu / dx;
        c.AWW = -1.0 / 12 * dy * nu / dx;
        c.ANN = -1.0 / 12 * dx * nu / dy;
        c.ASS = -1.0 / 12 * dx * nu / dy;
        c.AP = -nu * (-2.5 * dx / dy - 2.5 * dy / dx);
    }
    return c;
}

Coeffs::Coeffs(int rows, int cols)
    : AE(rows, cols), AW(rows, cols), AN(rows, cols), AS(rows, cols), AEE(rows, cols),
      AWW(rows, cols), ANN(rows, cols), ASS(rows, cols), AP(rows, cols) {}

void Coeffs::set(int i, int j, const Coef9& c) {
    AE(i, j) = c.AE;
    AW(i, j) = c.AW;
    AN(i, j) = c.AN;
    AS(i, j) = c.AS;
    AEE(i, j) = c.AEE;
    AWW(i, j) = c.AWW;
    ANN(i, j) = c.ANN;
    ASS(i, j) = c.ASS;
    AP(i, j) = c.AP;
}
