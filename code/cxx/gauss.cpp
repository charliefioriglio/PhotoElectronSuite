#include "gauss.h"

CartesianPrimitive::CartesianPrimitive(double a, int x, int y, int z, double c) 
    : alpha(a), lx(x), ly(y), lz(z), coef(c) {
    normalization_constant = calculate_normalization(alpha, lx, ly, lz);
}

double CartesianPrimitive::calculate_normalization(double alpha, int lx, int ly, int lz) {
    int l = lx + ly + lz;
    double prefactor = std::pow(2.0 * alpha / PI, 0.75);
    double numerator = std::pow(4.0 * alpha, l);
    double denom = double_factorial(2 * lx - 1) * 
                   double_factorial(2 * ly - 1) * 
                   double_factorial(2 * lz - 1);
    return prefactor * std::sqrt(numerator / denom);
}

double CartesianPrimitive::evaluate(double x, double y, double z, double cx, double cy, double cz) const {
    double dx = x - cx;
    double dy = y - cy;
    double dz = z - cz;
    double r2 = dx*dx + dy*dy + dz*dz;

    double val = std::exp(-alpha * r2);
    if (lx > 0) {
        if (lx == 1) val *= dx;
        else if (lx == 2) val *= dx * dx;
        else if (lx == 3) val *= dx * dx * dx;
        else val *= std::pow(dx, lx);
    }
    if (ly > 0) {
        if (ly == 1) val *= dy;
        else if (ly == 2) val *= dy * dy;
        else if (ly == 3) val *= dy * dy * dy;
        else val *= std::pow(dy, ly);
    }
    if (lz > 0) {
        if (lz == 1) val *= dz;
        else if (lz == 2) val *= dz * dz;
        else if (lz == 3) val *= dz * dz * dz;
        else val *= std::pow(dz, lz);
    }

    return normalization_constant * val;
}

void ContractedGaussian::add_primitive(double alpha, double coef) {
    primitives.emplace_back(alpha, lx, ly, lz, coef);
}

double ContractedGaussian::evaluate(double x, double y, double z, double cx, double cy, double cz) const {
    double sum = 0.0;
    for (const auto& prim : primitives) {
        sum += prim.coef * prim.evaluate(x, y, z, cx, cy, cz);
    }
    return sum;
}

// Analytical overlap between two primitives on the same center
// For normalization, we only care about overlap on SAME center
// <g1|g2> = integral N1 N2 (x-xc)^(l1+l2) ... exp(-(a1+a2)r^2)
double overlap_same_center(const CartesianPrimitive& p1, const CartesianPrimitive& p2) {
    double alpha_sum = p1.alpha + p2.alpha;
    int Lx = p1.lx + p2.lx;
    int Ly = p1.ly + p2.ly;
    int Lz = p1.lz + p2.lz;

    // Integral x^L exp(-k x^2) dx from -inf to inf
    // If L is odd, 0. If L is even, (L-1)!! / (2k)^(L/2) * sqrt(pi/k)
    // Here we have 3D integral
    
    auto integral_1d = [](int L, double k) {
        if (L % 2 != 0) return 0.0;
        return double_factorial(L - 1) / std::pow(2 * k, L / 2.0) * std::sqrt(PI / k);
    };

    double Ix = integral_1d(Lx, alpha_sum);
    double Iy = integral_1d(Ly, alpha_sum);
    double Iz = integral_1d(Lz, alpha_sum);

    return p1.normalization_constant * p2.normalization_constant * Ix * Iy * Iz;
}

void ContractedGaussian::normalize() {
    double self_overlap = 0.0;
    for (const auto& p1 : primitives) {
        for (const auto& p2 : primitives) {
            self_overlap += p1.coef * p2.coef * overlap_same_center(p1, p2);
        }
    }
    double scale = 1.0 / std::sqrt(self_overlap);
    for (auto& p : primitives) {
        p.coef *= scale;
    }
}
