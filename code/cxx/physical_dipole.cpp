#include "physical_dipole.h"
#include "linalg.h"
#include "tools.h"
#include <cmath>
#include <iostream>
#include <numeric>

// Factorial wrapper using double to avoid overflow for l=10+
// n! = tgamma(n+1)
double factorial_double(int n) { return std::tgamma(n + 1.0); }

std::tuple<std::vector<double>, std::vector<std::vector<double>>,
           std::vector<int>>
PhysicalDipoleAngular::Solve(int m, int l_max, double E, double a, double D) {

  m = std::abs(m);
  double c_val = std::sqrt(2.0 * E * a * a);
  double c2 = c_val * c_val;

  // l values: m, m+1, ..., l_max
  std::vector<int> ell_vals;
  for (int l = m; l <= l_max; ++l) {
    ell_vals.push_back(l);
  }

  int N = ell_vals.size();
  if (N == 0)
    return {};

  // Initialize H (N x N) and S_diag (size N)
  std::vector<std::vector<double>> H(N, std::vector<double>(N, 0.0));
  std::vector<double> S_diag(N);

  for (int i = 0; i < N; ++i) {
    int l = ell_vals[i];

    // Calculate S diagonal
    // S = 2 * (l+m)! / ((2l+1) * (l-m)!)
    double num = 2.0 * factorial_double(l + m);
    double den = (2.0 * l + 1.0) * factorial_double(l - m);
    S_diag[i] = num / den;
  }

  for (int i = 0; i < N; ++i) {
    int l = ell_vals[i];

    // Diagonal Elements H[i, i]
    if (l - m >= 0) {
      double f1 = -1.0 * (l * (l + 1)) * S_diag[i];

      double term1 =
          (double)((l + m) * (l - m)) / ((2.0 * l + 1.0) * (2.0 * l - 1.0));
      double term2 = (double)((l - m + 1) * (l + m + 1)) /
                     ((2.0 * l + 1.0) * (2.0 * l + 3.0));
      double f2 = -c2 * (term1 + term2) * S_diag[i];

      H[i][i] += f1 + f2;
    }

    // Off-diagonal i+1, i (and i, i+1)
    if (i + 1 < N) {
      double num = 2.0 * factorial_double(l + m + 1);
      double den = (2.0 * l + 3.0) * factorial_double(l - m + 1);
      double f = (-2.0 * D / (2.0 * l + 1.0)) * (l + 1 - m);
      f *= (num / den);

      H[i + 1][i] += f;
      H[i][i + 1] += f;
    }

    // Off-diagonal i+2, i
    if (i + 2 < N) {
      double num = 2.0 * factorial_double(l + m + 2);
      double den = (2.0 * l + 1.0) * (2.0 * l + 3.0) * (2.0 * l + 5.0) *
                   factorial_double(l - m + 2);
      double f = -c2 * (l - m + 1) * (l - m + 2);
      f *= (num / den);

      H[i + 2][i] += f;
      H[i][i + 2] += f;
    }
  }

  // Transform to Standard Eigenvalue Problem: H' c' = lambda c'
  // form H' = S^{-1/2} H S^{-1/2}
  // where S is diagonal. S^{-1/2} is diagonal with elements 1/sqrt(S_ii)

  std::vector<double> inv_sqrt_S(N);
  for (int i = 0; i < N; ++i) {
    inv_sqrt_S[i] = 1.0 / std::sqrt(S_diag[i]);
  }

  std::vector<std::vector<double>> H_prime(N, std::vector<double>(N));
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      H_prime[i][j] = H[i][j] * inv_sqrt_S[i] * inv_sqrt_S[j];
    }
  }

  // Diagonalize H_prime
  std::vector<double> eigenvalues_unsorted;
  std::vector<std::vector<double>> eigenvectors_prime;
  LinearAlgebra::Jacobi(N, H_prime, eigenvalues_unsorted, eigenvectors_prime);

  // Sort Descending
  std::vector<std::pair<double, int>> idx(N);
  for (int i = 0; i < N; ++i)
    idx[i] = {eigenvalues_unsorted[i], i};

  std::sort(idx.begin(), idx.end(),
            [](auto &a, auto &b) { return a.first > b.first; });

  std::vector<double> eigenvalues(N);
  std::vector<std::vector<double>> eigenvectors(N, std::vector<double>(N));

  for (int i = 0; i < N; ++i) {
    eigenvalues[i] = idx[i].first;
    int original_col = idx[i].second;

    // Recover original eigenvector c: c = S^{-1/2} c'
    // c'_k is eigenvectors_prime[k][j] (j is the eigenvector index)

    for (int k = 0; k < N; ++k) {
      double c_prime_k = eigenvectors_prime[k][original_col];
      eigenvectors[k][i] = inv_sqrt_S[k] * c_prime_k;
    }
  }

  return {eigenvalues, eigenvectors, ell_vals};
}

// Helper to compute P_l^m(x) unnormalized
// Matches Boost/C++17 definition with (-1)^m phase
double legendre_P(int l, int m, double x) {
  int abs_m = std::abs(m);
  if (abs_m > l)
    return 0.0;

  // 1. P_m^m
  double somx2 = std::sqrt((1.0 - x) * (1.0 + x));
  double cur = 1.0;
  // (2m-1)!!
  for (int i = 1; i <= abs_m; i++)
    cur *= (2 * i - 1);

  // Condon-Shortley Phase (-1)^m
  if (abs_m % 2 != 0)
    cur = -cur;

  cur *= std::pow(somx2, abs_m);

  if (l == abs_m) {
    return cur;
  }

  // 2. P_{m+1}^m
  double P_m_m = cur;
  double P_mp1_m = x * (2 * abs_m + 1) * P_m_m;

  if (l == abs_m + 1) {
    return P_mp1_m;
  }

  // 3. P_l^m recursion
  double P_prev = P_mp1_m;
  double P_prev2 = P_m_m;
  double P_curr = 0.0;

  for (int ll = abs_m + 2; ll <= l; ll++) {
    P_curr = ((2.0 * ll - 1.0) * x * P_prev - (ll + abs_m - 1.0) * P_prev2) /
             (ll - abs_m);
    P_prev2 = P_prev;
    P_prev = P_curr;
  }

  return P_curr;
}

double PhysicalDipoleAngular::Evaluate(double eta, int m,
                                       const std::vector<double> &coeffs,
                                       const std::vector<int> &l_vals) {
  double val = 0.0;
  for (size_t i = 0; i < coeffs.size(); ++i) {
    if (std::abs(coeffs[i]) < 1e-12)
      continue;
    int l = l_vals[i];
    double plm = legendre_P(l, std::abs(m), eta);
    val += coeffs[i] * plm;
  }
  return val;
}

// ==========================================
// Physical Dipole Radial Logic
// ==========================================

#include "math_special.h"
#include <map>

// --------------------------------------------------------------------------
// Super-Critical (Numeric) Solver Implementation
// For m=0, 2Da > 0.639
// --------------------------------------------------------------------------

// 4-term recurrence coefficients for power series a_k
// a_k = -1/(2k^2) * [ (c^2 - lambda + k(k-1)) a_{k-1} + 2c^2 a_{k-2} + c^2
// a_{k-3} ] Base: a_0 = const (set to 1 temporarily)
//       a_1 = -1/2 * (c^2 - lambda) * a_0
//       a_k = 0 for k < 0

std::pair<std::vector<double>, std::vector<double>>
PhysicalDipoleRadial::SolveNumeric(double c, int m, double Alm, double xi_max) {
  if (m != 0) {
    std::cerr << "Error: SolveNumeric only implemented for m=0" << std::endl;
    return {};
  }

  // Parameters
  double lambda = -Alm; // Alm is -1 * eigenvalue
  double c2 = c * c;
  int k_max = 50; // Number of series terms

  // 1. Compute Series Coefficients a_k
  std::vector<double> a(k_max + 1, 0.0);
  a[0] = 1.0;

  // a_1
  a[1] = -0.5 * (c2 - lambda) * a[0];

  for (int k = 2; k <= k_max; ++k) {
    double term1 = (c2 - lambda + (double)k * (k - 1.0)) * a[k - 1];
    double term2 = 2.0 * c2 * a[k - 2];
    double term3 = (k >= 3) ? c2 * a[k - 3] : 0.0;

    a[k] = -1.0 / (2.0 * k * k) * (term1 + term2 + term3);
  }

  // 2. Evaluate Series near xi=1 (e.g. up to xi=1.1)
  // S(xi) = Sum a_k (xi-1)^k
  double xi_start = 1.00001;
  double xi_switch = 1.1;
  double h = 0.001; // Step size

  std::vector<double> xi_vals;
  std::vector<double> S_vals;

  // Fill initial points using series
  for (double xi = xi_start; xi <= xi_switch; xi += h) {
    double rho = xi - 1.0;
    double val = 0.0;
    // double derivative = 0.0; // S' needed for RK4 start

    double pow_rho = 1.0;
    for (int k = 0; k <= k_max; ++k) {
      val += a[k] * pow_rho;
      if (k > 0) { /* derivative += ... */
      }
      pow_rho *= rho;
    }

    xi_vals.push_back(xi);
    S_vals.push_back(val);

    // Store last state for integration
    if (std::abs(xi - xi_switch) < h / 2.0) {
      // Exact boundary state
    }
  }

  // 3. Outward Integration (RK4)
  // Equation: [(xi-1)+2]S'' + [2 + 2/(xi-1)]S' + [c^2(xi-1) + 2c^2 +
  // (c^2-lambda)/(xi-1)]S = 0 S'' = - { [2 + 2/(xi-1)]S' + [c^2(xi-1) + 2c^2 +
  // (c^2-lambda)/(xi-1)]S } / (xi+1)

  auto S_double_prime = [&](double x, double S, double Sp) {
    double xm1 = x - 1.0;
    double xp1 = x + 1.0;

    double term_Sp = (2.0 + 2.0 / xm1) * Sp;
    double term_S = (c2 * xm1 + 2.0 * c2 + (c2 - lambda) / xm1) * S;

    return -(term_Sp + term_S) / xp1;
  };

  // Get last point from series
  double x_curr = xi_vals.back();

  // Re-evaluate exact S and S' at x_curr using series for precision
  double S_curr = 0.0;
  double Sp_curr = 0.0;
  double rho = x_curr - 1.0;
  double pow_rho = 1.0;
  for (int k = 0; k <= k_max; ++k) {
    S_curr += a[k] * pow_rho;
    if (k > 0)
      Sp_curr += k * a[k] * std::pow(rho, k - 1.0);
    pow_rho *= rho;
  }
  // Update last stored value to be consistent
  S_vals.back() = S_curr;

  while (x_curr < xi_max) {
    // RK4 for 2nd order ODE: y'' = f(x, y, y')
    // k1_y = h * y'
    // k1_yp = h * f(x, y, y')
    // ...

    double k1_S = h * Sp_curr;
    double k1_Sp = h * S_double_prime(x_curr, S_curr, Sp_curr);

    double k2_S = h * (Sp_curr + 0.5 * k1_Sp);
    double k2_Sp = h * S_double_prime(x_curr + 0.5 * h, S_curr + 0.5 * k1_S,
                                      Sp_curr + 0.5 * k1_Sp);

    double k3_S = h * (Sp_curr + 0.5 * k2_Sp);
    double k3_Sp = h * S_double_prime(x_curr + 0.5 * h, S_curr + 0.5 * k2_S,
                                      Sp_curr + 0.5 * k2_Sp);

    double k4_S = h * (Sp_curr + k3_Sp);
    double k4_Sp =
        h * S_double_prime(x_curr + h, S_curr + k3_S, Sp_curr + k3_Sp);

    S_curr += (k1_S + 2.0 * k2_S + 2.0 * k3_S + k4_S) / 6.0;
    Sp_curr += (k1_Sp + 2.0 * k2_Sp + 2.0 * k3_Sp + k4_Sp) / 6.0;
    x_curr += h;

    xi_vals.push_back(x_curr);
    S_vals.push_back(S_curr);
  }

  // 4. Asymptotic Normalization
  // Theory: S(xi) -> C' * sin(c*xi + delta) / (c*xi)
  // We want to normalize such that the asymptotic amplitude matches a unit
  // plane wave (1/x decay). Envelope is A / xi. Target envelope is 1 / (c *
  // xi). So we need to scale by (1/c) / A.

  // Estimate envelope A = max(|S| * xi) in the tail
  double max_envelope_A = 0.0;

  // Scan tail (last 500 points or so, assuming oscillating)
  size_t scan_start =
      (xi_vals.size() > 1000) ? xi_vals.size() - 1000 : xi_vals.size() / 2;

  for (size_t i = scan_start; i < xi_vals.size(); ++i) {
    double envelope_val = std::abs(S_vals[i]) * xi_vals[i];
    if (envelope_val > max_envelope_A) {
      max_envelope_A = envelope_val;
    }
  }

  double norm_factor = 1.0;
  if (max_envelope_A > 1e-15) {
    // Current envelope: S ~ A / xi
    // Target: S ~ 1 / (c * xi)
    // Factor = (1/c) / A
    norm_factor = (1.0 / c) / max_envelope_A;
  }

  for (auto &v : S_vals)
    v *= norm_factor;

  return {xi_vals, S_vals};
}

double PhysicalDipoleRadial::InterpolateResult(
    double xi,
    const std::pair<std::vector<double>, std::vector<double>> &result) {
  const auto &x_vec = result.first;
  const auto &y_vec = result.second;

  if (x_vec.empty())
    return 0.0;
  if (xi <= x_vec.front())
    return y_vec.front();
  if (xi >= x_vec.back())
    return y_vec.back();

  // Binary search
  auto it = std::lower_bound(x_vec.begin(), x_vec.end(), xi);
  size_t idx = std::distance(x_vec.begin(), it);
  if (idx == 0)
    idx = 1;

  double x1 = x_vec[idx - 1];
  double x2 = x_vec[idx];
  double y1 = y_vec[idx - 1];
  double y2 = y_vec[idx];

  double t = (xi - x1) / (x2 - x1);
  return y1 + t * (y2 - y1);
}

// ==========================================
// Complex Implementation
// ==========================================

using Complex = std::complex<double>;

Complex PhysicalDipoleRadial::AlphaL(int L, double c, int m, Complex nu) {
  Complex L_nu = (Complex)L + nu;
  Complex denom = (2.0 * L_nu + 3.0) * (2.0 * L_nu + 5.0);
  if (std::abs(denom) < 1e-15)
    return 0.0;

  Complex num = -c * c * (L_nu - (double)m + 1.0) * (L_nu - (double)m + 2.0);
  return num / denom;
}

Complex PhysicalDipoleRadial::BetaL(int L, double c, int m, Complex nu,
                                    double Alm) {
  Complex L_nu = (Complex)L + nu;
  Complex denom = (2.0 * L_nu - 1.0) * (2.0 * L_nu + 3.0);
  Complex term1(0.0, 0.0);

  if (std::abs(denom) > 1e-15) {
    Complex inside = 2.0 * (L_nu * (L_nu + 1.0) - (double)(m * m)) - 1.0;
    term1 = c * c * inside / denom;
  }

  Complex term2 = L_nu * (L_nu + 1.0) - Alm;
  return term1 + term2;
}

Complex PhysicalDipoleRadial::GammaL(int L, double c, int m, Complex nu) {
  Complex L_nu = (Complex)L + nu;
  Complex denom = (2.0 * L_nu - 1.0) * (2.0 * L_nu - 3.0);
  if (std::abs(denom) < 1e-15)
    return 0.0;

  Complex num = -c * c * (L_nu + (double)m) * (L_nu + (double)m - 1.0);
  return num / denom;
}

Complex PhysicalDipoleRadial::EvaluateCF(double c, int m, Complex nu,
                                         double Alm, bool right_side) {
  int max_terms = 100;
  Complex cf_val(0.0, 0.0);

  if (right_side) {
    for (int i = max_terms; i >= 1; --i) {
      int L = 2 * i;
      Complex alpha = AlphaL(L, c, m, nu);
      Complex gamma = GammaL(L + 2, c, m, nu);
      Complex beta = BetaL(L, c, m, nu, Alm);

      if (i == max_terms) {
        cf_val = beta;
      } else {
        if (std::abs(cf_val) < 1e-20)
          cf_val = 1e-20;
        cf_val = beta - alpha * gamma / cf_val;
      }
    }
    if (std::abs(cf_val) < 1e-20)
      return 0.0;
    Complex alpha0 = AlphaL(0, c, m, nu);
    Complex gamma2 = GammaL(2, c, m, nu);
    return alpha0 * gamma2 / cf_val;
  } else {
    for (int i = max_terms; i >= 1; --i) {
      int L = -2 * i;
      Complex alpha = AlphaL(L - 2, c, m, nu);
      Complex gamma = GammaL(L, c, m, nu);
      Complex beta = BetaL(L, c, m, nu, Alm);

      if (i == max_terms) {
        cf_val = beta;
      } else {
        if (std::abs(cf_val) < 1e-20)
          cf_val = 1e-20;
        cf_val = beta - alpha * gamma / cf_val;
      }
    }
    if (std::abs(cf_val) < 1e-20)
      return 0.0;
    Complex alpha_neg2 = AlphaL(-2, c, m, nu);
    Complex gamma0 = GammaL(0, c, m, nu);
    return alpha_neg2 * gamma0 / cf_val;
  }
}

Complex PhysicalDipoleRadial::CharacteristicEq(Complex nu, double c, int m,
                                               double Alm) {
  Complex beta0 = BetaL(0, c, m, nu, Alm);
  Complex right = EvaluateCF(c, m, nu, Alm, true);

  // Regularized left continued fraction at nu = 1.5
  Complex cf_val(0.0, 0.0);
  int max_terms = 100;
  for (int i = max_terms; i >= 2; --i) {
    int L = -2 * i;
    Complex alpha = AlphaL(L - 2, c, m, nu);
    Complex gamma = GammaL(L, c, m, nu);
    Complex beta = BetaL(L, c, m, nu, Alm);

    if (i == max_terms) {
      cf_val = beta;
    } else {
      if (std::abs(cf_val) < 1e-20)
        cf_val = 1e-20;
      cf_val = beta - alpha * gamma / cf_val;
    }
  }

  Complex tilde_gamma0 =
      -c * c * (nu + (double)m) * (nu + (double)m - 1.0) / (2.0 * nu - 1.0);
  Complex tilde_alpha_neg4 = -c * c * (nu - 3.0 - (double)m) *
                             (nu - 2.0 - (double)m) / (2.0 * nu - 5.0);

  Complex beta_neg2 = BetaL(-2, c, m, nu, Alm);
  Complex gamma_neg2 = GammaL(-2, c, m, nu);
  Complex alpha_neg2 = AlphaL(-2, c, m, nu);

  Complex denom =
      (2.0 * nu - 3.0) * beta_neg2 - tilde_alpha_neg4 * gamma_neg2 / cf_val;
  Complex left = alpha_neg2 * tilde_gamma0 / denom;

  return beta0 - left - right;
}

// Helper: check if nu is near a half-integer singularity
static bool IsHalfInteger(Complex z, double eps = 0.03) {
  if (std::abs(z.imag()) > eps)
    return false; // Complex nu can't be a half-integer singularity
  double r = z.real();
  // Check if r is close to n + 0.5 for any integer n
  double frac = r - std::floor(r);
  return (std::abs(frac - 0.5) < eps);
}

Complex PhysicalDipoleRadial::SolveNu(double target_c, int m, double Alm,
                                      double Alm_c0) {
  // Compute L_eff at c=0 using Alm_c0 as starting point
  double disc = 1.0 + 4.0 * Alm_c0;
  Complex L_eff;
  if (disc >= 0) {
    L_eff = Complex(0.5 * (-1.0 + std::sqrt(disc)), 0.0);
  } else {
    L_eff = Complex(-0.5, 0.5 * std::sqrt(-disc));
  }

  if (target_c < 1e-10)
    return L_eff;

  // Homotopy continuation: step c from 0 to target_c
  int n_steps = std::max(30, int(std::ceil(target_c / 0.02)));
  double dc = target_c / n_steps;

  Complex current_nu = L_eff;

  for (int step = 1; step <= n_steps; ++step) {
    double current_c = step * dc;
    double current_Alm = Alm_c0 + (Alm - Alm_c0) * (current_c / target_c);

    Complex z0 = current_nu;
    Complex z1 = current_nu + Complex(1e-4, 1e-4);
    Complex f0 = CharacteristicEq(z0, current_c, m, current_Alm);
    Complex f1 = CharacteristicEq(z1, current_c, m, current_Alm);

    Complex best_z = current_nu;
    double best_res = std::abs(f0);
    bool converged = false;

    for (int iter = 0; iter < 80; ++iter) {
      Complex df = f1 - f0;
      if (std::abs(df) < 1e-15)
        break;

      Complex dz = -f1 * (z1 - z0) / df;
      if (std::abs(dz) > 0.2) {
        dz = dz * (0.2 / std::abs(dz));
      }
      Complex z2 = z1 + dz;
      Complex f2 = CharacteristicEq(z2, current_c, m, current_Alm);
      bool is_h = IsHalfInteger(z2);

      if (std::abs(f2) < best_res && !is_h) {
        best_z = z2;
        best_res = std::abs(f2);
      }

      if (std::abs(f2) < 1e-8 && !is_h) {
        current_nu = z2;
        converged = true;
        break;
      }

      z0 = z1;
      f0 = f1;
      z1 = z2;
      f1 = f2;
    }
    if (!converged) {
      if (best_res < 1e-5 && !IsHalfInteger(best_z)) {
        current_nu = best_z;
      }
    }
  }
  return current_nu;
}

std::map<int, Complex> PhysicalDipoleRadial::ComputeCoefficients(Complex nu,
                                                                 double c,
                                                                 int m,
                                                                 double Alm) {
  std::map<int, Complex> coeffs;
  coeffs[0] = 1.0;
  int L_max = 30;

  // Positive L
  for (int L = 2; L <= L_max; L += 2) {
    Complex cf_ratio(0.0, 0.0);
    for (int i = 50; i > 0; --i) {
      int L_star = L + 2 * i;
      Complex a = AlphaL(L_star, c, m, nu);
      Complex g = GammaL(L_star + 2, c, m, nu);
      Complex b = BetaL(L_star, c, m, nu, Alm);
      if (i == 50)
        cf_ratio = b;
      else {
        if (std::abs(cf_ratio) < 1e-25)
          cf_ratio = 1e-25;
        cf_ratio = b - a * g / cf_ratio;
      }
    }
    Complex g_L = GammaL(L, c, m, nu);
    if (std::abs(cf_ratio) < 1e-25)
      cf_ratio = 1e-25;
    Complex ratio = -g_L / cf_ratio;

    coeffs[L] = ratio * coeffs[L - 2];
  }

  // Negative L
  for (int L = -2; L >= -L_max; L -= 2) {
    Complex cf_ratio(0.0, 0.0);
    for (int i = 50; i > 0; --i) {
      int L_star = L - 2 * i;
      Complex a = AlphaL(L_star - 2, c, m, nu);
      Complex g = GammaL(L_star, c, m, nu);
      Complex b = BetaL(L_star, c, m, nu, Alm);
      if (i == 50)
        cf_ratio = b;
      else {
        if (std::abs(cf_ratio) < 1e-25)
          cf_ratio = 1e-25;
        cf_ratio = b - a * g / cf_ratio;
      }
    }
    Complex a_L = AlphaL(L, c, m, nu);
    if (std::abs(cf_ratio) < 1e-25)
      cf_ratio = 1e-25;
    Complex ratio = -a_L / cf_ratio;
    coeffs[L] = ratio * coeffs[L + 2];
  }

  // Normalize
  double total = 0.0;
  for (auto const &[key, val] : coeffs)
    total += std::abs(val);
  for (auto &[key, val] : coeffs)
    val /= total;

  return coeffs;
}

Complex PhysicalDipoleRadial::Evaluate(double xi,
                                       const std::map<int, Complex> &coeffs,
                                       Complex nu, double c, int m) {
  if (xi <= 1.0)
    xi = 1.0000001;

  double prefactor = 1.0;
  if (m != 0) {
    double ratio = (xi * xi - 1.0) / (xi * xi);
    prefactor = std::pow(ratio, std::abs(m) / 2.0);
  }

  Complex sum(0.0, 0.0);
  for (auto const &[L, a_L] : coeffs) {
    if (std::abs(a_L) < 1e-15)
      continue;

    Complex order = (Complex)L + nu;
    double arg = c * xi;
    Complex bessel_val = MathSpecial::SphericalBesselJComplex(order, arg);

    sum += a_L * bessel_val;
  }

  Complex phase = std::exp(Complex(0.0, 1.0) * (M_PI * 0.5 * nu));

  // Asymptotic Amplitude Normalization
  Complex sum_alt(0.0, 0.0);
  for (auto const &[L, a_L] : coeffs) {
    int k = L / 2;
    double sign = (k % 2 == 0) ? 1.0 : -1.0;
    sum_alt += a_L * sign;
  }
  double Y = M_PI * 0.5 * nu.imag();
  double A_asymp = std::abs(sum_alt) * std::exp(-Y) * std::cosh(Y);
  if (A_asymp < 1e-15)
    A_asymp = 1e-15;

  return (prefactor * sum * phase) / A_asymp;
}

// ==========================================
// High-Level PhysicalDipole Implementation
// ==========================================

PhysicalDipole::Solution PhysicalDipole::Solve(double E, int m, int l_max) {
  Solution sol;
  sol.k = std::sqrt(2.0 * E); // Atomic units
  sol.c = sol.k * this->a;

  // 1. Solve Angular
  auto ang_res = PhysicalDipoleAngular::Solve(m, l_max, E, this->a, this->D);
  sol.ang_eigenvalues = std::get<0>(ang_res);
  sol.ang_eigenvectors = std::get<1>(ang_res);
  sol.l_vals = std::get<2>(ang_res);

  // 2. Solve Radial for each mode
  int n_modes = sol.ang_eigenvalues.size();
  sol.radial_solutions.resize(n_modes);

  // Solve angular eigensystem at E=0 to get c=0 eigenvalues (point dipole
  // limit)
  auto ang_res_c0 =
      PhysicalDipoleAngular::Solve(m, l_max, 0.0, this->a, this->D);
  const auto &ang_eigenvalues_c0 = std::get<0>(ang_res_c0);

  for (int n = 0; n < n_modes; ++n) {
    // Alm = -lambda.
    double lambda = sol.ang_eigenvalues[n];
    double Alm = -lambda;
    double Alm_c0 = -ang_eigenvalues_c0[n];

    // Solve Radial
    sol.radial_solutions[n].nu =
        PhysicalDipoleRadial::SolveNu(sol.c, m, Alm, Alm_c0);
    sol.radial_solutions[n].coeffs = PhysicalDipoleRadial::ComputeCoefficients(
        sol.radial_solutions[n].nu, sol.c, m, Alm);
  }

  return sol;
}

std::complex<double> PhysicalDipole::EvaluateMode(double x, double y, double z,
                                                  int m, int n_mode,
                                                  const Solution &sol) {
  double zA = this->a;
  double zB = -this->a;

  double rA = std::sqrt(x * x + y * y + (z - zA) * (z - zA));
  double rB = std::sqrt(x * x + y * y + (z - zB) * (z - zB));

  double xi = (rA + rB) / (2.0 * this->a);
  double eta = (rA - rB) / (2.0 * this->a);
  double phi = std::atan2(y, x);

  // Bounds check
  if (xi < 1.0)
    xi = 1.0;
  if (eta > 1.0)
    eta = 1.0;
  if (eta < -1.0)
    eta = -1.0;

  // 1. Radial Part S(xi)
  const auto &rad_sol = sol.radial_solutions[n_mode];
  std::complex<double> S =
      PhysicalDipoleRadial::Evaluate(xi, rad_sol.coeffs, rad_sol.nu, sol.c, m);

  // 2. Angular Part T(eta)
  // Coefficients in sol.ang_eigenvectors[column n_mode]
  std::vector<double> ang_coeffs(sol.l_vals.size());
  for (size_t i = 0; i < sol.l_vals.size(); ++i) {
    ang_coeffs[i] = sol.ang_eigenvectors[i][n_mode];
  }
  double T = PhysicalDipoleAngular::Evaluate(eta, m, ang_coeffs, sol.l_vals);

  if (m < 0) {
    if (std::abs(m) % 2 != 0)
      T = -T;
  }

  // 3. Azimuthal Phi(phi)
  // exp(i m phi) / sqrt(2pi)
  std::complex<double> Phi = std::exp(std::complex<double>(0, m * phi));
  Phi /= std::sqrt(2.0 * M_PI);

  // Energy normalization: sqrt(2*k/pi)
  // Since S(xi) asymptotically scales as 1/(k*r), multiplying by
  // sqrt(2*k/pi) yields the correct energy-normalized asymptotic amplitude
  // of sqrt(2/(pi*k))/r.
  double energy_norm = std::sqrt(2.0 * sol.k / M_PI);

  return S * T * Phi * energy_norm;
}

double PhysicalDipole::EvaluateAngular(double eta, int m, int n_mode,
                                       const Solution &sol) {
  if (n_mode < 0 || n_mode >= (int)sol.l_vals.size())
    return 0.0;

  // Bounds check
  if (eta > 1.0)
    eta = 1.0;
  if (eta < -1.0)
    eta = -1.0;

  // Extract coeffs for this mode
  std::vector<double> ang_coeffs(sol.l_vals.size());
  for (size_t i = 0; i < sol.l_vals.size(); ++i) {
    ang_coeffs[i] = sol.ang_eigenvectors[i][n_mode];
  }

  double T = PhysicalDipoleAngular::Evaluate(eta, m, ang_coeffs, sol.l_vals);

  if (m < 0) {
    if (std::abs(m) % 2 != 0)
      T = -T;
  }

  return T;
}
