#ifndef PHYSICAL_DIPOLE_H
#define PHYSICAL_DIPOLE_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <tuple>
#include <vector>

class PhysicalDipoleAngular {
public:
  // Solves the angular equation for the Physical Dipole system.
  // Returns: {Eigenvalues, Eigenvectors (columns), l_values (indices)}
  // The coefficients in Eigenvectors correspond to the P_l^m basis functions
  // where l = l_values[i].
  //
  // D is the parameter used in the radial/angular equation.
  static std::tuple<std::vector<double>, std::vector<std::vector<double>>,
                    std::vector<int>>
  Solve(int m, int l_max, double E, double a, double D);

  // Evaluate the angular function T(eta) at a given eta using the coefficients
  // from Solve.
  static double Evaluate(double eta, int m, const std::vector<double> &coeffs,
                         const std::vector<int> &l_vals);
};

class PhysicalDipoleRadial {
public:
  // Solve for the radial parameter nu (Complex)
  static std::complex<double> SolveNu(double c, int m, double Alm,
                                      double Alm_c0);

  // Compute coefficients (Complex)
  static std::map<int, std::complex<double>>
  ComputeCoefficients(std::complex<double> nu, double c, int m, double Alm);

  // Evaluate S(xi) (Complex)
  static std::complex<double>
  Evaluate(double xi, const std::map<int, std::complex<double>> &coeffs,
           std::complex<double> nu, double c, int m);

  // [Preserve Numeric Solver as fallback/hybrid]
  static std::pair<std::vector<double>, std::vector<double>>
  SolveNumeric(double c, int m, double Alm, double xi_max);
  static double InterpolateResult(
      double xi,
      const std::pair<std::vector<double>, std::vector<double>> &result);

private:
  // Recurrence coefficients (Complex nu)
  static std::complex<double> AlphaL(int L, double c, int m,
                                     std::complex<double> nu);
  static std::complex<double> BetaL(int L, double c, int m,
                                    std::complex<double> nu, double Alm);
  static std::complex<double> GammaL(int L, double c, int m,
                                     std::complex<double> nu);

  // Continued Fraction Evaluation (Complex)
  static std::complex<double> EvaluateCF(double c, int m,
                                         std::complex<double> nu, double Alm,
                                         bool right_side);

  // Characteristic Equation (Complex)
  static std::complex<double> CharacteristicEq(std::complex<double> nu,
                                               double c, int m, double Alm);
};

// High-level class to manage the full 3D wavefunction
class PhysicalDipole {
private:
  double a;
  double D;

public:
  PhysicalDipole(double a_val, double D_val) : a(a_val), D(D_val) {}

  // Struct to hold solution for a specific (E, m)
  struct Solution {
    // Angular
    std::vector<double> ang_eigenvalues;
    std::vector<std::vector<double>> ang_eigenvectors;
    std::vector<int> l_vals;

    // Radial (Map from mode index n -> Radial Coeffs + Nu)
    // Actually radial depends on lambda (eigenvalue).
    // Each mode n has a lambda_n -> Alm_n.
    // And thus a specific nu_n and set of radial coeffs.
    struct ModeRadial {
      std::complex<double> nu;
      std::map<int, std::complex<double>> coeffs;
    };
    std::vector<ModeRadial> radial_solutions;

    double k; // wavenumber associated with E
    double c; // c = k*a
  };

  // Solve for a specific Energy and m
  Solution Solve(double E, int m, int l_max);

  // Evaluate a specific mode n at Cartesian point (x,y,z)
  // Returns: Psi_{n,m}(r)
  std::complex<double> EvaluateMode(double x, double y, double z, int m,
                                    int n_mode, const Solution &sol);

  // Evaluate just the angular part S_{mn}(eta)
  // eta = cos(theta)
  double EvaluateAngular(double eta, int m, int n_mode, const Solution &sol);
};

#endif
