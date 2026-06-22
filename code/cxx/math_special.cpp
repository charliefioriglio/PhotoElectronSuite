#include "math_special.h"
#include <cmath>
#include <complex>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace MathSpecial {

    double CylBesselJ(double nu, double x) {
        if (x < 0) return 0.0; // Assume x>=0
        if (x < 1e-10) {
            return (nu < 1e-10) ? 1.0 : 0.0;
        }

        // Series expansion for small x
        if (x < 12.0) {
             double x2 = x * 0.5;
             double term = std::pow(x2, nu) / std::tgamma(nu + 1);
             double sum = term;
             double x2_sq = - x2 * x2; // - (x/2)^2
             
             for (int k = 1; k < 100; ++k) {
                 term *= x2_sq / (k * (nu + k));
                 sum += term;
                 if (std::abs(term) < 1e-15 * std::abs(sum)) break;
             }
             return sum;
        } else {
             // Asymptotic expansion (Hanckel's)
             double mu = 4.0 * nu * nu;
             double term8x = 8.0 * x;
             
             double P = 1.0 - (mu - 1.0)*(mu - 9.0) / (2.0 * term8x * term8x);
             double Q = (mu - 1.0) / term8x - (mu - 1.0)*(mu - 9.0)*(mu - 25.0) / (6.0 * std::pow(term8x, 3));
             
             double chi = x - (nu * 0.5 + 0.25) * M_PI;
             double sqrt_factor = std::sqrt(2.0 / (M_PI * x));
             
             return sqrt_factor * (P * std::cos(chi) - Q * std::sin(chi));
        }
    }

    double SphericalBesselJ(int l, double x) {
        if (std::abs(x) < 1e-8) {
            return (l == 0) ? 1.0 : 0.0;
        }

        if (l < x) {
             double j_0 = std::sin(x) / (x * x) - std::cos(x) / x;
             
             if (l==0) return std::sin(x)/x;
             if (l==1) return j_0;

             double j_curr = j_0; // j1
             double j_prev = std::sin(x)/x; // j0
             
             for (int i = 2; i <= l; ++i) {
                 double factor = (2.0 * (i - 1) + 1.0) / x;
                 double temp = factor * j_curr - j_prev;
                 j_prev = j_curr;
                 j_curr = temp;
             }
             return j_curr;
        } else {
            // Backward Recursion
            int l_start = l + 20 + int(x); 
            
            double j_next = 0.0;
            double j_curr = 1.0e-30;
            
            double computed_j_l = 0.0;
            
            for (int k = l_start; k >= 1; --k) {
                 double factor = (2.0 * k + 1.0) / x;
                 double j_prev = factor * j_curr - j_next;
                 
                 if (k == l) computed_j_l = j_curr; 
                 if ((k-1) == l) computed_j_l = j_prev; 

                 j_next = j_curr;
                 j_curr = j_prev;
            }
            double true_j0 = std::sin(x) / x;
            double scale = true_j0 / j_curr;
            
            return computed_j_l * scale;
        }
    }

    double SphericalBesselJGeneric(double nu, double x) {
        if (x < 0) return 0.0; 
        if (x < 1e-10) {
           return (std::abs(nu) < 1e-10) ? 1.0 : 0.0;
        }

        // j_nu(x) = sqrt(pi/(2x)) * J_{nu+0.5}(x)
        double prefactor = std::sqrt(M_PI / (2.0 * x));
        double val = CylBesselJ(nu + 0.5, x);
        return prefactor * val;
    }

    // Helper: Lanczos approximation for Gamma(z)
    // Forward declaration for recursion
    std::complex<double> ComplexGamma(std::complex<double> z) {
         // Coefficients for g=7
         const double p[] = {
             0.99999999999980993, 676.5203681218851, -1259.13921700441,
             771.32342877765313, -176.61502916214059, 12.507343278686905,
             -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
         };
         
         if (z.real() < 0.5) {
             // Reflection formula: Gamma(1-z)Gamma(z) = pi/sin(pi*z)
             return M_PI / (std::sin(M_PI * z) * ComplexGamma(1.0 - z));
         }
         
         z -= 1.0;
         std::complex<double> x = p[0];
         for (int i=1; i<9; i++) x += p[i] / (z + (double)i);
         
         std::complex<double> t = z + 7.5;
         return std::sqrt(2 * M_PI) * std::pow(t, z + 0.5) * std::exp(-t) * x;
    }

    std::complex<double> SphericalBesselJComplex(std::complex<double> nu, double x) {
        if (x < 1e-10) {
            return (std::abs(nu) < 1e-10) ? 1.0 : 0.0;
        }
        if (x > 30.0) {
            std::complex<double> phase = x - (nu * M_PI / 2.0);
            return std::sin(phase) / x;
        }
        std::complex<double> nu_cyl = nu + 0.5;
        std::complex<double> gamma_nu_cyl_plus_1 = ComplexGamma(nu_cyl + 1.0);
        return SphericalBesselJComplex(nu, x, gamma_nu_cyl_plus_1);
    }

    std::complex<double> SphericalBesselJComplex(std::complex<double> nu, double x, std::complex<double> gamma_nu_cyl_plus_1) {
        if (x < 1e-10) {
            return (std::abs(nu) < 1e-10) ? 1.0 : 0.0;
        }
        if (x > 30.0) {
            std::complex<double> phase = x - (nu * M_PI / 2.0);
            return std::sin(phase) / x;
        }

        std::complex<double> nu_cyl = nu + 0.5;
        double z_2 = x / 2.0;
        double z_2_sq = z_2 * z_2;
        double tol = 1e-15;

        bool nu_is_real = (std::abs(nu_cyl.imag()) < 1e-12);
        if (nu_is_real) {
            double nu_real = nu_cyl.real();
            double term_real = 1.0 / gamma_nu_cyl_plus_1.real();
            double sum_real = term_real;
            for (int k = 1; k < 150; ++k) {
                double factor = (double)k * (nu_real + (double)k);
                term_real *= -z_2_sq / factor;
                sum_real += term_real;
                if (std::abs(term_real) < tol * std::abs(sum_real) && k > 5) break;
            }
            double z_2_pow_nu = std::pow(z_2, nu_real);
            double J_nu = z_2_pow_nu * sum_real;
            return std::sqrt(M_PI / (2.0 * x)) * J_nu;
        } else {
            std::complex<double> term = 1.0 / gamma_nu_cyl_plus_1;
            std::complex<double> sum = term;
            for (int k = 1; k < 150; ++k) {
                std::complex<double> factor = (double)k * (nu_cyl + (double)k);
                term = (term * std::conj(factor)) * (-z_2_sq / std::norm(factor));
                sum += term;
                if (std::abs(term) < tol * std::abs(sum) && k > 5) break;
            }
            double alpha = nu_cyl.real();
            double beta = nu_cyl.imag();
            double ln_z_2 = std::log(z_2);
            double arg = beta * ln_z_2;
            std::complex<double> z_2_pow_nu = std::pow(z_2, alpha) * std::complex<double>(std::cos(arg), std::sin(arg));
            std::complex<double> J_nu = z_2_pow_nu * sum;
            return std::sqrt(M_PI / (2.0 * x)) * J_nu;
        }
    }

    // Adapted from reference sph.C
    std::complex<double> SphericalHarmonicY(int l, int m, double theta, double phi) {
        double x = std::cos(theta);
        int abs_m = std::abs(m);
        if (abs_m > l) return 0.0;
        
        // Compute Associated Legendre Polynomial P_l^m(x)
        double P_lm = 0.0;
        
        // 1. P_m^m
        double sin_theta = std::sin(theta);
        double cur = 1.0;
        
        if (abs_m == 1) {
            cur = -sin_theta;
        } else if (abs_m == 2) {
            cur = 3.0 * sin_theta * sin_theta;
        } else if (abs_m == 3) {
            cur = -15.0 * sin_theta * sin_theta * sin_theta;
        } else if (abs_m > 3) {
            for (int i = 1; i <= abs_m; i++) cur *= (2*i - 1);
            if (abs_m % 2 != 0) cur = -cur;
            cur *= std::pow(sin_theta, abs_m);
        }
        
        if (l == abs_m) {
            P_lm = cur;
        } else {
            // 2. P_{m+1}^m
            double P_m_m = cur;
            double P_mp1_m = x * (2 * abs_m + 1) * P_m_m;
            if (l == abs_m + 1) {
                P_lm = P_mp1_m;
            } else {
                // 3. P_l^m recursion
                double P_prev = P_mp1_m;
                double P_prev2 = P_m_m;
                double P_curr = 0.0;
                for (int ll = abs_m + 2; ll <= l; ll++) {
                    P_curr = ((2.0 * ll - 1.0) * x * P_prev - (ll + abs_m - 1.0) * P_prev2) / (ll - abs_m);
                    P_prev2 = P_prev;
                    P_prev = P_curr;
                }
                P_lm = P_curr;
            }
        }
        
        // Normalization
        static const double norm_table[4][4] = {
            {0.28209479177387814, 0, 0, 0},                                      // l=0
            {0.4886025119029199, 0.34549414940282035, 0, 0},                     // l=1
            {0.6307831305050401, 0.25751247738580665, 0.12875623869290333, 0},    // l=2
            {0.7463526651877685, 0.2230158869152206, 0.07052341258661601, 0.028790757088921864} // l=3
        };
        
        double norm;
        if (l <= 3 && abs_m <= l) {
            norm = norm_table[l][abs_m];
        } else {
            double num = 1.0;
            double den = 1.0;
            for (int k = 1; k <= (l - abs_m); k++) num *= k;
            for (int k = 1; k <= (l + abs_m); k++) den *= k;
            norm = std::sqrt( ((2.0*l + 1.0) / (4.0 * M_PI)) * (num / den) );
        }
        
        double res = norm * P_lm;
        
        double m_phi = m * phi;
        std::complex<double> phase(std::cos(m_phi), std::sin(m_phi));
        
        // Handle negative m: Y_{l,-m} = (-1)^m Y_{l,m}^*
        if (m < 0) {
            if (abs_m % 2 != 0) res = -res;
        }

        return res * phase;
    }

}
