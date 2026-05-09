#include "beta.h"
#include "cross_section.h"
#include "continuum.h"
#include "rotation.h"
#include "tools.h"
#include "clebsch_gordan.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "math_special.h"
#include <cmath>
#include <complex>
#include <iostream>

std::complex<double> BetaCalculator::ComputeNumericalMatrixElement(
    const std::vector<double>& dyson_vals,
    const UniformGrid& grid,
    const double* k_vec,
    const double* pol_vec
) {
    std::complex<double> integral(0.0, 0.0);
    double dV = grid.dx * grid.dy * grid.dz;
    
    double k_mag = std::sqrt(k_vec[0]*k_vec[0] + k_vec[1]*k_vec[1] + k_vec[2]*k_vec[2]);
    int l_max = 10 + int(k_mag * 15.0); 
    
    for (int ix = 0; ix < grid.nx; ++ix) {
        double x = grid.xmin + ix * grid.dx;
        for (int iy = 0; iy < grid.ny; ++iy) {
            double y = grid.ymin + iy * grid.dy;
            for (int iz = 0; iz < grid.nz; ++iz) {
                double z = grid.zmin + iz * grid.dz;
                
                int idx = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;
                double dyson_val = dyson_vals[idx];
                if (std::abs(dyson_val) < 1e-12) continue; 
                
                double r_dot_eps = x * pol_vec[0] + y * pol_vec[1] + z * pol_vec[2];
                double r_vec[3] = {x, y, z};
                std::complex<double> psi_val = Continuum::EvaluatePlaneWaveExpansion(k_vec, r_vec, l_max);
                
                integral += std::conj(psi_val) * r_dot_eps * dyson_val;
            }
        }
    }
    
    return integral * dV;
}

std::vector<BetaResult> BetaCalculator::CalculateBeta(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const AngleGrid& angle_grid,
    const std::vector<double>& photoelectron_energies_ev 
) {
    std::vector<BetaResult> results;
    const double HARTREE_EV = 27.211386;
    
    std::vector<double> phi_L_vals(grid.nx * grid.ny * grid.nz);
    std::vector<double> phi_R_vals(grid.nx * grid.ny * grid.nz);
    #pragma omp parallel for collapse(3)
    for (int ix = 0; ix < grid.nx; ++ix) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int iz = 0; iz < grid.nz; ++iz) {
                int idx = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;
                double x = grid.xmin + ix * grid.dx;
                double y = grid.ymin + iy * grid.dy;
                double z = grid.zmin + iz * grid.dz;
                phi_L_vals[idx] = dyson_L.evaluate(x, y, z);
                phi_R_vals[idx] = dyson_R.evaluate(x, y, z);
            }
        }
    }
    
    double pol_lab[3] = {0.0, 0.0, 1.0};      
    double k_par_lab[3] = {0.0, 0.0, 1.0};    
    double k_perp1_lab[3] = {1.0, 0.0, 0.0};  
    double k_perp2_lab[3] = {0.0, 1.0, 0.0};  
    
    for (double E_eV : photoelectron_energies_ev) {
        double E_au = E_eV / HARTREE_EV;
        double k_mag = std::sqrt(2.0 * E_au);
        
        double sum_sigma_par = 0.0;
        double sum_sigma_perp = 0.0;
        
        #pragma omp parallel for reduction(+:sum_sigma_par, sum_sigma_perp)
        for (int i = 0; i < int(angle_grid.points.size()); ++i) {
            const auto& orient = angle_grid.points[i];
            RotationMatrix R;
            // Use the same symmetric Euler convention as point-dipole numeric averaging
            R.SetFromEuler(orient.alpha, orient.beta, orient.alpha);
            RotationMatrix RT = R.Transpose();
            
            double pol_mol[3] = {pol_lab[0], pol_lab[1], pol_lab[2]};
            RT.Apply(pol_mol[0], pol_mol[1], pol_mol[2]);
            
            auto rotate_k = [&](const double* k_lab_vec, double k_mag) {
                double k_mol[3] = {k_lab_vec[0]*k_mag, k_lab_vec[1]*k_mag, k_lab_vec[2]*k_mag};
                RT.Apply(k_mol[0], k_mol[1], k_mol[2]);
                return std::vector<double>{k_mol[0], k_mol[1], k_mol[2]};
            };

            auto k_par_mol = rotate_k(k_par_lab, k_mag);
            auto k_perp1_mol = rotate_k(k_perp1_lab, k_mag);
            auto k_perp2_mol = rotate_k(k_perp2_lab, k_mag);
            
            std::complex<double> M_par_L = ComputeNumericalMatrixElement(phi_L_vals, grid, k_par_mol.data(), pol_mol);
            std::complex<double> M_par_R = ComputeNumericalMatrixElement(phi_R_vals, grid, k_par_mol.data(), pol_mol);
            
            std::complex<double> A_par_L = M_par_L;
            std::complex<double> A_par_R = std::conj(M_par_R); 
            double sigma_par_orient = std::real(A_par_L * A_par_R);
            
            std::complex<double> M_perp1_L = ComputeNumericalMatrixElement(phi_L_vals, grid, k_perp1_mol.data(), pol_mol);
            std::complex<double> M_perp1_R = ComputeNumericalMatrixElement(phi_R_vals, grid, k_perp1_mol.data(), pol_mol);
            double sigma_perp1_orient = std::real(M_perp1_L * std::conj(M_perp1_R));
            
            std::complex<double> M_perp2_L = ComputeNumericalMatrixElement(phi_L_vals, grid, k_perp2_mol.data(), pol_mol);
            std::complex<double> M_perp2_R = ComputeNumericalMatrixElement(phi_R_vals, grid, k_perp2_mol.data(), pol_mol);
            double sigma_perp2_orient = std::real(M_perp2_L * std::conj(M_perp2_R));
            
            double sigma_perp_orient = 0.5 * (sigma_perp1_orient + sigma_perp2_orient);
            
            sum_sigma_par += sigma_par_orient * orient.weight;
            sum_sigma_perp += sigma_perp_orient * orient.weight;
        }
        
        double norms = dyson_L.qchem_norm * dyson_R.qchem_norm;
        double sigma_par = sum_sigma_par * norms;
        double sigma_perp = sum_sigma_perp * norms;
        
        double beta = 2.0 * (sigma_par - sigma_perp) / (sigma_par + 2.0 * sigma_perp);
        
        results.push_back({E_eV, sigma_par, sigma_perp, beta});
    }
    
    return results;
}

// Helper to compute Spherical Matrix Elements C_{klm, mu}
// mu = -1, 0, 1 (Spherical Dipole Component)
// result[l*(l+1) + m][mu+1]
std::vector<std::vector<std::complex<double>>> ComputeSphericalMatrixElements(
    const std::vector<double>& dyson_vals,
    const UniformGrid& grid,
    double k,
    int l_max
) {
    int num_lm = (l_max + 1) * (l_max + 1);
    std::vector<std::vector<std::complex<double>>> moments(num_lm, std::vector<std::complex<double>>(3, {0.0, 0.0}));
    
    double dV = grid.dx * grid.dy * grid.dz;
    
    #pragma omp parallel
    {
        auto local_moments = moments;
        
        #pragma omp for
        for (int ix = 0; ix < grid.nx; ++ix) {
            double x = grid.xmin + ix * grid.dx;
            for (int iy = 0; iy < grid.ny; ++iy) {
                double y = grid.ymin + iy * grid.dy;
                for (int iz = 0; iz < grid.nz; ++iz) {
                    int idx_grid = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;
                    double z = grid.zmin + iz * grid.dz;
                    
                    double dyson_val = dyson_vals[idx_grid];
                    if (std::abs(dyson_val) < 1e-12) continue;
                    
                    double r = std::sqrt(x*x + y*y + z*z);
                    if(r < 1e-10) continue;
                    
                    double theta = std::acos(z/r);
                    double phi = std::atan2(y, x);
                    
                    std::complex<double> Y1_m1 = MathSpecial::SphericalHarmonicY(1, -1, theta, phi);
                    std::complex<double> Y1_0  = MathSpecial::SphericalHarmonicY(1, 0,  theta, phi);
                    std::complex<double> Y1_1  = MathSpecial::SphericalHarmonicY(1, 1,  theta, phi);
                    
                    std::complex<double> dip_m1 = r * std::conj(Y1_m1); 
                    std::complex<double> dip_0  = r * std::conj(Y1_0);
                    std::complex<double> dip_1  = r * std::conj(Y1_1);

                    int idx = 0;
                    for(int l=0; l<=l_max; ++l) {
                        double jl = MathSpecial::SphericalBesselJ(l, k*r);
                        
                        std::complex<double> i_pow_l = (l % 4 == 0) ? 1.0 :
                                                       (l % 4 == 1) ? std::complex<double>(0, 1) :
                                                       (l % 4 == 2) ? -1.0 : std::complex<double>(0, -1);
                                                       
                        double radial_part = jl * dyson_val * dV; 
                        
                        for(int m=-l; m<=l; ++m) {
                            std::complex<double> Ylm = MathSpecial::SphericalHarmonicY(l, m, theta, phi);
                            std::complex<double> Ylm_conj = std::conj(Ylm);
                            
                            std::complex<double> common = radial_part * i_pow_l * Ylm_conj;
                            
                            local_moments[idx][0] += common * dip_m1;
                            local_moments[idx][1] += common * dip_0;
                            local_moments[idx][2] += common * dip_1;
                            
                            idx++;
                        }
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            for(size_t i=0; i<moments.size(); ++i) {
                moments[i][0] += local_moments[i][0];
                moments[i][1] += local_moments[i][1];
                moments[i][2] += local_moments[i][2];
            }
        }
    }
    
    return moments;
}
// Helper to compute Beta from Matrix Elements C_{lm, mu}
// Shared by PWE and Point Dipole
std::vector<BetaResult> ComputeBetaFromMatrixElements(
    const std::vector<std::vector<std::vector<std::complex<double>>>>& matrix_elements_L, // [energy][idx][mu]
    const std::vector<std::vector<std::vector<std::complex<double>>>>& matrix_elements_R,
    const std::vector<double>& energies_ev,
    double qchem_norm_sq,
    int l_max
) {
    std::vector<BetaResult> results;
    ClebschGordan cg; 
    
    // Dipole normalization factor
    double dipole_norm = std::sqrt(4.0 * M_PI / 3.0); 

    for(size_t ie=0; ie<energies_ev.size(); ++ie) {
        double E_eV = energies_ev[ie];
        const auto& C_L = matrix_elements_L[ie];
        const auto& C_R = matrix_elements_R[ie];
        
        double sigma_par = 0.0;
        double sigma_perp = 0.0;
        
        for (int l=0; l<=l_max; ++l) {
            for (int m1=-l; m1<=l; m1++) {
                
                std::complex<double> Y_par = MathSpecial::SphericalHarmonicY(l, m1, 0.0, 0.0);
                std::complex<double> Y_perp = MathSpecial::SphericalHarmonicY(l, m1, M_PI/2.0, 0.0);
                
                double term_contrib = 0.0;
                
                for (int m21=-l; m21<=l; m21++) {
                   for (int m22=-l; m22<=l; m22++) {
                      for (int v1=0; v1<3; v1++) { 
                         for (int v2=0; v2<3; v2++) {
                           
                            if ((m21+v1) == (m22+v2)) {
                               for (int ltot=(std::abs(l-1)); ltot<=(l+1); ltot++) {
                                  
                                  int idx1 = l*l + (m21+l);
                                  int idx2 = l*l + (m22+l);
                                  
                                  std::complex<double> val1 = C_L[idx1][v1] * dipole_norm;
                                  std::complex<double> val2 = C_R[idx2][v2] * dipole_norm;
                                  
                                  double tmp = std::real(val1 * std::conj(val2)); 
                                  
                                  int L_idx = ltot - std::abs(l-1);
                                  
                                  double cgc1 = cg.cgc[l][L_idx][m1+l][1]; 
                                  double cgc21 = cg.cgc[l][L_idx][m21+l][v1];
                                  double cgc22 = cg.cgc[l][L_idx][m22+l][v2];
                                  
                                  tmp *= cgc1 * cgc1 * cgc21 * cgc22;
                                  tmp /= (2.0 * ltot + 1.0);
                                  
                                  term_contrib += tmp;
                               }
                            }
                         }
                      }
                   }
                }
                
                sigma_par += term_contrib * std::norm(Y_par);
                sigma_perp += term_contrib * std::norm(Y_perp);
            }
        }
        
        // Interference Terms
        for (int l=0; l<=l_max; ++l) {
            for (int m11=-l; m11<=l; m11++) {
                
                std::complex<double> Y1_par = MathSpecial::SphericalHarmonicY(l, m11, 0.0, 0.0);
                std::complex<double> Y1_perp = MathSpecial::SphericalHarmonicY(l, m11, M_PI/2.0, 0.0);
                
                for (int l2=0; l2<=l_max; ++l2) {
                   if (l == l2) continue; 
                   
                   for (int m12=-l2; m12<=l2; m12++) {
                       if (m12 != m11) continue; 
                       
                       std::complex<double> Y2_par = std::conj(MathSpecial::SphericalHarmonicY(l2, m12, 0.0, 0.0));
                       std::complex<double> Y2_perp = std::conj(MathSpecial::SphericalHarmonicY(l2, m12, M_PI/2.0, 0.0));
                       
                       double term_contrib = 0.0;
                       
                       for (int m21=-l; m21<=l; m21++) {
                          for (int m22=-l2; m22<=l2; m22++) {
                             for (int v1=0; v1<3; v1++) {
                                for (int v2=0; v2<3; v2++) {
                                   
                                   if ((m21+v1-1) == (m22+v2-1)) { 
                                      
                                      int idx1 = l*l + (m21+l);
                                      int idx2 = l2*l2 + (m22+l2);
                                      
                                      std::complex<double> val1 = C_L[idx1][v1] * dipole_norm;
                                      std::complex<double> val2 = C_R[idx2][v2] * dipole_norm;
                                      
                                      double tmp = std::real(val1 * std::conj(val2));
                                      
                                      for (int ltot=(std::abs(l-1)); ltot<=(l+1); ltot++) {
                                         for (int ltot2=(std::abs(l2-1)); ltot2<=(l2+1); ltot2++) {
                                            if (ltot == ltot2) {
                                               
                                               int L_idx1 = ltot - std::abs(l-1);
                                               int L_idx2 = ltot2 - std::abs(l2-1);
                                               
                                               double cgc11 = cg.cgc[l][L_idx1][m11+l][1];
                                               double cgc12 = cg.cgc[l2][L_idx2][m12+l2][1];
                                               double cgc21 = cg.cgc[l][L_idx1][m21+l][v1];
                                               double cgc22 = cg.cgc[l2][L_idx2][m22+l2][v2];
                                               
                                               double term = tmp * cgc11 * cgc12 * cgc21 * cgc22;
                                               term /= (2.0 * ltot + 1.0);
                                               
                                               term_contrib += term;
                                            }
                                         }
                                      }
                                   }
                                }
                             }
                          }
                       }
                       sigma_par += term_contrib * std::real(Y1_par * Y2_par); 
                       sigma_perp += term_contrib * std::real(Y1_perp * Y2_perp);
                   }
                }
            }
        }
        
        double prefactor = 3.0 / (4.0 * M_PI);
        sigma_par *= prefactor;
        sigma_perp *= prefactor;
        
        sigma_par *= qchem_norm_sq;
        sigma_perp *= qchem_norm_sq;
        
        double denom = sigma_par + 2.0 * sigma_perp;
        double beta = 0.0;
        if (std::abs(denom) > 1e-18) {
            beta = 2.0 * (sigma_par - sigma_perp) / denom;
        }
        
        results.push_back({E_eV, sigma_par, sigma_perp, beta});
    }
    return results;
}

std::vector<BetaResult> BetaCalculator::CalculateBetaAnalytic(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const std::vector<double>& photoelectron_energies_ev,
    int l_max
) {
    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_L;
    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_R;
    
    const double HARTREE_EV = 27.211386;
    
    std::vector<double> phi_L_vals(grid.nx * grid.ny * grid.nz);
    std::vector<double> phi_R_vals(grid.nx * grid.ny * grid.nz);
    #pragma omp parallel for collapse(3)
    for (int ix = 0; ix < grid.nx; ++ix) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int iz = 0; iz < grid.nz; ++iz) {
                int idx = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;
                double x = grid.xmin + ix * grid.dx;
                double y = grid.ymin + iy * grid.dy;
                double z = grid.zmin + iz * grid.dz;
                phi_L_vals[idx] = dyson_L.evaluate(x, y, z);
                phi_R_vals[idx] = dyson_R.evaluate(x, y, z);
            }
        }
    }
    
    for (double E_eV : photoelectron_energies_ev) {
        double E_au = E_eV / HARTREE_EV;
        double k = std::sqrt(2.0 * E_au);
        
        // For k=0, return zeros
        if (k < 1e-6) {
             int num_lm = (l_max+1)*(l_max+1);
             matrices_L.push_back(std::vector<std::vector<std::complex<double>>>(num_lm, std::vector<std::complex<double>>(3, 0.0)));
             matrices_R.push_back(std::vector<std::vector<std::complex<double>>>(num_lm, std::vector<std::complex<double>>(3, 0.0)));
             continue;
        }

        matrices_L.push_back(ComputeSphericalMatrixElements(phi_L_vals, grid, k, l_max));
        matrices_R.push_back(ComputeSphericalMatrixElements(phi_R_vals, grid, k, l_max));
    }
    
    double norm_sq = dyson_L.qchem_norm * dyson_R.qchem_norm;
    return ComputeBetaFromMatrixElements(matrices_L, matrices_R, photoelectron_energies_ev, norm_sq, l_max);
}

// Point Dipole Matrix Elements
#include "point_dipole.h"

// Returns C_{l_in, m, mu}
std::vector<std::vector<std::complex<double>>> ComputePointDipoleMatrixElements(
    const std::vector<double>& dyson_vals,
    const UniformGrid& grid,
    double k,
    double dipole_strength,
    int l_max
) {
    int num_lm = (l_max + 1) * (l_max + 1);
    std::vector<std::vector<std::complex<double>>> moments(num_lm, std::vector<std::complex<double>>(3, {0.0, 0.0}));
    
    PointDipole pd(dipole_strength);
    double dV = grid.dx * grid.dy * grid.dz;
    
    // We iterate over possible incident channels (l_in, m)
    // For Point Dipole, l is coupled, but m is conserved. 
    // We compute the expansion: D_mu(k) = Sum_{l_in, m} Y_{l_in, m}(k) * T_{l_in, m, mu}
    // T_{l_in, m, mu} = < phi | r_mu | Psi_{k, l_in, m} >
    // Psi_{k, l_in, m} = 4 pi * Sum_N [ c_{l_in}^N * i^{L_{eff}} * (u_N/r) * Omega_N(r) ]
    
    // Optimized Strategy:
    // 1. Precompute Overlaps: O_{m, N, mu} = < phi | r_mu | (u_N/r) Omega_N >
    // 2. Assemble T_{l_in, m, mu} = Sum_N (4pi * c_{l_in}^N * i^{L_{eff}}) * O_{m, N, mu}
    
    // Storage for O_{m, N, mu}. 
    // Map key: m (-lmax to lmax). Value: vector of N moments (each N is vec<complex>[3])
    // The number of modes N depends on m.
    std::vector<std::vector<std::vector<std::complex<double>>>> overlaps(2*l_max + 1);
    
    // Initialize overlaps
    for (int lam = -l_max; lam <= l_max; ++lam) {
        Eigensystem sys = pd.GetEigensystem(lam, l_max);
        int n_modes = sys.l_vals.size();
        overlaps[lam + l_max].resize(n_modes, std::vector<std::complex<double>>(3, {0.0, 0.0}));


    }
    
    // Grid Loop
    #pragma omp parallel
    {
        auto local_overlaps = overlaps;
        // Zero out local
        for(auto& v1 : local_overlaps) for(auto& v2 : v1) for(auto& val : v2) val = {0.0, 0.0};
        
        // Thread-local reusable buffer for Y_vals (max size 2*l_max + 1)
        std::vector<std::complex<double>> Y_vals;
        Y_vals.reserve(2 * l_max + 1);

        #pragma omp for
        for (int ix = 0; ix < grid.nx; ++ix) {
           for (int iy = 0; iy < grid.ny; ++iy) {
              for (int iz = 0; iz < grid.nz; ++iz) {
                 double x = grid.xmin + ix * grid.dx;
                 double y = grid.ymin + iy * grid.dy;
                 double z = grid.zmin + iz * grid.dz;
                 int idx_grid = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;
                 
                 double dyson_val = dyson_vals[idx_grid];
                 if (std::abs(dyson_val) < 1e-12) continue;
                 
                 double r = std::sqrt(x*x + y*y + z*z);
                 if (r < 1e-10) continue;
                 
                 double theta = std::acos(z/r);
                 double phi = std::atan2(y, x);
                 
                 // Precompute dipoles Y1(r)*
                 std::complex<double> Y1_m1 = MathSpecial::SphericalHarmonicY(1, -1, theta, phi);
                 std::complex<double> Y1_0  = MathSpecial::SphericalHarmonicY(1, 0,  theta, phi);
                 std::complex<double> Y1_1  = MathSpecial::SphericalHarmonicY(1, 1,  theta, phi);
                 std::complex<double> dip[3];
                 dip[0] = r * std::conj(Y1_m1);
                 dip[1] = r * std::conj(Y1_0);
                 dip[2] = r * std::conj(Y1_1);
                 
                 for (int lam = -l_max; lam <= l_max; ++lam) {
                      Eigensystem sys = pd.GetEigensystem(lam, l_max);
                      int n_modes = sys.l_vals.size();
                      
                      // Precompute Y_lm(r) for all l in this block
                      Y_vals.resize(n_modes);
                      for(int i=0; i<n_modes; ++i) {
                          // Match PWE: Use Conjugate of Y_lm(r)
                          Y_vals[i] = std::conj(MathSpecial::SphericalHarmonicY(sys.l_vals[i], lam, theta, phi));
                      }
                      
                      for(int N=0; N<n_modes; ++N) {
                          double eigval = sys.eigvals[N];
                          if (eigval < -0.25) continue; 
                          double L_eff = 0.5 * (-1.0 + std::sqrt(1.0 + 4.0 * eigval));
                          
                          // Radial Part
                          double bessel = MathSpecial::CylBesselJ(L_eff + 0.5, k*r);
                          double radial = std::sqrt(M_PI / (2.0 * k * r)) * bessel; 
                          
                          // Angular Part Omega_N(r)*
                          std::complex<double> omega_r = 0.0;
                          for(int i=0; i<n_modes; ++i) {
                              omega_r += sys.eigvecs[i][N] * Y_vals[i];
                          }
                          
                          // Overlap fragment
                          std::complex<double> fragment = radial * omega_r * dyson_val * dV;
                          
                          // Accumulate
                          for(int mu=0; mu<3; ++mu) {
                              local_overlaps[lam+l_max][N][mu] += fragment * dip[mu];
                          }
                      }
                 }
              }
           }
        }
        
        #pragma omp critical
        {
            for(size_t i=0; i<overlaps.size(); ++i) {
                for(size_t j=0; j<overlaps[i].size(); ++j) {
                    for(int mu=0; mu<3; ++mu) {
                        overlaps[i][j][mu] += local_overlaps[i][j][mu];
                    }
                }
            }
        }
    }
    
    // Assemble Final Matrix Elements T_{l_in, m, mu}
    for (int l_in = 0; l_in <= l_max; ++l_in) {
        for (int m = -l_in; m <= l_in; ++m) {
             // Index in flat vector
             int idx = l_in * l_in + (l_in + m);
             
             // Get Eigensystem for this m
             Eigensystem sys = pd.GetEigensystem(m, l_max);
             int n_modes = sys.l_vals.size();
             
             // Find index of l_in in sys.l_vals
             // sys is built from |lam| to l_max.
             int l_idx_in_sys = -1;
             for(int k=0; k<n_modes; ++k) if(sys.l_vals[k] == l_in) { l_idx_in_sys = k; break; }
             
             if (l_idx_in_sys == -1) continue;
             
             for(int N=0; N<n_modes; ++N) {
                  double eigval = sys.eigvals[N];
                  if(eigval < -0.25) continue;
                  
                  double L_eff = 0.5 * (-1.0 + std::sqrt(1.0 + 4.0 * eigval));
                  
                  // i^L_eff 
                  std::complex<double> phase = std::exp(std::complex<double>(0.0, M_PI * 0.5 * L_eff));
                  
                  // Coefficient: sys.eigvecs[l_idx_in_sys][N]
                  double c_lin = sys.eigvecs[l_idx_in_sys][N];
                  
                  double prefactor = 1.0; 
                  
                  std::complex<double> weight = prefactor * c_lin * phase;
                  
                  for(int mu=0; mu<3; ++mu) {
                       moments[idx][mu] += weight * overlaps[m + l_max][N][mu];
                  }
             }
        }
    } 

    return moments;
}


std::vector<BetaResult> BetaCalculator::CalculateBetaPointDipole(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const std::vector<double>& photoelectron_energies_ev,
    double dipole_strength,
    int l_max
) {
    const double HARTREE_EV = 27.211386;
    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_L;
    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_R;
    
    std::vector<double> phi_L_vals(grid.nx * grid.ny * grid.nz);
    std::vector<double> phi_R_vals(grid.nx * grid.ny * grid.nz);
    #pragma omp parallel for collapse(3)
    for (int ix = 0; ix < grid.nx; ++ix) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int iz = 0; iz < grid.nz; ++iz) {
                int idx = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;
                double x = grid.xmin + ix * grid.dx;
                double y = grid.ymin + iy * grid.dy;
                double z = grid.zmin + iz * grid.dz;
                phi_L_vals[idx] = dyson_L.evaluate(x, y, z);
                phi_R_vals[idx] = dyson_R.evaluate(x, y, z);
            }
        }
    }
    
    for(double E_eV : photoelectron_energies_ev) {
         double E_au = E_eV / HARTREE_EV;
         double k = std::sqrt(2.0 * E_au);
         
         if (k < 1e-6) {
             int num_lm = (l_max+1)*(l_max+1);
             matrices_L.push_back(std::vector<std::vector<std::complex<double>>>(num_lm, std::vector<std::complex<double>>(3, 0.0)));
             matrices_R.push_back(std::vector<std::vector<std::complex<double>>>(num_lm, std::vector<std::complex<double>>(3, 0.0)));
             continue;
         }
         
         matrices_L.push_back(ComputePointDipoleMatrixElements(phi_L_vals, grid, k, dipole_strength, l_max));
         matrices_R.push_back(ComputePointDipoleMatrixElements(phi_R_vals, grid, k, dipole_strength, l_max));
    }
    
    double norm_sq = dyson_L.qchem_norm * dyson_R.qchem_norm;
    return ComputeBetaFromMatrixElements(matrices_L, matrices_R, photoelectron_energies_ev, norm_sq, l_max);
}

// Helper for Numeric Averaging (Generic Amplitude Evaluator)
std::vector<BetaResult> BetaCalculator::CalculateBetaPWENumeric(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const std::vector<double>& photoelectron_energies_ev,
    const AngleGrid& angle_grid,
    int l_max
) {
    std::vector<BetaResult> results;
    const double HARTREE_EV = 27.211386;
    
    // Precompute Body-Frame Matrix Elements for all energies
    // Reuse ComputeSphericalMatrixElements
    std::vector<double> phi_L_vals(grid.nx * grid.ny * grid.nz);
    std::vector<double> phi_R_vals(grid.nx * grid.ny * grid.nz);
    #pragma omp parallel for collapse(3)
    for (int ix = 0; ix < grid.nx; ++ix) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int iz = 0; iz < grid.nz; ++iz) {
                int idx = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;
                double x = grid.xmin + ix * grid.dx;
                double y = grid.ymin + iy * grid.dy;
                double z = grid.zmin + iz * grid.dz;
                phi_L_vals[idx] = dyson_L.evaluate(x, y, z);
                phi_R_vals[idx] = dyson_R.evaluate(x, y, z);
            }
        }
    }
    
    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_L;
    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_R;
    
    for (double E_eV : photoelectron_energies_ev) {
        double E_au = E_eV / HARTREE_EV;
        double k = std::sqrt(2.0 * E_au);
        if (k < 1e-6) k = 1e-6; // Avoid zero div
        matrices_L.push_back(ComputeSphericalMatrixElements(phi_L_vals, grid, k, l_max));
        matrices_R.push_back(ComputeSphericalMatrixElements(phi_R_vals, grid, k, l_max));
    }
    
    // Lab Frame Vectors
    double pol_lab[3] = {0.0, 0.0, 1.0};      
    double k_par_lab[3] = {0.0, 0.0, 1.0};    
    double k_perp1_lab[3] = {1.0, 0.0, 0.0};  
    double k_perp2_lab[3] = {0.0, 1.0, 0.0};
    
    for(size_t ie=0; ie<photoelectron_energies_ev.size(); ++ie) {
        double E_eV = photoelectron_energies_ev[ie];
        const auto& C_L = matrices_L[ie]; // [lm_idx][mu] (mu=0->-1, 1->0, 2->1)
        const auto& C_R = matrices_R[ie];
        
        double sum_sigma_par = 0.0;
        double sum_sigma_perp = 0.0;
        
        #pragma omp parallel for reduction(+:sum_sigma_par, sum_sigma_perp)
        for(int i=0; i<int(angle_grid.points.size()); ++i) {
            const auto& orient = angle_grid.points[i];
            RotationMatrix R;
            // alpha beta alpha was necessary to get accurate results
            R.SetFromEuler(orient.alpha, orient.beta, orient.alpha);
            RotationMatrix RT = R.Transpose(); // Lab to Body
            
            // Lab Polarization in Body Frame
            double eps_body[3] = {pol_lab[0], pol_lab[1], pol_lab[2]};
            RT.Apply(eps_body[0], eps_body[1], eps_body[2]);
            
            auto ComputeAmp = [&](const double* k_lab, const std::vector<std::vector<std::complex<double>>>& C) -> std::complex<double> {
                double k_body[3] = {k_lab[0], k_lab[1], k_lab[2]};
                RT.Apply(k_body[0], k_body[1], k_body[2]);
                
                double r = std::sqrt(k_body[0]*k_body[0] + k_body[1]*k_body[1] + k_body[2]*k_body[2]);
                if (r < 1e-12) return 0.0;
                double theta = std::acos(k_body[2]/r);
                double phi = std::atan2(k_body[1], k_body[0]);
                
                std::complex<double> total_amp = 0.0;
                
                int idx = 0;
                for(int l=0; l<=l_max; ++l) {
                    for(int m=-l; m<=l; ++m) {
                        std::complex<double> Ylm = MathSpecial::SphericalHarmonicY(l, m, theta, phi);
                        
                        // C coefficients are for spherical components (-1, 0, 1)
                        // eps_body is Cartesian (x, y, z)
                        // Convert eps_body to Spherical:
                        // e_-1 = (x - iy)/sqrt(2)
                        // e_0 = z
                        // e_1 = -(x + iy)/sqrt(2)
                        
                        std::complex<double> e_m1 = (eps_body[0] - std::complex<double>(0,1)*eps_body[1]) / std::sqrt(2.0);
                        std::complex<double> e_0  = eps_body[2];
                        std::complex<double> e_1  = -(eps_body[0] + std::complex<double>(0,1)*eps_body[1]) / std::sqrt(2.0);
                        
                        // C structure from SphericalMatrixElements: [idx][0]=m1, [1]=0, [2]=1
                        std::complex<double> term = C[idx][0] * e_m1 + C[idx][1] * e_0 + C[idx][2] * e_1;
                        
                        total_amp += term * Ylm; 
                        
                        idx++;
                    }
                }
                return total_amp;
            };
            
            std::complex<double> A_par_L = ComputeAmp(k_par_lab, C_L);
            std::complex<double> A_par_R = ComputeAmp(k_par_lab, C_R);
            double sig_par = std::real(A_par_L * std::conj(A_par_R));
            
            std::complex<double> A_perp1_L = ComputeAmp(k_perp1_lab, C_L);
            std::complex<double> A_perp1_R = ComputeAmp(k_perp1_lab, C_R);
            
            std::complex<double> A_perp2_L = ComputeAmp(k_perp2_lab, C_L);
            std::complex<double> A_perp2_R = ComputeAmp(k_perp2_lab, C_R);
            
            double sig_perp = 0.5 * (std::real(A_perp1_L * std::conj(A_perp1_R)) + std::real(A_perp2_L * std::conj(A_perp2_R)));
            
            sum_sigma_par += sig_par * orient.weight;
            sum_sigma_perp += sig_perp * orient.weight;
        }
        
        double norms = dyson_L.qchem_norm * dyson_R.qchem_norm;
        double sigma_par = sum_sigma_par * norms;
        double sigma_perp = sum_sigma_perp * norms;
        
        double denom = sigma_par + 2.0 * sigma_perp;
        double beta = 0.0;
        if (std::abs(denom) > 1e-15) {
            beta = 2.0 * (sigma_par - sigma_perp) / denom;
        }
        
        results.push_back({E_eV, sigma_par, sigma_perp, beta});
    }
    return results;
}

// Per-energy point dipole overlap data indexed by (m, N) instead of (l, m)
struct PointDipoleOverlapData {
    // overlaps[m_idx][N][mu], where m_idx = m + l_max
    std::vector<std::vector<std::vector<std::complex<double>>>> overlaps;
    // Eigensystems per m: eigsys[m_idx]
    std::vector<Eigensystem> eigsys;
};

// Compute raw overlaps O_{m,N,mu} = < phi | r_mu | R_N(kr) Omega_N*(r) >
// These are the spatial integrals per mode (m, N), WITHOUT the i^L or eigenvector assembly.
static PointDipoleOverlapData ComputePointDipoleOverlaps(
    const std::vector<double>& dyson_vals,
    const UniformGrid& grid,
    double k,
    double dipole_strength,
    int l_max
) {
    PointDipoleOverlapData data;
    PointDipole pd(dipole_strength);
    double dV = grid.dx * grid.dy * grid.dz;

    data.eigsys.resize(2*l_max + 1);
    data.overlaps.resize(2*l_max + 1);

    for (int lam = -l_max; lam <= l_max; ++lam) {
        data.eigsys[lam + l_max] = pd.GetEigensystem(lam, l_max);
        int n_modes = data.eigsys[lam + l_max].l_vals.size();
        data.overlaps[lam + l_max].resize(n_modes, std::vector<std::complex<double>>(3, {0.0, 0.0}));
    }

    // Grid Loop — same integration as ComputePointDipoleMatrixElements
    #pragma omp parallel
    {
        auto local_overlaps = data.overlaps;
        for(auto& v1 : local_overlaps) for(auto& v2 : v1) for(auto& val : v2) val = {0.0, 0.0};

        // Thread-local reusable buffer for Y_vals
        std::vector<std::complex<double>> Y_vals;
        Y_vals.reserve(2 * l_max + 1);

        #pragma omp for
        for (int ix = 0; ix < grid.nx; ++ix) {
           for (int iy = 0; iy < grid.ny; ++iy) {
              for (int iz = 0; iz < grid.nz; ++iz) {
                 double x = grid.xmin + ix * grid.dx;
                 double y = grid.ymin + iy * grid.dy;
                 double z = grid.zmin + iz * grid.dz;
                 int idx_grid = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;

                 double dyson_val = dyson_vals[idx_grid];
                 if (std::abs(dyson_val) < 1e-12) continue;

                 double r = std::sqrt(x*x + y*y + z*z);
                 if (r < 1e-10) continue;

                 double theta = std::acos(z/r);
                 double phi = std::atan2(y, x);

                 // Spherical dipole components: r * Y_1^mu*(r)
                 std::complex<double> Y1_m1 = MathSpecial::SphericalHarmonicY(1, -1, theta, phi);
                 std::complex<double> Y1_0  = MathSpecial::SphericalHarmonicY(1, 0,  theta, phi);
                 std::complex<double> Y1_1  = MathSpecial::SphericalHarmonicY(1, 1,  theta, phi);
                 std::complex<double> dip[3];
                 dip[0] = r * std::conj(Y1_m1);
                 dip[1] = r * std::conj(Y1_0);
                 dip[2] = r * std::conj(Y1_1);

                 for (int lam = -l_max; lam <= l_max; ++lam) {
                      const auto& sys = data.eigsys[lam + l_max];
                      int n_modes = sys.l_vals.size();

                      Y_vals.resize(n_modes);
                      for(int i=0; i<n_modes; ++i) {
                          Y_vals[i] = std::conj(MathSpecial::SphericalHarmonicY(sys.l_vals[i], lam, theta, phi));
                      }

                      for(int N=0; N<n_modes; ++N) {
                          double eigval = sys.eigvals[N];
                          if (eigval < -0.25) continue;
                          double L_eff = 0.5 * (-1.0 + std::sqrt(1.0 + 4.0 * eigval));

                          double bessel = MathSpecial::CylBesselJ(L_eff + 0.5, k*r);
                          double radial = std::sqrt(M_PI / (2.0 * k * r)) * bessel;

                          // Angular eigenfunction Omega_N*(r) = sum_l c_l^N Y_lm*(r)
                          std::complex<double> omega_r = 0.0;
                          for(int i=0; i<n_modes; ++i) {
                              omega_r += sys.eigvecs[i][N] * Y_vals[i];
                          }

                          std::complex<double> fragment = radial * omega_r * dyson_val * dV;

                          for(int mu=0; mu<3; ++mu) {
                              local_overlaps[lam+l_max][N][mu] += fragment * dip[mu];
                          }
                      }
                 }
              }
           }
        }

        #pragma omp critical
        {
            for(size_t i=0; i<data.overlaps.size(); ++i) {
                for(size_t j=0; j<data.overlaps[i].size(); ++j) {
                    for(int mu=0; mu<3; ++mu) {
                        data.overlaps[i][j][mu] += local_overlaps[i][j][mu];
                    }
                }
            }
        }
    }

    return data;
}

// Helper for Point Dipole Numeric
// Uses point dipole angular eigenfunctions Omega_N(k_hat) instead of Y_lm(k_hat)
// to reconstruct the directional continuum wavefunction.
std::vector<BetaResult> BetaCalculator::CalculateBetaPointDipoleNumeric(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const std::vector<double>& photoelectron_energies_ev,
    double dipole_strength,
    const AngleGrid& angle_grid,
    int l_max
) {
    std::vector<BetaResult> results;
    const double HARTREE_EV = 27.211386;

    // Compute per-energy overlap data indexed by (m, N)
    std::vector<PointDipoleOverlapData> data_L, data_R;
    
    std::vector<double> phi_L_vals(grid.nx * grid.ny * grid.nz);
    std::vector<double> phi_R_vals(grid.nx * grid.ny * grid.nz);
    #pragma omp parallel for collapse(3)
    for (int ix = 0; ix < grid.nx; ++ix) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int iz = 0; iz < grid.nz; ++iz) {
                int idx = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;
                double x = grid.xmin + ix * grid.dx;
                double y = grid.ymin + iy * grid.dy;
                double z = grid.zmin + iz * grid.dz;
                phi_L_vals[idx] = dyson_L.evaluate(x, y, z);
                phi_R_vals[idx] = dyson_R.evaluate(x, y, z);
            }
        }
    }

    for (double E_eV : photoelectron_energies_ev) {
        double E_au = E_eV / HARTREE_EV;
        double k = std::sqrt(2.0 * E_au);
        if (k < 1e-6) k = 1e-6;
        data_L.push_back(ComputePointDipoleOverlaps(phi_L_vals, grid, k, dipole_strength, l_max));
        data_R.push_back(ComputePointDipoleOverlaps(phi_R_vals, grid, k, dipole_strength, l_max));
    }

    double pol_lab[3] = {0.0, 0.0, 1.0};
    double k_par_lab[3] = {0.0, 0.0, 1.0};
    double k_perp1_lab[3] = {1.0, 0.0, 0.0};
    double k_perp2_lab[3] = {0.0, 1.0, 0.0};

    for(size_t ie=0; ie<photoelectron_energies_ev.size(); ++ie) {
        double E_eV = photoelectron_energies_ev[ie];
        const auto& dL = data_L[ie];
        const auto& dR = data_R[ie];

        double sum_sigma_par = 0.0;
        double sum_sigma_perp = 0.0;

        #pragma omp parallel for reduction(+:sum_sigma_par, sum_sigma_perp)
        for(int i=0; i<int(angle_grid.points.size()); ++i) {
            const auto& orient = angle_grid.points[i];
            RotationMatrix R;
            // Match ezDyson convention: grid_alpha -> rotation_gamma, rotation_alpha=0
            R.SetFromEuler(orient.alpha, orient.beta, orient.alpha);
            RotationMatrix RT = R.Transpose();

            double eps_body[3] = {pol_lab[0], pol_lab[1], pol_lab[2]};
            RT.Apply(eps_body[0], eps_body[1], eps_body[2]);

            // Spherical polarization components
            std::complex<double> eps_sph[3];
            eps_sph[0] = (eps_body[0] - std::complex<double>(0,1)*eps_body[1]) / std::sqrt(2.0); // mu=-1
            eps_sph[1] = eps_body[2]; // mu=0
            eps_sph[2] = -(eps_body[0] + std::complex<double>(0,1)*eps_body[1]) / std::sqrt(2.0); // mu=1

            // Compute amplitude using point dipole angular eigenfunctions
            // A(k_hat) = Sum_{m,N} Omega_N^(m)(k_hat) * i^L_N * O_{m,N} . eps
            auto ComputeAmp = [&](const double* k_lab, const PointDipoleOverlapData& data) -> std::complex<double> {
                double k_body[3] = {k_lab[0], k_lab[1], k_lab[2]};
                RT.Apply(k_body[0], k_body[1], k_body[2]);

                double r = std::sqrt(k_body[0]*k_body[0] + k_body[1]*k_body[1] + k_body[2]*k_body[2]);
                if (r < 1e-12) return 0.0;
                double theta = std::acos(k_body[2]/r);
                double phi = std::atan2(k_body[1], k_body[0]);

                std::complex<double> total_amp = 0.0;

                // Loop over m (conserved quantum number)
                for (int m = -l_max; m <= l_max; ++m) {
                    int m_idx = m + l_max;
                    const auto& sys = data.eigsys[m_idx];
                    int n_modes = sys.l_vals.size();
                    if (n_modes == 0) continue;

                    // Precompute Y_lm(k_hat) for all l in this m-block
                    std::vector<std::complex<double>> Y_k(n_modes);
                    for (int j = 0; j < n_modes; ++j) {
                        Y_k[j] = MathSpecial::SphericalHarmonicY(sys.l_vals[j], m, theta, phi);
                    }

                    // Loop over eigenmodes N
                    for (int N = 0; N < n_modes; ++N) {
                        double eigval = sys.eigvals[N];
                        if (eigval < -0.25) continue;
                        double L_eff = 0.5 * (-1.0 + std::sqrt(1.0 + 4.0 * eigval));

                        // Phase factor i^L_eff
                        std::complex<double> phase = std::exp(std::complex<double>(0.0, M_PI * 0.5 * L_eff));

                        // Point dipole angular eigenfunction at k_hat direction
                        // Omega_N^(m)(k_hat) = sum_l c_l^N * Y_{l,m}(k_hat)
                        std::complex<double> omega_k = 0.0;
                        for (int j = 0; j < n_modes; ++j) {
                            omega_k += sys.eigvecs[j][N] * Y_k[j];
                        }

                        // Dot overlap with polarization
                        std::complex<double> O_dot_eps =
                            data.overlaps[m_idx][N][0] * eps_sph[0] +
                            data.overlaps[m_idx][N][1] * eps_sph[1] +
                            data.overlaps[m_idx][N][2] * eps_sph[2];

                        total_amp += omega_k * phase * O_dot_eps;
                    }
                }
                return total_amp;
            };

            std::complex<double> A_par_L = ComputeAmp(k_par_lab, dL);
            std::complex<double> A_par_R = ComputeAmp(k_par_lab, dR);
            double sig_par = std::real(A_par_L * std::conj(A_par_R));

            std::complex<double> A_perp1_L = ComputeAmp(k_perp1_lab, dL);
            std::complex<double> A_perp1_R = ComputeAmp(k_perp1_lab, dR);
            std::complex<double> A_perp2_L = ComputeAmp(k_perp2_lab, dL);
            std::complex<double> A_perp2_R = ComputeAmp(k_perp2_lab, dR);
            double sig_perp = 0.5 * (std::real(A_perp1_L * std::conj(A_perp1_R)) + std::real(A_perp2_L * std::conj(A_perp2_R)));

            sum_sigma_par += sig_par * orient.weight;
            sum_sigma_perp += sig_perp * orient.weight;
        }

        double norms = dyson_L.qchem_norm * dyson_R.qchem_norm;
        double sigma_par = sum_sigma_par * norms;
        double sigma_perp = sum_sigma_perp * norms;

        double denom = sigma_par + 2.0 * sigma_perp;
        double beta = 0.0;
        if (std::abs(denom) > 1e-15) {
            beta = 2.0 * (sigma_par - sigma_perp) / denom;
        }

        results.push_back({E_eV, sigma_par, sigma_perp, beta});
    }
    return results;
}

#include "physical_dipole.h"
#include <map>

static std::vector<std::vector<std::complex<double>>> ComputePhysicalDipoleAnalyticME(
    const std::vector<double>& dyson_vals,
    const UniformGrid& grid,
    double eKE_au,
    double dipole_strength,
    double dipole_length,
    const std::vector<double>& dipole_axis,
    const std::vector<double>& dipole_center,
    int l_max
);

std::vector<BetaResult> BetaCalculator::CalculateBetaPhysicalDipole(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const std::vector<double>& photoelectron_energies_ev,
    double dipole_strength,
    double dipole_length,
    const std::vector<double>& dipole_axis,
    const std::vector<double>& dipole_center,
    const AngleGrid& angle_grid,
    int l_max
) {
    const double HARTREE_EV = 27.211386;
    std::vector<BetaResult> results;

    // Build body-frame Y_lm matrix elements with the exact same convention as
    // the analytic physical-dipole path. Numeric averaging then differs only by
    // orientation quadrature (not by angular normalization/phase conventions).
    std::vector<double> phi_L_vals(grid.nx * grid.ny * grid.nz);
    std::vector<double> phi_R_vals(grid.nx * grid.ny * grid.nz);
    #pragma omp parallel for collapse(3)
    for (int ix = 0; ix < grid.nx; ++ix) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int iz = 0; iz < grid.nz; ++iz) {
                int idx = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;
                double x = grid.xmin + ix * grid.dx;
                double y = grid.ymin + iy * grid.dy;
                double z = grid.zmin + iz * grid.dz;
                phi_L_vals[idx] = dyson_L.evaluate(x, y, z);
                phi_R_vals[idx] = dyson_R.evaluate(x, y, z);
            }
        }
    }

    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_L;
    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_R;
    for (double E_eV : photoelectron_energies_ev) {
        double eKE_au = E_eV / HARTREE_EV;
        if (eKE_au <= 0) {
            int num_lm = (l_max + 1) * (l_max + 1);
            matrices_L.push_back(std::vector<std::vector<std::complex<double>>>(
                num_lm, std::vector<std::complex<double>>(3, 0.0)));
            matrices_R.push_back(std::vector<std::vector<std::complex<double>>>(
                num_lm, std::vector<std::complex<double>>(3, 0.0)));
            continue;
        }

        matrices_L.push_back(ComputePhysicalDipoleAnalyticME(
            phi_L_vals, grid, eKE_au, dipole_strength, dipole_length,
            dipole_axis, dipole_center, l_max));
        matrices_R.push_back(ComputePhysicalDipoleAnalyticME(
            phi_R_vals, grid, eKE_au, dipole_strength, dipole_length,
            dipole_axis, dipole_center, l_max));
    }
    
    double pol_lab[3] = {0.0, 0.0, 1.0};      
    double k_par_lab[3] = {0.0, 0.0, 1.0};    
    double k_perp1_lab[3] = {1.0, 0.0, 0.0};  
    double k_perp2_lab[3] = {0.0, 1.0, 0.0}; 

    for(size_t ie=0; ie<photoelectron_energies_ev.size(); ++ie) {
        double eKE = photoelectron_energies_ev[ie];
        if(eKE <= 0) {
            results.push_back({eKE, 0.0, 0.0, 0.0});
            continue;
        }

        const auto& C_L = matrices_L[ie];
        const auto& C_R = matrices_R[ie];
        
        double sum_sigma_par = 0.0;
        double sum_sigma_perp = 0.0;
        
        // Parallelize Orientations
        #pragma omp parallel for reduction(+:sum_sigma_par, sum_sigma_perp)
        for(int i=0; i<int(angle_grid.points.size()); ++i) {
            const auto& orient = angle_grid.points[i];
            RotationMatrix R;
            // Match point-dipole numeric orientation convention
            R.SetFromEuler(orient.alpha, orient.beta, orient.alpha);
            RotationMatrix RT = R.Transpose(); // Lab to Body
            
            // Lab Polarization in Body Frame
            double eps_body[3] = {pol_lab[0], pol_lab[1], pol_lab[2]};
            RT.Apply(eps_body[0], eps_body[1], eps_body[2]);

            // Cartesian -> spherical polarization components
            std::complex<double> e_m1 = (eps_body[0] - std::complex<double>(0,1)*eps_body[1]) / std::sqrt(2.0);
            std::complex<double> e_0  = eps_body[2];
            std::complex<double> e_1  = -(eps_body[0] + std::complex<double>(0,1)*eps_body[1]) / std::sqrt(2.0);
            
            // Helper to compute amplitude from Y_lm matrix elements for a given k direction
            auto ComputeAmp = [&](const double* k_lab, const std::vector<std::vector<std::complex<double>>>& C) -> std::complex<double> {
                double k_body[3] = {k_lab[0], k_lab[1], k_lab[2]};
                RT.Apply(k_body[0], k_body[1], k_body[2]);
                
                double r = std::sqrt(k_body[0]*k_body[0] + k_body[1]*k_body[1] + k_body[2]*k_body[2]); 
                if (r < 1e-12) return 0.0;
                
                double theta_k = std::acos(k_body[2]/r);
                double phi_k = std::atan2(k_body[1], k_body[0]);
                
                std::complex<double> total_amp = 0.0;

                int idx_lm = 0;
                for (int l = 0; l <= l_max; ++l) {
                    for (int m = -l; m <= l; ++m) {
                        std::complex<double> Ylm = MathSpecial::SphericalHarmonicY(l, m, theta_k, phi_k);
                        std::complex<double> term = C[idx_lm][0] * e_m1 + C[idx_lm][1] * e_0 + C[idx_lm][2] * e_1;
                        total_amp += term * Ylm;
                        idx_lm++;
                    }
                }

                return total_amp;
            };
            
            std::complex<double> A_par_L = ComputeAmp(k_par_lab, C_L);
            std::complex<double> A_par_R = ComputeAmp(k_par_lab, C_R);
            double sig_par = std::real(A_par_L * std::conj(A_par_R));
            
            std::complex<double> A_perp1_L = ComputeAmp(k_perp1_lab, C_L);
            std::complex<double> A_perp1_R = ComputeAmp(k_perp1_lab, C_R);
            std::complex<double> A_perp2_L = ComputeAmp(k_perp2_lab, C_L);
            std::complex<double> A_perp2_R = ComputeAmp(k_perp2_lab, C_R);
            double sig_perp = 0.5 * (
                std::real(A_perp1_L * std::conj(A_perp1_R)) +
                std::real(A_perp2_L * std::conj(A_perp2_R))
            );
            
            sum_sigma_par += sig_par * orient.weight;
            sum_sigma_perp += sig_perp * orient.weight;
        }
        
        // Finalize
        double norms = dyson_L.qchem_norm * dyson_R.qchem_norm; 
        
        double sigma_par_total = sum_sigma_par * norms;
        double sigma_perp_total = sum_sigma_perp * norms;
        
        double denom = sigma_par_total + 2.0 * sigma_perp_total;
        double beta = 0.0;
        if (std::abs(denom) > 1e-15) {
            beta = 2.0 * (sigma_par_total - sigma_perp_total) / denom;
        }
        
        results.push_back({eKE, sigma_par_total, sigma_perp_total, beta});
    }
    
    return results;
}

// ============================================================
// Physical Dipole Analytic Averaging
// ============================================================
// Mirrors ComputePointDipoleMatrixElements exactly:
//   1. Solve physical dipole angular eigensystem → eigenvalues & eigenvectors
//   2. Convert eigenvectors from P_l^|m| basis to Y_lm basis: d_l = c_l * sqrt(S_ll)
//   3. Map eigenvalues to point-dipole convention: Alm = -lambda → L_eff
//   4. Compute overlaps using j_{L_eff}(kr) * Omega_N*(r) in spherical coords
//   5. Assembly: T_{l_in,m,mu} = sum_N d_{l_in}^N * i^{L_eff_N} * O_{m,N,mu}
//   6. Pass to ComputeBetaFromMatrixElements for CG averaging
//
// The only input from the physical dipole is the ANGULAR eigensystem.
// Radial functions use standard spherical Bessel j_{L_eff}(kr), same as point dipole.

// Helper: compute S_ll = 2*(l+|m|)! / ((2l+1)*(l-|m|)!) for Plm normalization
static double ComputeS_ll(int l, int abs_m) {
    // (l+|m|)!/(l-|m|)! = product_{k=l-|m|+1}^{l+|m|} k
    double fact_ratio = 1.0;
    for (int k = l - abs_m + 1; k <= l + abs_m; ++k)
        fact_ratio *= static_cast<double>(k);
    return 2.0 * fact_ratio / (2.0 * l + 1.0);
}

// Holds the physical dipole angular eigensystem converted to point-dipole conventions
struct PhysDipoleEigen {
    std::vector<double> eigvals_pd;       // Point-dipole-convention eigenvalues (Alm = -lambda)
    std::vector<std::complex<double>> L_eff; // Effective angular momentum (complex for supercritical)
    std::vector<std::vector<double>> eigvecs_ylm; // Eigenvectors converted to Y_lm basis [basis_idx][mode]
    std::vector<int> l_vals;              // l values for basis functions
};

static std::vector<std::vector<std::complex<double>>> ComputePhysicalDipoleAnalyticME(
    const std::vector<double>& dyson_vals,
    const UniformGrid& grid,
    double eKE_au,
    double dipole_strength,
    double dipole_length,
    const std::vector<double>& /*dipole_axis*/,
    const std::vector<double>& /*dipole_center*/,
    int l_max
) {
    int num_lm = (l_max + 1) * (l_max + 1);
    std::vector<std::vector<std::complex<double>>> moments(
        num_lm, std::vector<std::complex<double>>(3, {0.0, 0.0}));

    if (eKE_au <= 0) return moments;

    double k = std::sqrt(2.0 * eKE_au);
    double D_phys = 0.5 * dipole_strength;  // Physical dipole convention: D_sph = 0.5 * D_cli
    double dV = grid.dx * grid.dy * grid.dz;

    // 1. Solve physical dipole angular eigensystem for each m,
    //    and convert to point-dipole conventions.
    std::vector<PhysDipoleEigen> eigen_sys(2 * l_max + 1);

    for (int lam = -l_max; lam <= l_max; ++lam) {
        auto [eigvals, eigvecs, l_vals] = PhysicalDipoleAngular::Solve(
            lam, l_max, eKE_au, dipole_length, D_phys);

        PhysDipoleEigen& pe = eigen_sys[lam + l_max];
        pe.l_vals = l_vals;
        int n_modes = (int)eigvals.size();
        int n_basis = (int)l_vals.size();
        pe.eigvals_pd.resize(n_modes);
        pe.L_eff.resize(n_modes);
        pe.eigvecs_ylm = eigvecs; // Copy, then convert in-place

        int abs_m = std::abs(lam);

        // Convert eigenvalues: physical dipole lambda → point-dipole Alm = -lambda
        // Compute L_eff = 0.5*(-1 + sqrt(1 + 4*Alm))
        // For supercritical modes (disc < 0), L_eff is complex.
        for (int n = 0; n < n_modes; ++n) {
            pe.eigvals_pd[n] = -eigvals[n]; // Alm
            double disc = 1.0 + 4.0 * pe.eigvals_pd[n];
            if (disc >= 0) {
                pe.L_eff[n] = std::complex<double>(0.5 * (-1.0 + std::sqrt(disc)), 0.0);
            } else {
                // Supercritical: L_eff = -0.5 + i*sqrt(|disc|)/2
                pe.L_eff[n] = std::complex<double>(-0.5, 0.5 * std::sqrt(-disc));
            }
        }

        // Convert eigenvectors from P_l^|m| basis to Y_lm basis:
        // d_l = c_l * sqrt(S_ll) * gauge_factor
        //
        // Gauge fix: The physical dipole off-diagonal coupling is always negative,
        // while the point dipole coupling picks up (-1)^m from 3j symbols.
        // For odd m, the coupling signs differ, producing eigenvectors in a
        // different gauge. Apply (-1)^{l-|m|} for odd |m| to align conventions.
        for (int i = 0; i < n_basis; ++i) {
            int l = l_vals[i];
            double conv = std::sqrt(ComputeS_ll(l, abs_m));
            // Gauge: for odd |m|, flip sign of every other basis function
            if (abs_m % 2 != 0) {
                int gauge_exp = (l - abs_m);
                if (gauge_exp % 2 != 0) conv = -conv;
            }
            for (int n = 0; n < n_modes; ++n) {
                pe.eigvecs_ylm[i][n] *= conv;
            }
        }

    }

    // 2. Compute overlaps O_{m, N, mu} using j_{L_eff}(kr) * Omega_N*(r_hat)
    //    This is IDENTICAL to ComputePointDipoleMatrixElements grid loop.
    std::vector<std::vector<std::vector<std::complex<double>>>> overlaps(2 * l_max + 1);

    for (int lam = -l_max; lam <= l_max; ++lam) {
        int n_modes = (int)eigen_sys[lam + l_max].eigvals_pd.size();
        overlaps[lam + l_max].resize(n_modes, std::vector<std::complex<double>>(3, {0.0, 0.0}));
    }

    #pragma omp parallel
    {
        auto local_overlaps = overlaps;
        for (auto& v1 : local_overlaps) for (auto& v2 : v1) for (auto& val : v2) val = {0.0, 0.0};

        // Thread-local reusable buffer for Y_vals
        std::vector<std::complex<double>> Y_vals;
        Y_vals.reserve(2 * l_max + 1);

        #pragma omp for
        for (int ix = 0; ix < grid.nx; ++ix) {
            for (int iy = 0; iy < grid.ny; ++iy) {
                for (int iz = 0; iz < grid.nz; ++iz) {
                    double x = grid.xmin + ix * grid.dx;
                    double y = grid.ymin + iy * grid.dy;
                    double z = grid.zmin + iz * grid.dz;
                    int idx_grid = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;

                    double dyson_val = dyson_vals[idx_grid];
                    if (std::abs(dyson_val) < 1e-12) continue;

                    double r = std::sqrt(x * x + y * y + z * z);
                    if (r < 1e-10) continue;

                    double theta = std::acos(z / r);
                    double phi = std::atan2(y, x);

                    // Spherical dipole components r * Y_1^mu*(r_hat)
                    std::complex<double> Y1_m1 = MathSpecial::SphericalHarmonicY(1, -1, theta, phi);
                    std::complex<double> Y1_0  = MathSpecial::SphericalHarmonicY(1, 0,  theta, phi);
                    std::complex<double> Y1_1  = MathSpecial::SphericalHarmonicY(1, 1,  theta, phi);
                    std::complex<double> dip[3];
                    dip[0] = r * std::conj(Y1_m1);
                    dip[1] = r * std::conj(Y1_0);
                    dip[2] = r * std::conj(Y1_1);

                    for (int lam = -l_max; lam <= l_max; ++lam) {
                        const auto& pe = eigen_sys[lam + l_max];
                        int n_modes = (int)pe.eigvals_pd.size();

                        // Precompute Y_{l,lam}*(r_hat) for all basis functions
                        Y_vals.resize(pe.l_vals.size());
                        for (size_t i = 0; i < pe.l_vals.size(); ++i) {
                            Y_vals[i] = std::conj(MathSpecial::SphericalHarmonicY(
                                pe.l_vals[i], lam, theta, phi));
                        }

                        for (int N = 0; N < n_modes; ++N) {
                            // Radial: j_{L_eff}(kr) — complex-order for supercritical
                            std::complex<double> radial_c = MathSpecial::SphericalBesselJComplex(pe.L_eff[N], k * r);

                            // Angular: Omega_N*(r_hat) = sum_l d_l^N * Y_{l,lam}*(r_hat)
                            std::complex<double> omega_r = 0.0;
                            for (size_t i = 0; i < pe.l_vals.size(); ++i) {
                                omega_r += pe.eigvecs_ylm[i][N] * Y_vals[i];
                            }

                            // Overlap fragment (radial is now complex for supercritical)
                            std::complex<double> fragment = radial_c * omega_r * dyson_val * dV;

                            for (int mu = 0; mu < 3; ++mu) {
                                local_overlaps[lam + l_max][N][mu] += fragment * dip[mu];
                            }
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (size_t i = 0; i < overlaps.size(); ++i) {
                for (size_t j = 0; j < overlaps[i].size(); ++j) {
                    for (int mu = 0; mu < 3; ++mu) {
                        overlaps[i][j][mu] += local_overlaps[i][j][mu];
                    }
                }
            }
        }
    }

    // 3. Assembly: T_{l_in, m, mu} = sum_N d_{l_in}^N * i^{L_eff_N} * O_{m, N, mu}
    //    Identical to ComputePointDipoleMatrixElements assembly.
    for (int l_in = 0; l_in <= l_max; ++l_in) {
        for (int m = -l_in; m <= l_in; ++m) {
            int idx = l_in * l_in + (l_in + m);

            const auto& pe = eigen_sys[m + l_max];
            int n_modes = (int)pe.eigvals_pd.size();

            // Find index of l_in in pe.l_vals
            int l_idx_in = -1;
            for (size_t kk = 0; kk < pe.l_vals.size(); ++kk) {
                if (pe.l_vals[kk] == l_in) { l_idx_in = (int)kk; break; }
            }
            if (l_idx_in == -1) continue;

            for (int N = 0; N < n_modes; ++N) {
                // Phase: i^{L_eff} (complex L_eff handled naturally)
                std::complex<double> phase = std::exp(
                    std::complex<double>(0.0, 1.0) * (M_PI * 0.5 * pe.L_eff[N]));

                // Y_lm coefficient of mode N at l_in (already converted)
                double d_lin = pe.eigvecs_ylm[l_idx_in][N];

                std::complex<double> weight = d_lin * phase;

                for (int mu = 0; mu < 3; ++mu) {
                    moments[idx][mu] += weight * overlaps[m + l_max][N][mu];
                }
            }
        }
    }

    return moments;
}

std::vector<BetaResult> BetaCalculator::CalculateBetaPhysicalDipoleAnalytic(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const std::vector<double>& photoelectron_energies_ev,
    double dipole_strength,
    double dipole_length,
    const std::vector<double>& dipole_axis,
    const std::vector<double>& dipole_center,
    int l_max
) {
    const double HARTREE_EV = 27.211386;
    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_L, matrices_R;

    std::vector<double> phi_L_vals(grid.nx * grid.ny * grid.nz);
    std::vector<double> phi_R_vals(grid.nx * grid.ny * grid.nz);
    #pragma omp parallel for collapse(3)
    for (int ix = 0; ix < grid.nx; ++ix) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int iz = 0; iz < grid.nz; ++iz) {
                int idx = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;
                double x = grid.xmin + ix * grid.dx;
                double y = grid.ymin + iy * grid.dy;
                double z = grid.zmin + iz * grid.dz;
                phi_L_vals[idx] = dyson_L.evaluate(x, y, z);
                phi_R_vals[idx] = dyson_R.evaluate(x, y, z);
            }
        }
    }

    for (double E_eV : photoelectron_energies_ev) {
        double eKE_au = E_eV / HARTREE_EV;

        if (eKE_au <= 0) {
            int num_lm = (l_max + 1) * (l_max + 1);
            matrices_L.push_back(std::vector<std::vector<std::complex<double>>>(
                num_lm, std::vector<std::complex<double>>(3, 0.0)));
            matrices_R.push_back(std::vector<std::vector<std::complex<double>>>(
                num_lm, std::vector<std::complex<double>>(3, 0.0)));
            continue;
        }

        // Compute Y_lm matrix elements separately for L and R Dyson orbitals
        matrices_L.push_back(ComputePhysicalDipoleAnalyticME(
            phi_L_vals, grid, eKE_au, dipole_strength, dipole_length,
            dipole_axis, dipole_center, l_max));
        matrices_R.push_back(ComputePhysicalDipoleAnalyticME(
            phi_R_vals, grid, eKE_au, dipole_strength, dipole_length,
            dipole_axis, dipole_center, l_max));
    }

    double norm_sq = dyson_L.qchem_norm * dyson_R.qchem_norm;
    return ComputeBetaFromMatrixElements(
        matrices_L, matrices_R, photoelectron_energies_ev, norm_sq, l_max);
}
