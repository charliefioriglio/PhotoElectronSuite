#include "cross_section.h"
#include "tools.h"
#include "math_special.h"
#include "continuum.h"
#include "tools.h" // For Constants if needed
#include <cmath>
#include <complex>
#include <iostream>
// #include <omp.h>

// Constants
constexpr double HARTREE_TO_EV = 27.211386;
constexpr double EV_TO_HARTREE = 1.0 / HARTREE_TO_EV;
constexpr double C_SPEED_AU = 137.035999; 

// Plane wave expansion:
// psi_{klm} = Y_{lm}(r_hat) * j_l(kr)
// C_{klm} = i^l * integral( phi_dyson(r) * r_alpha * psi_{klm}(r) )

std::vector<double> CrossSectionCalculator::ComputeTotalCrossSection(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const std::vector<double>& photon_energies_ev,
    double ionization_energy_ev,
    int l_max
) {
    std::vector<double> results;
    results.reserve(photon_energies_ev.size());
    
    // Pre-compute Dyson values on grid to avoid re-evaluation in inner loops
    std::vector<double> phi_L_vals;
    std::vector<double> phi_R_vals;
    phi_L_vals.reserve(grid.nx * grid.ny * grid.nz);
    phi_R_vals.reserve(grid.nx * grid.ny * grid.nz);
    
    // Fill grids
    double x0 = grid.xmin; double y0 = grid.ymin; double z0 = grid.zmin;
    double step = grid.dx; // Assume uniform
    
    // Loop once to fill
    for (int ix = 0; ix < grid.nx; ++ix) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int iz = 0; iz < grid.nz; ++iz) {
                 double x = x0 + ix * step;
                 double y = y0 + iy * step;
                 double z = z0 + iz * step;
                 phi_L_vals.push_back(dyson_L.evaluate(x, y, z));
                 phi_R_vals.push_back(dyson_R.evaluate(x, y, z));
            }
        }
    }

    // Optimization: Loop Grid (Outer) -> Energy (Inner)
    // Avoids re-evaluating Dyson and Ylm for every energy.
    // Dyson(r) and Ylm(theta,phi) are Energy-independent.
    // Bessel(kr) depends on k(E).
    
    int num_energies = photon_energies_ev.size();
    
    // Store partial sums for each energy: total_sum[E]
    std::vector<double> energy_sums(num_energies, 0.0);
    
    // Pre-calculate k values
    std::vector<double> k_values;
    k_values.reserve(num_energies);
    for(double E_ev : photon_energies_ev) {
        double E_au = E_ev * EV_TO_HARTREE;
        double IE_au = ionization_energy_ev * EV_TO_HARTREE;
        double eKE = E_au - IE_au;
        if(eKE > 0) {
            k_values.push_back(std::sqrt(2.0 * eKE));
        } else {
            k_values.push_back(0.0);
        }
    }
    
    // Iterate Grid points once
    int nx = grid.nx; int ny = grid.ny; int nz = grid.nz;
    double dV = step * step * step;
    
    struct PartialWaveAccumulator {
        std::complex<double> val_L;
        std::complex<double> val_R;
    };
    
    // Indexing: [energy_idx][l][m_idx][alpha]
    
    int num_lm = (l_max + 1) * (l_max + 1);
    int num_alpha = 3;
    
    // [energy][lm][alpha]
    // Global accumulator (merged results from threads)
    std::vector<std::vector<std::vector<PartialWaveAccumulator>>> final_partial_sums(
        num_energies, 
        std::vector<std::vector<PartialWaveAccumulator>>(
            num_lm,
            std::vector<PartialWaveAccumulator>(num_alpha, {0.0, 0.0})
        )
    );
    
    // Parallel Region
    #pragma omp parallel
    {
        // Thread-local Accumulator
        auto thread_partial_sums = final_partial_sums; // Copy structure (zeros)
        // Reset just in case copy not zero
        for(auto& e_vec : thread_partial_sums)
            for(auto& lm_vec : e_vec)
                for(auto& a_val : lm_vec) a_val = {0.0, 0.0};
        
        #pragma omp for
        for (int ix = 0; ix < nx; ++ix) {
            double x = x0 + ix * step;
            for (int iy = 0; iy < ny; ++iy) {
                 double y = y0 + iy * step;
                 for (int iz = 0; iz < nz; ++iz) {
                    double z = z0 + iz * step;
                    
                    int idx = ix * (ny * nz) + iy * nz + iz;
                    double phi_L = phi_L_vals[idx];
                    double phi_R = phi_R_vals[idx];
                    
                    if (std::abs(phi_L) < 1e-15 && std::abs(phi_R) < 1e-15) continue;
                    
                    double r_sq = x*x + y*y + z*z;
                    if (r_sq < 1e-18) continue;
                    double r = std::sqrt(r_sq);
                    
                    double theta = std::acos(z/r);
                    double phi_ang = std::atan2(y, x);
                    
                    double dip_x = x;
                    double dip_y = y;
                    double dip_z = z;
                    
                    // For each LM
                    int lm_idx = 0;
                    for(int l=0; l<=l_max; ++l) {
                        for(int m=-l; m<=l; ++m) {
                             std::complex<double> Ylm_conj = std::conj(MathSpecial::SphericalHarmonicY(l, m, theta, phi_ang));
                             
                             std::complex<double> termL_base = phi_L * Ylm_conj;
                             std::complex<double> termR_base = phi_R * Ylm_conj;
                             
                             for(int e=0; e<num_energies; ++e) {
                                 if(k_values[e] <= 0) continue;
                                 
                                 double k = k_values[e];
                                 double jl = MathSpecial::SphericalBesselJ(l, k*r);
                                 double factor = jl * dV;
                                 
                                 // Alpha 0 (x)
                                 thread_partial_sums[e][lm_idx][0].val_L += termL_base * dip_x * factor;
                                 thread_partial_sums[e][lm_idx][0].val_R += termR_base * dip_x * factor;
                                 
                                 // Alpha 1 (y)
                                 thread_partial_sums[e][lm_idx][1].val_L += termL_base * dip_y * factor;
                                 thread_partial_sums[e][lm_idx][1].val_R += termR_base * dip_y * factor;
                                 
                                 // Alpha 2 (z)
                                 thread_partial_sums[e][lm_idx][2].val_L += termL_base * dip_z * factor;
                                 thread_partial_sums[e][lm_idx][2].val_R += termR_base * dip_z * factor;
                             }
                             lm_idx++;
                        }
                    }
                 }
            }
        } // End Grid Loop (Parallel)
        
        // Merge thread results
        #pragma omp critical
        {
            for(int e=0; e<num_energies; ++e) {
                for(int lm=0; lm<num_lm; ++lm) {
                    for(int a=0; a<num_alpha; ++a) {
                        final_partial_sums[e][lm][a].val_L += thread_partial_sums[e][lm][a].val_L;
                        final_partial_sums[e][lm][a].val_R += thread_partial_sums[e][lm][a].val_R;
                    }
                }
            }
        }
    } // End Parallel
    
    // Final Assembly (Use merged results)
    for(int e=0; e<num_energies; ++e) {
        if(k_values[e] <= 0) {
            results.push_back(0.0);
            continue;
        }
        
        double E_ph_au = photon_energies_ev[e] * EV_TO_HARTREE;
        double k = k_values[e];
        double prefactor = (8.0 * M_PI * k * E_ph_au) / C_SPEED_AU;
        
        double total_sum = 0.0;
        
        // Sum over LM and Alpha
        for(int lm=0; lm<num_lm; ++lm) {
            double sum_alpha = 0.0;
            for(int alpha=0; alpha<3; ++alpha) {
                std::complex<double> amp_L = final_partial_sums[e][lm][alpha].val_L;
                std::complex<double> amp_R = final_partial_sums[e][lm][alpha].val_R;
                
                sum_alpha += (std::conj(amp_L) * amp_R).real();
            }
            total_sum += sum_alpha;
        }

        
        total_sum /= 3.0; // Average over three orthogonal directions
        double norms = dyson_L.qchem_norm * dyson_R.qchem_norm;
        double sigma = prefactor * total_sum * norms * 2.0;
        
        results.push_back(sigma);
    }

    return results;
}


#include "point_dipole.h"

std::vector<double> CrossSectionCalculator::ComputePointDipoleCrossSection(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const std::vector<double>& photon_energies_ev,
    double ionization_energy_ev,
    int l_max,
    double dipole_magnitude
) {
    std::vector<double> results;
    results.reserve(photon_energies_ev.size());
    
    // Fill Grid Caches
    std::vector<double> phi_L_vals;
    std::vector<double> phi_R_vals;
    phi_L_vals.reserve(grid.nx * grid.ny * grid.nz);
    phi_R_vals.reserve(grid.nx * grid.ny * grid.nz);
    
    double x0 = grid.xmin; double y0 = grid.ymin; double z0 = grid.zmin;
    double step = grid.dx;
    
    for (int ix = 0; ix < grid.nx; ++ix) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int iz = 0; iz < grid.nz; ++iz) {
                 double x = x0 + ix * step;
                 double y = y0 + iy * step;
                 double z = z0 + iz * step;
                 phi_L_vals.push_back(dyson_L.evaluate(x, y, z));
                 phi_R_vals.push_back(dyson_R.evaluate(x, y, z));
            }
        }
    }
    
    std::vector<double> k_values;
    for(double E_ev : photon_energies_ev) {
        double E_au = E_ev * EV_TO_HARTREE;
        double IE_au = ionization_energy_ev * EV_TO_HARTREE;
        double eKE = E_au - IE_au;
        if(eKE > 0) k_values.push_back(std::sqrt(2.0 * eKE));
        else k_values.push_back(0.0);
    }
    
    PointDipole pd(dipole_magnitude);
    
    int nx = grid.nx; int ny = grid.ny; int nz = grid.nz;
    double dV = step * step * step;

    // Loop Energies
    for(size_t e=0; e < photon_energies_ev.size(); ++e) {
        if (k_values[e] <= 0) {
            results.push_back(0.0);
            continue;
        }
        
        double k = k_values[e];
        double total_sum = 0.0;
        
        // Sum over m and N
        for (int lam = -l_max; lam <= l_max; ++lam) {
            Eigensystem sys = pd.GetEigensystem(lam, l_max);
            
            int n_modes = sys.l_vals.size();
            for (int N = 0; N < n_modes; ++N) {
                double eigval = sys.eigvals[N];
                if (eigval < -0.25) continue;
                
                double L_eff = 0.5 * (-1.0 + std::sqrt(1.0 + 4.0 * eigval));
                
                // Compute Moment vector I_alpha for this mode (L and R)
                // I_alpha = Sum_l c_{lN}^* Integral( phi * r_alpha * Ylm* * j_{Leff}(kr) )
                
                std::complex<double> I_L[3] = {0.0, 0.0, 0.0};
                std::complex<double> I_R[3] = {0.0, 0.0, 0.0};
                
                // Pre-calculate integrals for each l component: J_{l,alpha}
                // Loop Grid -> Compute Ylm* -> Add to accumulators for each l.
                
                std::vector<std::vector<std::complex<double>>> partial_integrals_L(n_modes, std::vector<std::complex<double>>(3, 0.0));
                std::vector<std::vector<std::complex<double>>> partial_integrals_R(n_modes, std::vector<std::complex<double>>(3, 0.0));
            }
        }
        
        struct Moment { std::complex<double> val_L; std::complex<double> val_R; };
        
        std::vector<std::vector<std::vector<Moment>>> mode_moments; 
        // mode_moments[lam_offset][N][alpha]
        mode_moments.resize(2*l_max + 1);
        
        // Setup systems and resize
        for (int lam = -l_max; lam <= l_max; ++lam) {
            Eigensystem sys = pd.GetEigensystem(lam, l_max);
            mode_moments[lam + l_max].resize(sys.l_vals.size(), std::vector<Moment>(3, {0.0, 0.0}));
        }
        
        #pragma omp parallel
        {
            auto local_moments = mode_moments;
            // Zero out
            for(auto& v1 : local_moments) for(auto& v2 : v1) for(auto& m : v2) m = {0.0, 0.0};
            
            // Thread-local reusable buffer for Y_vals (max size l_max + 1)
            std::vector<std::complex<double>> Y_vals;
            Y_vals.reserve(l_max + 1);
            
            #pragma omp for
            for (int ix = 0; ix < nx; ++ix) {
                for (int iy = 0; iy < ny; ++iy) {
                    for (int iz = 0; iz < nz; ++iz) {
                        int idx = ix * ny * nz + iy * nz + iz;
                        double phi_L = phi_L_vals[idx];
                        double phi_R = phi_R_vals[idx];
                        
                        if (std::abs(phi_L) < 1e-15 && std::abs(phi_R) < 1e-15) continue;
                        
                        double x = x0 + ix * step;
                        double y = y0 + iy * step;
                        double z = z0 + iz * step;
                        double r_sq = x*x + y*y + z*z;
                        if (r_sq < 1e-18) continue;
                        double r = std::sqrt(r_sq);
                        double theta = std::acos(z/r);
                        double phi_ang = std::atan2(y, x);
                        
                        double dip[3] = {x, y, z};
                        
                        // Per m
                        for (int lam = -l_max; lam <= l_max; ++lam) {
                            Eigensystem sys = pd.GetEigensystem(lam, l_max);
                            int n_modes = sys.l_vals.size();
                            
                            // We need Sum_l c_{lN}* Y_{l,lam}*
                            // Optimization: Calculate Y_{l,lam}* for all l first.
                            Y_vals.resize(n_modes);
                            for(int i=0; i<n_modes; ++i) {
                                Y_vals[i] = std::conj(MathSpecial::SphericalHarmonicY(sys.l_vals[i], lam, theta, phi_ang));
                            }
                            
                            for (int N = 0; N < n_modes; ++N) {
                                double eigval = sys.eigvals[N];
                                if (eigval < -0.25) continue;
                                double L_eff = 0.5 * (-1.0 + std::sqrt(1.0 + 4.0 * eigval));
                                
                                // Radial part
                                double bessel = MathSpecial::CylBesselJ(L_eff + 0.5, k*r);
                                double radial = std::sqrt(M_PI/(2.0*k*r)) * bessel; 
                                if (k*r < 1e-10) { // Limit case
                                   radial = (L_eff < 0.1) ? 1.0 : 0.0;
                                }
                                
                                std::complex<double> omega_conj = 0.0;
                                for(int i=0; i<n_modes; ++i) {
                                    omega_conj += sys.eigvecs[i][N] * Y_vals[i];
                                }
                                
                                std::complex<double> basis_val = omega_conj * radial * dV;
                                
                                // Accumulate
                                for(int alpha=0; alpha<3; ++alpha) {
                                    local_moments[lam+l_max][N][alpha].val_L += phi_L * dip[alpha] * basis_val;
                                    local_moments[lam+l_max][N][alpha].val_R += phi_R * dip[alpha] * basis_val;
                                }
                            }
                        }
                    } 
                }
            } // End Grid
            
            #pragma omp critical
            {
                for(int m=0; m < (int)local_moments.size(); ++m) {
                    for(int N=0; N < (int)local_moments[m].size(); ++N) {
                         for(int a=0; a<3; ++a) {
                             mode_moments[m][N][a].val_L += local_moments[m][N][a].val_L;
                             mode_moments[m][N][a].val_R += local_moments[m][N][a].val_R;
                         }
                    }
                }
            }
        } // End Parallel
        
        // Sum Contribution
        double prefactor = (8.0 * M_PI * k * (photon_energies_ev[e] * EV_TO_HARTREE)) / C_SPEED_AU;
        
        for(int lam = -l_max; lam <= l_max; ++lam) {
             for(int N = 0; N < (int)mode_moments[lam+l_max].size(); ++N) {
                 double sum_alpha = 0.0;
                 for(int a=0; a<3; ++a) {
                     std::complex<double> mL = mode_moments[lam+l_max][N][a].val_L;
                     std::complex<double> mR = mode_moments[lam+l_max][N][a].val_R;
                     sum_alpha += std::real(std::conj(mL) * mR);
                 }
                 total_sum += sum_alpha;
             }
        }
        
        results.push_back(total_sum * prefactor / 3.0 * dyson_L.qchem_norm * dyson_R.qchem_norm * 2.0);
    }
    
    return results;
}

#include <map>
#include <set>
#include <algorithm>

RelativeXSResult CrossSectionCalculator::ComputeRelativeCrossSections(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const std::vector<double>& photon_energies_ev,
    const std::vector<VibState>& vib_states,
    int l_max,
    double dipole_magnitude,
    bool use_point_dipole,
    double dipole_length
) {
    RelativeXSResult result;
    result.photon_energies = photon_energies_ev;
    int n_energies = photon_energies_ev.size();
    int n_channels = vib_states.size();
    
    result.total_cross_section.resize(n_energies, 0.0);
    result.channel_fractions.resize(n_energies, std::vector<double>(n_channels, 0.0));
    
    // 1. Identify required unique eKEs
    std::set<double> unique_eKEs;
    for (double E_ph : photon_energies_ev) {
        for (const auto& vib : vib_states) {
            double eKE = E_ph - vib.energy_bind;
            if (eKE > 0.0001) { // Threshold
                unique_eKEs.insert(eKE);
            }
        }
    }
    
    if (unique_eKEs.empty()) return result; // All below threshold
    
    std::vector<double> calc_energies(unique_eKEs.begin(), unique_eKEs.end());
    
    // 2. Compute Base Cross Sections (IE=0, so E_input interpreted as eKE)
    std::vector<double> base_xs;
    
    // Logic: 
    // If dipole_length > 0 -> Physical Dipole
    // Else If use_point_dipole -> Point Dipole
    // Else -> PWE
    
    if (dipole_length > 1e-6) {
        base_xs = ComputePhysicalDipoleCrossSection(
            dyson_L, dyson_R, grid, calc_energies, 0.0, l_max, dipole_magnitude, dipole_length
        );
    } else if (use_point_dipole) {
        base_xs = ComputePointDipoleCrossSection(
            dyson_L, dyson_R, grid, calc_energies, 0.0, l_max, dipole_magnitude
        );
    } else {
        base_xs = ComputeTotalCrossSection(
            dyson_L, dyson_R, grid, calc_energies, 0.0, l_max
        );
    }
    
    // Map eKE -> XS
    std::map<double, double> xs_map;
    for(size_t i=0; i<calc_energies.size(); ++i) {
        xs_map[calc_energies[i]] = base_xs[i];
    }
    
    // 3. Aggregate
    for(int i=0; i<n_energies; ++i) {
        double E_ph = photon_energies_ev[i];
        
        std::vector<double> weights(n_channels, 0.0);
        double sum_weights = 0.0;
        
        for(int c=0; c<n_channels; ++c) {
            double eKE = E_ph - vib_states[c].energy_bind;
            if (eKE > 0.0001) {
                auto it = xs_map.find(eKE);
                double sigma_base = 0.0;
                if (it != xs_map.end()) {
                    sigma_base = it->second;
                } else {
                     auto it2 = unique_eKEs.lower_bound(eKE - 1e-7);
                     if (it2 != unique_eKEs.end() && std::abs(*it2 - eKE) < 1e-6) {
                         sigma_base = xs_map[*it2];
                     }
                }
                
                double correction = 1.0;
                if (eKE > 1e-9) {
                    correction = E_ph / eKE;
                }
                
                double sigma_corr = sigma_base * correction;
                double w = (vib_states[c].fcf_overlap * vib_states[c].fcf_overlap) * sigma_corr;
                weights[c] = w;
                sum_weights += w;
            }
        }
        
        result.total_cross_section[i] = sum_weights;
        
        if (sum_weights > 1e-20) {
            for(int c=0; c<n_channels; ++c) {
                result.channel_fractions[i][c] = weights[c] / sum_weights;
            }
        }
    }
    
    return result;
}

#include "physical_dipole.h"

std::vector<double> CrossSectionCalculator::ComputePhysicalDipoleCrossSection(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const std::vector<double>& photon_energies_ev,
    double ionization_energy_ev,
    int l_max,
    double dipole_magnitude,
    double dipole_length,
    const std::vector<double>& dipole_axis,
    const std::vector<double>& dipole_center
) {
    std::vector<double> results;
    results.reserve(photon_energies_ev.size());
    
    // Construct Rotation Matrix R such that R * v_global = v_local
    
    std::vector<double> d_hat = dipole_axis;
    double d_norm = std::sqrt(d_hat[0]*d_hat[0] + d_hat[1]*d_hat[1] + d_hat[2]*d_hat[2]);
    if (d_norm > 1e-9) {
        d_hat[0] /= d_norm; d_hat[1] /= d_norm; d_hat[2] /= d_norm;
    } else {
        d_hat = {0,0,1};
    }
    
    // Basis Vectors for Local Frame (rows of R)
    // z_prime = d_hat
    // Construct x_prime, y_prime orthonormal
    double zp[3] = {d_hat[0], d_hat[1], d_hat[2]};
    double xp[3] = {0,0,0};
    double yp[3] = {0,0,0};
    
    // Pick arbitrary temp vector not parallel to zp
    double tmp[3] = {1,0,0};
    if (std::abs(zp[0]) > 0.9) { tmp[0]=0; tmp[1]=1; tmp[2]=0; }
    
    double yp_raw[3];
    yp_raw[0] = zp[1]*tmp[2] - zp[2]*tmp[1];
    yp_raw[1] = zp[2]*tmp[0] - zp[0]*tmp[2];
    yp_raw[2] = zp[0]*tmp[1] - zp[1]*tmp[0];
    
    double yp_norm = std::sqrt(yp_raw[0]*yp_raw[0] + yp_raw[1]*yp_raw[1] + yp_raw[2]*yp_raw[2]);
    yp[0] = yp_raw[0]/yp_norm; yp[1] = yp_raw[1]/yp_norm; yp[2] = yp_raw[2]/yp_norm;
    
    // xp = cross(yp, zp)
    xp[0] = yp[1]*zp[2] - yp[2]*zp[1];
    xp[1] = yp[2]*zp[0] - yp[0]*zp[2];
    xp[2] = yp[0]*zp[1] - yp[1]*zp[0];
    
    // Rotation Matrix R_mat[3][3]
    // v_local = R * v_global
    double R_mat[3][3] = {
        {xp[0], xp[1], xp[2]},
        {yp[0], yp[1], yp[2]},
        {zp[0], zp[1], zp[2]}
    }; // Rows are x', y', z'
    
    // Fill Grids
    std::vector<double> phi_L_vals(grid.nx * grid.ny * grid.nz);
    std::vector<double> phi_R_vals(grid.nx * grid.ny * grid.nz);
    
    double x0 = grid.xmin; double y0 = grid.ymin; double z0 = grid.zmin;
    double step = grid.dx;
    double dV = step * step * step;
    
    #pragma omp parallel for collapse(3)
    for (int ix = 0; ix < grid.nx; ++ix) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int iz = 0; iz < grid.nz; ++iz) {
                 double x = x0 + ix * step;
                 double y = y0 + iy * step;
                 double z = z0 + iz * step;
                 int idx = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;
                 phi_L_vals[idx] = dyson_L.evaluate(x, y, z);
                 phi_R_vals[idx] = dyson_R.evaluate(x, y, z);
            }
        }
    }
    
    // D = Dipole Strength
    PhysicalDipole phys(dipole_length, dipole_magnitude);
    
    // Loop Energies
    for (size_t e = 0; e < photon_energies_ev.size(); ++e) {
        double E_ph_ev = photon_energies_ev[e];
        double E_ph_au = E_ph_ev * EV_TO_HARTREE;
        double IE_au = ionization_energy_ev * EV_TO_HARTREE;
        double eKE = E_ph_au - IE_au;
        
        if (eKE <= 0) {
            results.push_back(0.0);
            continue;
        }
        
        double k = std::sqrt(2.0 * eKE); // Atomic units
        double total_sum = 0.0;
        
        // Loop m
        for (int m = -l_max; m <= l_max; ++m) {
            // Solve for this Energy/m
            auto sol = phys.Solve(eKE, m, l_max);
            int n_modes = sol.radial_solutions.size();
            
            // Loop Modes
            for (int n = 0; n < n_modes; ++n) {
                
                std::complex<double> I_L[3] = {0.0, 0.0, 0.0};
                std::complex<double> I_R[3] = {0.0, 0.0, 0.0};
                
                double IL_real[3] = {0,0,0}; double IL_imag[3] = {0,0,0};
                double IR_real[3] = {0,0,0}; double IR_imag[3] = {0,0,0};
                
                #pragma omp parallel for collapse(3) reduction(+:IL_real[:3], IL_imag[:3], IR_real[:3], IR_imag[:3])
                for (int ix = 0; ix < grid.nx; ++ix) {
                    for (int iy = 0; iy < grid.ny; ++iy) {
                        for (int iz = 0; iz < grid.nz; ++iz) {
                             int idx = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;
                             double phi_L = phi_L_vals[idx];
                             double phi_R = phi_R_vals[idx];
                             if (std::abs(phi_L) < 1e-15 && std::abs(phi_R) < 1e-15) continue;
                             
                             double x = x0 + ix * step;
                             double y = y0 + iy * step;
                             double z = z0 + iz * step;
                             
                             // Shift to Dipole Frame
                             double xs = x - dipole_center[0];
                             double ys = y - dipole_center[1];
                             double zs = z - dipole_center[2];
                             
                             // Rotate to Local Frame for Wavefunction Evaluation
                             double x_loc = R_mat[0][0]*xs + R_mat[0][1]*ys + R_mat[0][2]*zs;
                             double y_loc = R_mat[1][0]*xs + R_mat[1][1]*ys + R_mat[1][2]*zs;
                             double z_loc = R_mat[2][0]*xs + R_mat[2][1]*ys + R_mat[2][2]*zs;
                             
                             std::complex<double> Psi = phys.EvaluateMode(x_loc, y_loc, z_loc, m, n, sol);
                             std::complex<double> Psi_conj = std::conj(Psi);
                             
                             double dip[3] = {x, y, z};

                             for(int a=0; a<3; ++a) {
                                 std::complex<double> termL = phi_L * dip[a] * Psi_conj * dV;
                                 std::complex<double> termR = phi_R * dip[a] * Psi_conj * dV;
                                 
                                 IL_real[a] += termL.real();
                                 IL_imag[a] += termL.imag();
                                 IR_real[a] += termR.real();
                                 IR_imag[a] += termR.imag();
                             }
                        }
                    }
                }
                
                // Aggregate moments
                for(int a=0; a<3; ++a) {
                    I_L[a] = std::complex<double>(IL_real[a], IL_imag[a]);
                    I_R[a] = std::complex<double>(IR_real[a], IR_imag[a]);
                }
                
                // Add to Total Sum (Average polarization sum)
                // |I|^2 = sum_alpha Re(I_L^* * I_R)
                double sum_alpha = 0.0;
                for(int a=0; a<3; ++a) {
                    sum_alpha += std::real(std::conj(I_L[a]) * I_R[a]);
                }
                total_sum += sum_alpha;
            }
        }
        
        double prefactor = (8.0 * M_PI * k * E_ph_au) / C_SPEED_AU;
        
        results.push_back(total_sum * prefactor / 3.0 * dyson_L.qchem_norm * dyson_R.qchem_norm * 2.0);
    }
    
    return results;
}

std::vector<std::vector<CrossSectionCalculator::DipoleMatrixElement>> CrossSectionCalculator::ComputePhysicalDipoleMatrixElements(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const std::vector<double>& photon_energies_ev,
    double ionization_energy_ev,
    int l_max,
    double dipole_magnitude,
    double dipole_length,
    const std::vector<double>& dipole_axis,
    const std::vector<double>& dipole_center
) {
    std::vector<std::vector<DipoleMatrixElement>> all_matrix_elements;
    all_matrix_elements.reserve(photon_energies_ev.size());
    
    // 1. Construct Rotation Matrix R to Local Frame (Dipole along Z)
    std::vector<double> d_hat = dipole_axis;
    double d_norm = std::sqrt(d_hat[0]*d_hat[0] + d_hat[1]*d_hat[1] + d_hat[2]*d_hat[2]);
    if (d_norm > 1e-9) {
        d_hat[0] /= d_norm; d_hat[1] /= d_norm; d_hat[2] /= d_norm;
    } else {
        d_hat = {0,0,1};
    }
    
    double zp[3] = {d_hat[0], d_hat[1], d_hat[2]};
    double tmp[3] = {1,0,0};
    if (std::abs(zp[0]) > 0.9) { tmp[0]=0; tmp[1]=1; tmp[2]=0; }
    
    double yp_raw[3];
    yp_raw[0] = zp[1]*tmp[2] - zp[2]*tmp[1];
    yp_raw[1] = zp[2]*tmp[0] - zp[0]*tmp[2];
    yp_raw[2] = zp[0]*tmp[1] - zp[1]*tmp[0];
    double yp_norm = std::sqrt(yp_raw[0]*yp_raw[0] + yp_raw[1]*yp_raw[1] + yp_raw[2]*yp_raw[2]);
    double yp[3] = {yp_raw[0]/yp_norm, yp_raw[1]/yp_norm, yp_raw[2]/yp_norm};
    
    double xp[3];
    xp[0] = yp[1]*zp[2] - yp[2]*zp[1];
    xp[1] = yp[2]*zp[0] - yp[0]*zp[2];
    xp[2] = yp[0]*zp[1] - yp[1]*zp[0];
    
    double R_mat[3][3] = {
        {xp[0], xp[1], xp[2]},
        {yp[0], yp[1], yp[2]},
        {zp[0], zp[1], zp[2]}
    };
    
    // 2. Pre-compute Dyson Values
    std::vector<double> phi_L_vals(grid.nx * grid.ny * grid.nz);
    std::vector<double> phi_R_vals(grid.nx * grid.ny * grid.nz);
    
    double x0 = grid.xmin; double y0 = grid.ymin; double z0 = grid.zmin;
    double step = grid.dx;
    double dV = step * step * step;
    
    #pragma omp parallel for collapse(3)
    for (int ix = 0; ix < grid.nx; ++ix) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int iz = 0; iz < grid.nz; ++iz) {
                 double x = x0 + ix * step;
                 double y = y0 + iy * step;
                 double z = z0 + iz * step;
                 int idx = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;
                 phi_L_vals[idx] = dyson_L.evaluate(x, y, z);
                 phi_R_vals[idx] = dyson_R.evaluate(x, y, z);
            }
        }
    }
    
    PhysicalDipole phys(dipole_length, dipole_magnitude);
    
    // 3. Loop Energies
    for (size_t e = 0; e < photon_energies_ev.size(); ++e) {
        std::vector<DipoleMatrixElement> current_energy_elements;
        
        double E_ph_ev = photon_energies_ev[e];
        double E_ph_au = E_ph_ev * EV_TO_HARTREE;
        double IE_au = ionization_energy_ev * EV_TO_HARTREE;
        double eKE = E_ph_au - IE_au;
        
        if (eKE <= 0) {
            all_matrix_elements.push_back(current_energy_elements);
            continue;
        }
        
        // Loop m
        for (int m = -l_max; m <= l_max; ++m) {
            auto sol = phys.Solve(eKE, m, l_max);
            int n_modes = sol.radial_solutions.size();
            
            // For each mode n
            for (int n = 0; n < n_modes; ++n) {
                // Compute Integral Vector I
                double IL_real[3] = {0,0,0}; double IL_imag[3] = {0,0,0};
                double IR_real[3] = {0,0,0}; double IR_imag[3] = {0,0,0};
                
                #pragma omp parallel for collapse(3) reduction(+:IL_real[:3], IL_imag[:3], IR_real[:3], IR_imag[:3])
                for (int ix = 0; ix < grid.nx; ++ix) {
                    for (int iy = 0; iy < grid.ny; ++iy) {
                        for (int iz = 0; iz < grid.nz; ++iz) {
                             int idx = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;
                             double phi_L = phi_L_vals[idx];
                             double phi_R = phi_R_vals[idx];
                             if (std::abs(phi_L) < 1e-15 && std::abs(phi_R) < 1e-15) continue;
                             
                             double x = x0 + ix * step;
                             double y = y0 + iy * step;
                             double z = z0 + iz * step;
                             
                             // Global indices for dipole operator r
                             double dip[3] = {x, y, z}; 
                             
                             // Local coords for wavefunction
                             double xs = x - dipole_center[0];
                             double ys = y - dipole_center[1];
                             double zs = z - dipole_center[2];
                             double x_loc = R_mat[0][0]*xs + R_mat[0][1]*ys + R_mat[0][2]*zs;
                             double y_loc = R_mat[1][0]*xs + R_mat[1][1]*ys + R_mat[1][2]*zs;
                             double z_loc = R_mat[2][0]*xs + R_mat[2][1]*ys + R_mat[2][2]*zs;
                             
                             std::complex<double> Psi = phys.EvaluateMode(x_loc, y_loc, z_loc, m, n, sol);
                             std::complex<double> Psi_conj = std::conj(Psi);
                             
                             // Integration
                             for(int a=0; a<3; ++a) {
                                 std::complex<double> termL = phi_L * dip[a] * Psi_conj * dV;
                                 std::complex<double> termR = phi_R * dip[a] * Psi_conj * dV;
                                 
                                 IL_real[a] += termL.real();
                                 IL_imag[a] += termL.imag();
                                 IR_real[a] += termR.real();
                                 IR_imag[a] += termR.imag();
                             }
                        }
                    }
                }
                
                std::complex<double> I_avg[3];
                for(int a=0; a<3; ++a) {
                    std::complex<double> vL(IL_real[a], IL_imag[a]);
                    std::complex<double> vR(IR_real[a], IR_imag[a]);
                    I_avg[a] = 0.5 * (vL + vR);
                }
                
                DipoleMatrixElement elem;
                elem.E_ph = E_ph_ev;
                elem.m = m;
                elem.n_mode = n;
                if (n < (int)sol.radial_solutions.size()) {
                    elem.nu = sol.radial_solutions[n].nu;
                } else {
                    elem.nu = 0.0;
                }
                elem.I_x = I_avg[0];
                elem.I_y = I_avg[1];
                elem.I_z = I_avg[2];
                
                current_energy_elements.push_back(elem);
            }
        }
        all_matrix_elements.push_back(current_energy_elements);
    }
    
    return all_matrix_elements;
}


