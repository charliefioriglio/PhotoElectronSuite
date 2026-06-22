#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "tools.h"
#include "molecule.h"
#include "dyson.h"
#include "grid.h"
#include "cross_section.h"
#include "cross_section.h"

// Input format expected from Python script:
// ATOM_COUNT
// SYMBOL INDEX X Y Z
// ...
// SHELL_COUNT
// ATOM_IDX L IS_PURE N_PRIM
// EXP COEF
// ...
// DYSON_COEFF_COUNT
// C_0 C_1 ...
// GRID_PARAMS
// X0 X1 Y0 Y1 Z0 Z1 STEP

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input_file output_file" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    std::ifstream in(input_file);
    if (!in) {
        std::cerr << "Error: Could not open input file " << input_file << std::endl;
        return 1;
    }

    Molecule mol;
    int n_atoms;
    if (!(in >> n_atoms)) return 1;

    for (int i = 0; i < n_atoms; ++i) {
        std::string sym;
        int idx;
        double x, y, z;
        in >> sym >> idx >> x >> y >> z;
        mol.add_atom(sym, idx, x, y, z);
    }
    
    int n_shells;
    in >> n_shells;
    for (int i = 0; i < n_shells; ++i) {
        int atom_idx, l, n_prim;
        bool is_pure;
        in >> atom_idx >> l >> is_pure >> n_prim;
        
        std::vector<double> exps(n_prim);
        std::vector<double> coeffs(n_prim);
        for (int j = 0; j < n_prim; ++j) {
            in >> exps[j] >> coeffs[j];
        }
        mol.add_shell_to_atom(atom_idx, l, is_pure, exps, coeffs);
    }
    
    int num_dyson_orbs;
    if (!(in >> num_dyson_orbs)) num_dyson_orbs = 1; // Backward compatibility fallback
    std::vector<Dyson> dysons;
    
    for (int d = 0; d < num_dyson_orbs; ++d) {
        int n_coeffs;
        double norm_val = 1.0;
        in >> n_coeffs >> norm_val; // Expecting "N NORM"
        
        std::vector<double> coeffs(n_coeffs);
        for (int i = 0; i < n_coeffs; ++i) in >> coeffs[i];
        
        // Label
        std::string label = (d==0) ? "Left/Primary" : "Right/Secondary";
        Dyson d_obj(&mol, coeffs, label);
        d_obj.qchem_norm = norm_val; // Store new property
        dysons.push_back(d_obj);
    }
    
    double x0, x1, y0, y1, z0, z1, step;
    in >> x0 >> x1 >> y0 >> y1 >> z0 >> z1 >> step;
    
    // Centering Logic (Identical to beta_gen.cpp)
    // 1. Calculate Centroid using the bounding box of the grid
    // std::cout << "Calculating Dyson centroid..." << std::endl;
    const Dyson& L_orig = dysons[0];
    Dyson::Vector3 centroid = L_orig.get_centroid(x0, x1, y0, y1, z0, z1, step);
    
    // std::cout << "Dyson Centroid: (" << centroid.x << ", " << centroid.y << ", " << centroid.z << ")" << std::endl;
    // std::cout << "Shifting molecule and grid to center Dyson orbital at (0,0,0)..." << std::endl;
    
    // 2. Shift Molecule
    mol.shift_geometry(-centroid.x, -centroid.y, -centroid.z);
    
    // 3. Update Dyson Basis Caches
    for(auto& d : dysons) {
        d.update_geometry();
    }
    
    // 4. Shift Grid Limits to match new centered frame
    // If the orbital moves by -C, the grid window must also move by -C to stay centered on it
    x0 -= centroid.x; x1 -= centroid.x;
    y0 -= centroid.y; y1 -= centroid.y;
    z0 -= centroid.z; z1 -= centroid.z;
    
    // Renormalize all
    for(auto& do_obj : dysons) {
        do_obj.renormalize(x0, x1, y0, y1, z0, z1, step);
    }
    
    // Evaluate grid (Primary/Left orbital)
    UniformGrid grid(x0, x1, y0, y1, z0, z1, step);
    std::vector<double> data(grid.nx * grid.ny * grid.nz);
    
    const Dyson& primary_dyson = dysons[0];

    #pragma omp parallel for collapse(3)
    for (int ix = 0; ix < grid.nx; ++ix) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int iz = 0; iz < grid.nz; ++iz) {
                double x = grid.xmin + ix * grid.dx;
                double y = grid.ymin + iy * grid.dy;
                double z = grid.zmin + iz * grid.dz;
                int idx = ix * (grid.ny * grid.nz) + iy * grid.nz + iz;
                data[idx] = primary_dyson.evaluate(x, y, z);
            }
        }
    }
    
    grid.write_binary(output_file, data);

    // XS Params
    double ie_ev, e_min, e_max;
    int input_l_max, n_pts;
    
    std::vector<double> dipoles;
    bool explicit_list = false;
    bool use_point_dipole = false;
    int override_lmax = -1;
    std::string args_xs_out = "";
    double args_dipole_length = 0.0;
    bool run_xs_calculation = false;
    
    // Parse Args for Dipoles
    for(int i=3; i<argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--point-dipole" && i+1 < argc) {
             dipoles = { std::stod(argv[++i]) };
             use_point_dipole = true;
        }
        else if (arg == "--point-dipole-list") {
             explicit_list = true;
             dipoles.clear();
             while (i+1 < argc) {
                  std::string val = argv[i+1];
                  if (val.substr(0, 2) == "--") break;
                  try { dipoles.push_back(std::stod(val)); } catch(...) {}
                  i++;
             }
        }
        else if (arg == "--lmax" && i+1 < argc) {
             try { override_lmax = std::stoi(argv[++i]); } catch(...) {}
        }
        else if (arg == "--xs-out" && i+1 < argc) {
             args_xs_out = argv[++i];
        }
        else if (arg == "--dipole-length" && i+1 < argc) {
             try { args_dipole_length = std::stod(argv[++i]); } catch(...) {}
        }
        else if (arg == "--run-xs") {
             run_xs_calculation = true;
        }
    }
    
    if (dipoles.empty()) dipoles = {0.0};

    if (run_xs_calculation && (in >> ie_ev >> input_l_max >> n_pts)) {
        int l_max = (override_lmax != -1) ? override_lmax : input_l_max;
        
        std::vector<double> energies(n_pts);
        for(int i=0; i<n_pts; ++i) in >> energies[i];
        
        int vib_count = 0;
        std::vector<VibState> vib_states;
        if (in >> vib_count) {
             for(int i=0; i<vib_count; ++i) {
                 double eb, fcf;
                 in >> eb >> fcf;
                 vib_states.push_back({eb, fcf});
             }
        }
        
        std::cout << "Reading XS params successful. IE=" << ie_ev << " LMax=" << input_l_max << " NPts=" << n_pts << std::endl;
        const Dyson& L = dysons[0];
        const Dyson& R = (dysons.size() > 1) ? dysons[1] : dysons[0];
        
        // Loop over dipoles
        std::cout << "Dipoles count: " << dipoles.size() << std::endl;
        for (double d_val : dipoles) {
            std::string d_str = std::to_string(d_val);
            // Simple trim
            d_str.erase(d_str.find_last_not_of('0') + 1, std::string::npos);
            if(d_str.back() == '.') d_str.pop_back();
            if(d_str.empty()) d_str = "0";

            if (vib_count > 0) {
                // Relative XS Mode
                RelativeXSResult res = CrossSectionCalculator::ComputeRelativeCrossSections(
                    L, R, grid, energies, vib_states, l_max, d_val, use_point_dipole || explicit_list, args_dipole_length
                );
                
                std::string fname = "xs_relative_D" + d_str + ".csv";
                if (!explicit_list) fname = "xs_relative.csv"; // Legacy compat

                std::ofstream out_rel(fname);
                out_rel << "E_photon,Total_XS";
                for(size_t i=0; i<vib_states.size(); ++i) out_rel << ",Rel_XS_Ch" << i;
                out_rel << "\n";
                
                for(size_t i=0; i<res.photon_energies.size(); ++i) {
                    out_rel << res.photon_energies[i] << "," << res.total_cross_section[i];
                    for(double frac : res.channel_fractions[i]) out_rel << "," << frac;
                    out_rel << "\n";
                }
                out_rel.close();
                
            } else {
                // Standard Mode
                std::vector<double> xs;
                
                if (args_dipole_length > 1e-6) {
                    std::cout << "Running Physical Dipole XS (D=" << d_val << ", a=" << args_dipole_length << ")..." << std::endl;
                    xs = CrossSectionCalculator::ComputePhysicalDipoleCrossSection(
                        L, R, grid, energies, ie_ev, l_max, d_val, args_dipole_length
                    );
                } else if (use_point_dipole || explicit_list) {
                     xs = CrossSectionCalculator::ComputePointDipoleCrossSection(
                        L, R, grid, energies, ie_ev, l_max, d_val
                    );
                } else {
                     xs = CrossSectionCalculator::ComputeTotalCrossSection(
                        L, R, grid, energies, ie_ev, l_max
                    );
                }
                
                std::string fname = "cross_section_D" + d_str + ".txt";
                if (!explicit_list) {
                    if (!args_xs_out.empty()) fname = args_xs_out;
                    else fname = "cross_section.txt"; 
                } 

                std::cout << "Writing XS to " << fname << std::endl;
                std::ofstream xs_file(fname);
                if (!xs_file) std::cerr << "Error opening file " << fname << std::endl;
                xs_file << "Energy,CrossSection\n";
                for(size_t i=0; i<energies.size(); ++i) {
                    xs_file << energies[i] << "," << xs[i] << "\n";
                }
                xs_file.close();
            }
        }
    }
    
    return 0;
}
