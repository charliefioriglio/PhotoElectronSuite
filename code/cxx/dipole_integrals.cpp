#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cross_section.h"
#include "dyson.h"
#include "grid.h"
#include "molecule.h"
#include "tools.h"

namespace {

struct PairNM {
    int n;
    int m;
};

std::vector<double> parse_double_list(int& i, int argc, char** argv) {
    std::vector<double> values;
    int j = i + 1;
    while (j < argc) {
        std::string val = argv[j];
        if (val.rfind("--", 0) == 0) break;
        try {
            values.push_back(std::stod(val));
        } catch (...) {
            break;
        }
        ++j;
    }
    i = j - 1;
    return values;
}

std::vector<PairNM> parse_pairs(int& i, int argc, char** argv) {
    std::vector<PairNM> pairs;
    int j = i + 1;
    while (j < argc) {
        std::string token = argv[j];
        if (token.rfind("--", 0) == 0) break;
        auto comma = token.find(',');
        if (comma == std::string::npos) break;
        try {
            int n = std::stoi(token.substr(0, comma));
            int m = std::stoi(token.substr(comma + 1));
            pairs.push_back({n, m});
        } catch (...) {
            break;
        }
        ++j;
    }
    i = j - 1;
    return pairs;
}

void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " cpp_input.dat output.csv [options]\n"
              << "Options:\n"
              << "  --eph <eV>             Photon energy (default 0.5)\n"
              << "  --lmax <int>           Maximum l (default 10)\n"
              << "  --a <bohr>             Physical dipole length a (default 1.5)\n"
              << "  --d-values <list>      Dipole strengths D (default 0 0.5 1 1.5 2 2.5 3)\n"
              << "  --pairs n,m ...        n,m pairs (default 0,0 1,1 1,0 2,2 2,1 2,0)\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    double eph = 0.5;
    int l_max = 10;
    double dipole_length = 1.5;

    std::vector<double> d_values = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    std::vector<PairNM> pairs = {
        {0, 0},
        {1, 1},
        {1, 0},
        {2, 2},
        {2, 1},
        {2, 0},
    };

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--eph" && i + 1 < argc) {
            eph = std::stod(argv[++i]);
        } else if (arg == "--lmax" && i + 1 < argc) {
            l_max = std::stoi(argv[++i]);
        } else if (arg == "--a" && i + 1 < argc) {
            dipole_length = std::stod(argv[++i]);
        } else if (arg == "--d-values") {
            auto vals = parse_double_list(i, argc, argv);
            if (!vals.empty()) d_values = vals;
        } else if (arg == "--pairs") {
            auto vals = parse_pairs(i, argc, argv);
            if (!vals.empty()) pairs = vals;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    std::ifstream in(input_file);
    if (!in) {
        std::cerr << "Error: Could not open input file " << input_file << "\n";
        return 1;
    }

    Molecule mol;
    int n_atoms = 0;
    if (!(in >> n_atoms)) {
        std::cerr << "Error: invalid cpp_input header.\n";
        return 1;
    }

    for (int i = 0; i < n_atoms; ++i) {
        std::string sym;
        int idx;
        double x, y, z;
        in >> sym >> idx >> x >> y >> z;
        mol.add_atom(sym, idx, x, y, z);
    }

    int n_shells = 0;
    in >> n_shells;
    for (int i = 0; i < n_shells; ++i) {
        int atom_idx = 0;
        int l = 0;
        bool is_pure = true;
        int n_prim = 0;
        in >> atom_idx >> l >> is_pure >> n_prim;

        std::vector<double> exps(n_prim);
        std::vector<double> coeffs(n_prim);
        for (int j = 0; j < n_prim; ++j) {
            in >> exps[j] >> coeffs[j];
        }
        mol.add_shell_to_atom(atom_idx, l, is_pure, exps, coeffs);
    }

    int num_dyson_orbs = 1;
    if (!(in >> num_dyson_orbs)) num_dyson_orbs = 1;

    std::vector<Dyson> dysons;
    for (int d = 0; d < num_dyson_orbs; ++d) {
        int n_coeffs = 0;
        double norm_val = 1.0;
        in >> n_coeffs >> norm_val;
        std::vector<double> coeffs(n_coeffs);
        for (int i = 0; i < n_coeffs; ++i) in >> coeffs[i];
        std::string label = (d == 0) ? "Left" : "Right";
        Dyson d_obj(&mol, coeffs, label);
        d_obj.qchem_norm = norm_val;
        dysons.push_back(d_obj);
    }

    double x0, x1, y0, y1, z0, z1, step;
    in >> x0 >> x1 >> y0 >> y1 >> z0 >> z1 >> step;

    for (auto& d_obj : dysons) {
        d_obj.renormalize(x0, x1, y0, y1, z0, z1, step);
    }

    UniformGrid grid(x0, x1, y0, y1, z0, z1, step);

    // Centering logic (match beta_gen)
    const Dyson& L_orig = dysons[0];
    Dyson::Vector3 centroid = L_orig.get_centroid(grid.xmin, grid.xmax, grid.ymin, grid.ymax, grid.zmin, grid.zmax, grid.dx);
    mol.shift_geometry(-centroid.x, -centroid.y, -centroid.z);
    for (auto& d_obj : dysons) {
        d_obj.update_geometry();
    }

    const Dyson& dyson_L = dysons[0];
    const Dyson& dyson_R = (dysons.size() > 1) ? dysons[1] : dysons[0];

    // Dipole axis/center (match beta_gen)
    std::vector<double> dipole_axis = {0.0, 0.0, 1.0};
    std::vector<double> dipole_center = {0.0, 0.0, 0.0};
    if (mol.atoms.size() >= 2) {
        double dx = mol.atoms[1].x - mol.atoms[0].x;
        double dy = mol.atoms[1].y - mol.atoms[0].y;
        double dz = mol.atoms[1].z - mol.atoms[0].z;
        dipole_axis = {dx, dy, dz};
        dipole_center[0] = 0.5 * (mol.atoms[0].x + mol.atoms[1].x);
        dipole_center[1] = 0.5 * (mol.atoms[0].y + mol.atoms[1].y);
        dipole_center[2] = 0.5 * (mol.atoms[0].z + mol.atoms[1].z);
    }

    std::ofstream out(output_file);
    if (!out) {
        std::cerr << "Error: Could not write output file " << output_file << "\n";
        return 1;
    }

    out << "D,n,m,mode_idx,Ix_real,Ix_imag,Iy_real,Iy_imag,Iz_real,Iz_imag,Ix_abs,Iy_abs,Iz_abs\n";

    std::vector<double> energies = {eph};
    double ionization_energy_ev = 0.0;

    for (double D : d_values) {
        auto all_elems = CrossSectionCalculator::ComputePhysicalDipoleMatrixElements(
            dyson_L,
            dyson_R,
            grid,
            energies,
            ionization_energy_ev,
            l_max,
            D,
            dipole_length,
            dipole_axis,
            dipole_center
        );

        if (all_elems.empty()) {
            continue;
        }

        const auto& elems = all_elems[0];
        std::map<std::pair<int, int>, CrossSectionCalculator::DipoleMatrixElement> elem_map;
        for (const auto& elem : elems) {
            if (elem.m < 0) continue;
            elem_map[{elem.m, elem.n_mode}] = elem;
        }

        for (const auto& pair : pairs) {
            int n = pair.n;
            int m = pair.m;
            int mode_idx = n - m;
            if (mode_idx < 0) {
                continue;
            }
            auto it = elem_map.find({m, mode_idx});
            if (it == elem_map.end()) {
                continue;
            }

            const auto& elem = it->second;
            std::complex<double> ix = elem.I_x_L;
            std::complex<double> iy = elem.I_y_L;
            std::complex<double> iz = elem.I_z_L;

            out << D << "," << n << "," << m << "," << mode_idx << ","
                << ix.real() << "," << ix.imag() << ","
                << iy.real() << "," << iy.imag() << ","
                << iz.real() << "," << iz.imag() << ","
                << std::abs(ix) << "," << std::abs(iy) << "," << std::abs(iz)
                << "\n";
        }
    }

    return 0;
}
