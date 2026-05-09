#ifndef CROSS_SECTION_H
#define CROSS_SECTION_H

#include "dyson.h"
#include "grid.h"
#include <vector>
#include <string>
#include <complex>

struct VibState {
    double energy_bind;
    double fcf_overlap; // Amplitude, squared for prob
};

struct RelativeXSResult {
    std::vector<double> photon_energies;
    std::vector<double> total_cross_section; // Weighted sum
    std::vector<std::vector<double>> channel_fractions; // [energy_idx][channel_idx]
};

// Computes total cross section for a given Dyson orbital and energy range
class CrossSectionCalculator {
public:
    // Compute Relative Cross Sections (Total and per-channel)
    // Now supports Physical Dipole if dipole_length > 0.
    static RelativeXSResult ComputeRelativeCrossSections(
        const Dyson& dyson_L,
        const Dyson& dyson_R,
        const UniformGrid& grid,
        const std::vector<double>& photon_energies_ev,
        const std::vector<VibState>& vib_states,
        int l_max,
        double dipole_magnitude, // Dipole Strength. (Physical D = 0.5 * Strength).
        bool use_point_dipole,
        double dipole_length = 0.0 // 'a'. If > 0, uses Physical Dipole
    );
    // Calculates sigma for a list of PHOTON energies (eV)
    static std::vector<double> ComputeTotalCrossSection(
        const Dyson& dyson_L,
        const Dyson& dyson_R,
        const UniformGrid& grid,
        const std::vector<double>& photon_energies_ev,
        double ionization_energy_ev,
        int l_max
    );

    // Calculates sigma using Point Dipole model
    static std::vector<double> ComputePointDipoleCrossSection(
        const Dyson& dyson_L,
        const Dyson& dyson_R,
        const UniformGrid& grid,
        const std::vector<double>& photon_energies_ev,
        double ionization_energy_ev,
        int l_max,
        double dipole_magnitude
    );

    // Calculate sigma using Physical Dipole model
    static std::vector<double> ComputePhysicalDipoleCrossSection(
        const Dyson& dyson_L,
        const Dyson& dyson_R,
        const UniformGrid& grid,
        const std::vector<double>& photon_energies_ev,
        double ionization_energy_ev,
        int l_max,
        double dipole_magnitude,
        double dipole_length,
        const std::vector<double>& dipole_axis = {0.0, 0.0, 1.0},
        const std::vector<double>& dipole_center = {0.0, 0.0, 0.0}
    );

    // For Beta Calculation: Return Matrix Elements in Body Frame
    struct DipoleMatrixElement {
        double E_ph;
        int m;
        int n_mode;
        std::complex<double> nu;
        std::complex<double> I_x_L;
        std::complex<double> I_y_L;
        std::complex<double> I_z_L;
        std::complex<double> I_x_R;
        std::complex<double> I_y_R;
        std::complex<double> I_z_R;
    };
    
    static std::vector<std::vector<DipoleMatrixElement>> ComputePhysicalDipoleMatrixElements(
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
    );

private:

};

#endif
