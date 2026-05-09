#!/usr/bin/env python3
"""Run extended Cnm integral sweeps for AgF and CuO_pVTZ.

This script tabulates Cnm overlap integrals at several electron kinetic energies (eKEs).
It also refreshes the Dyson orbital binary grid files for each system.

Targets:
- AgF (Test Dyson Orbitals/AgF.out)
- CuO_pVTZ (Test Dyson Orbitals/CuO_pVTZ.out)

eKEs: 0.01, 0.1, 0.2, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5
"""

import subprocess
import sys
from pathlib import Path
import numpy as np

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
CODE_PY = ROOT / "code" / "python"
sys.path.insert(0, str(CODE_PY))

try:
    import dyson_io
except ImportError:
    print("Error: Could not import dyson_io. Ensure code/python is in the path.")
    sys.exit(1)

# Configuration
SYSTEMS = [
    {"label": "AgF", "out": "Test Dyson Orbitals/AgF.out", "indices": [0]},
    {"label": "CuO_pVTZ", "out": "Test Dyson Orbitals/CuO_pVTZ.out", "indices": [0]},
]

# Parameters requested by user
EKES = [0.01, 0.1, 0.2, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
D_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] # Default from run_cnm_integrals.py
PAIRS = ["0,0", "1,1", "1,0", "2,2", "2,1", "2,0"] # Default from run_cnm_integrals.py

# Executables
DIPOLE_EXE = ROOT / "dipole_integrals"
DYSON_EXE = ROOT / "dyson_gen"
RESULTS_DIR = ROOT / "results" / "cnm_integrals"

def run_sweeps():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DIPOLE_EXE.exists():
        print(f"Error: {DIPOLE_EXE} not found. Build it with 'make dipole_integrals'.")
        return
    if not DYSON_EXE.exists():
        print(f"Error: {DYSON_EXE} not found. Build it with 'make dyson_gen'.")
        return

    for sys_info in SYSTEMS:
        label = sys_info["label"]
        qchem_path = ROOT / sys_info["out"]
        indices = sys_info["indices"]
        
        print(f"\n{'='*40}")
        print(f" Processing System: {label}")
        print(f"{'='*40}")
        
        if not qchem_path.exists():
            print(f"Skipping {label}: {qchem_path} not found.")
            continue

        # 1. Load data and prepare cpp_input.dat
        print(f"Loading Q-Chem output: {qchem_path.name}...")
        data = dyson_io.load_qchem(str(qchem_path))
        
        # Centering and Grid Logic
        coords = np.array([a.center_bohr for a in data.atoms])
        centroid = np.mean(coords, axis=0)
        for a in data.atoms:
            a.center_bohr = [c - o for c, o in zip(a.center_bohr, centroid)]
        
        coords = np.array([a.center_bohr for a in data.atoms])
        padding = 20.0
        min_c = coords.min(axis=0) - padding
        max_c = coords.max(axis=0) + padding
        grid = {
            "x0": min_c[0], "x1": max_c[0],
            "y0": min_c[1], "y1": max_c[1],
            "z0": min_c[2], "z1": max_c[2],
            "step": 0.2,
        }
        
        cpp_input = RESULTS_DIR / f"{label.lower()}_cpp_input.dat"
        dyson_io.write_cpp_input(data, indices, grid, str(cpp_input))
        print(f"Prepared C++ input: {cpp_input.relative_to(ROOT)}")
        
        # 2. Refresh the bin (run dyson_gen)
        # This updates the .bin file for the primary orbital
        bin_out = ROOT / f"{label.lower()}_dyson.bin"
        print(f"Refreshing binary grid: {bin_out.name}...")
        subprocess.run([str(DYSON_EXE), str(cpp_input), str(bin_out)], check=True)
        
        # 3. Sweep over eKEs
        for eke in EKES:
            csv_out = RESULTS_DIR / f"{label.lower()}_cnm_eke_{eke}.csv"
            print(f"  Calculating eKE = {eke} eV...")
            
            cmd = [
                str(DIPOLE_EXE),
                str(cpp_input),
                str(csv_out),
                "--eph", f"{eke}",
                "--lmax", "10",
                "--a", "1.5",
                "--d-values"
            ]
            cmd.extend(str(d) for d in D_VALUES)
            cmd.append("--pairs")
            cmd.extend(PAIRS)
            
            subprocess.run(cmd, check=True)
            # print(f"    -> Wrote {csv_out.name}")

    print(f"\nAll sweeps complete. Results are in {RESULTS_DIR.relative_to(ROOT)}/")

if __name__ == "__main__":
    run_sweeps()
