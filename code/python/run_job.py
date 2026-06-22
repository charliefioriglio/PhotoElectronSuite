import json
import argparse
import subprocess
import os
import sys
import tempfile
import json
import numpy as np

# Add local directories to sys.path
sys.path.append(os.path.join(os.getcwd(), "code/python"))
from dyson_io import load_qchem

def run_command(cmd, cwd=None):
    """Executes a shell command and checks for errors."""
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

def main():
    # Enforce 1 thread for Python linear algebra libraries to prevent 
    # oversubscription when calling C++ OpenMP binaries
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    parser = argparse.ArgumentParser(description="Unified Driver for Dyson Orbital & Beta Calculations")
    parser.add_argument("config_file", help="Path to JSON configuration file")
    args = parser.parse_args()

    # 1. Parse JSON Configuration
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    qchem_out = config.get("qchem_output")
    if not qchem_out or not os.path.exists(qchem_out):
        print(f"Error: Q-Chem output file '{qchem_out}' not found.")
        sys.exit(1)

    # Global settings
    project_root = os.getcwd() 
    # specific paths to scripts/executables
    dyson_io_script = os.path.join(project_root, "code_safe/python/dyson_io.py")
    beta_gen_exe = os.path.join(project_root, "beta_gen_safe")
    visualize_script = os.path.join(project_root, "code_safe/python/visualize.py")
    beta_plot_script = os.path.join(project_root, "code_safe/python/plot_beta.py")
    
    # 2. Dyson Generation Step
    dyson_cfg = config.get("dyson", {})
    calc_cfg = config.get("calculation", {})
    cpp_input_file = calc_cfg.get("cpp_input_file", "cpp_input.dat")
    cpp_input_dir = os.path.dirname(cpp_input_file)
    if cpp_input_dir:
        os.makedirs(cpp_input_dir, exist_ok=True)
    
    # Check if calculation requires dyson generation args implicitly
    do_calc = calc_cfg.get("do_calculation", False)
    skip_beta = calc_cfg.get("skip_beta_gen", False)
    
    if dyson_cfg.get("do_generation", True):
        print("\n=== Dyson Orbital Generation ===")
        
        # Use the same interpreter as this process so env/package selection is consistent.
        cmd = [sys.executable, dyson_io_script, qchem_out]
        
        # Output Binary
        bin_out = dyson_cfg.get("output_bin", "dyson.bin")
        cmd.extend(["--output", bin_out])
        cmd.extend(["--input-out", cpp_input_file])
        
        # Indices
        indices = dyson_cfg.get("indices", []) # Expect list [0] or [2, 3]
        if not indices:
             # Check legacy single key
             idx = dyson_cfg.get("dyson_index", None)
             if idx is not None: indices = [idx]
        
        if len(indices) == 2:
            cmd.extend(["--dyson-pair", str(indices[0]), str(indices[1])])
        elif len(indices) == 1:
            cmd.extend(["--dyson-index", str(indices[0])])
        else:
            if do_calc and len(indices) == 0:
                 print("Error: Calculation requested but no Dyson indices provided.")
                 sys.exit(1)
            # python script defaults to index 0
        
        # Grid Step
        step = dyson_cfg.get("grid_step", 0.3)
        cmd.extend(["--grid-step", str(step)])
        
        # Padding
        padding = dyson_cfg.get("padding") # Default in dyson_io is 20.0
        if padding is not None:
             cmd.extend(["--padding", str(padding)])
             
        # Relative XS (Vibrational File)
        vib_file = dyson_cfg.get("vib_file")
        if vib_file:
             if os.path.exists(vib_file):
                 cmd.extend(["--vib-file", vib_file])
             else:
                 print(f"Warning: Vibrational file '{vib_file}' not found. Skipping relative XS.")
        
        # If calculation is enabled, we need to pass calculation-specific flags to dyson_io so it generates 'cpp_input.dat' properly.
        if do_calc:
            cmd.append("--xs") # Enable XS/Beta mode in input generation
            
            # IE
            ie = calc_cfg.get("ie")
            if ie is not None:
                cmd.extend(["--ie", str(ie)])
            
            # L-Max
            l_max = calc_cfg.get("l_max", 3)
            cmd.extend(["--lmax", str(l_max)])
            
            # Model Args
            model = calc_cfg.get("model", "point_dipole").lower()
            
            if model == "point_dipole" or model == "physical_dipole":
                 D_list = calc_cfg.get("dipole_list")
                 if D_list:
                     cmd.append("--point-dipole-list")
                     cmd.extend([str(d) for d in D_list])
                 else:
                     D = calc_cfg.get("dipole", 0.0)
                     cmd.extend(["--point-dipole", str(D)])
                     
                 if model == "physical_dipole":
                     a = calc_cfg.get("dipole_length", 0.0)
                     cmd.extend(["--dipole-length", str(a)])
            
            # Pass explicit energies list to dyson_io
            energies = calc_cfg.get("energies", [])
            if energies:
                cmd.append("--energies")
                cmd.extend([str(e) for e in energies])
            
            if skip_beta:
                 out_csv = calc_cfg.get("output_csv")
                 if out_csv:
                     cmd.extend(["--xs-out", out_csv])

        run_command(cmd)

    # 3. Calculation Step (Beta / CSS)
    # Check if we should skip beta_gen (e.g. if we only wanted Relative XS from dyson_io)
    skip_beta = calc_cfg.get("skip_beta_gen", False)
    
    if do_calc and not skip_beta:
        print("\n=== Beta/Cross Section Calculation ===")
        
        # Check for generated C++ input file
        if not os.path.exists(cpp_input_file):
            print(f"Error: '{cpp_input_file}' not found. Ensure Dyson generation step ran successfully.")
            sys.exit(1)
        
        # Model (needed for beta_gen flags regardless of generation step)
        model = calc_cfg.get("model", "point_dipole").lower()
            
        # Prepare Dipole List
        dipole_list = calc_cfg.get("dipole_list")
        if not dipole_list:
            dipole_list = [calc_cfg.get("dipole", 0.0)]
            
        # Common Flags (energies, points, lmax)
        flags = []
        
        # Energies
        energies = calc_cfg.get("energies", [])
        if energies:
            flags.append("--energies")
            flags.extend([str(e) for e in energies])
            
        # Points
        pts = calc_cfg.get("points", 100)
        flags.extend(["--points", str(pts)])
        
        # L-Max (override)
        l_max = calc_cfg.get("l_max")
        if l_max:
             flags.extend(["--lmax", str(l_max)])
             
        # Numeric Averaging
        use_numeric = calc_cfg.get("numeric_averaging", False)
        if calc_cfg.get("averaging") == "numeric":
            use_numeric = True
            
        if use_numeric:
            flags.append("--numeric")

        # Loop over dipoles
        csv_out_base = calc_cfg.get("output_csv", "results.csv")
        
        for D in dipole_list:
            # Command Structure: exe input output [flags]
            current_cmd = [beta_gen_exe, cpp_input_file]
            
            # Output Filename handling
            if len(dipole_list) > 1:
                base, ext = os.path.splitext(csv_out_base)
                out_file = f"{base}_D{D}{ext}"
            else:
                out_file = csv_out_base
                
            current_cmd.append(out_file)
            
            # Add common flags
            current_cmd.extend(flags)
            
            # Model Args
            if model == "pwe":
                current_cmd.append("--pwe")
            elif model == "point_dipole":
                current_cmd.extend(["--point-dipole", str(D)])
            elif model == "physical_dipole":
                a = calc_cfg.get("dipole_length", 0.0)
                current_cmd.extend(["--physical-dipole", str(D), str(a)])
                
            run_command(current_cmd)
        
    # 4. Visualization Step
    vis_cfg = config.get("visualization", {})
    if vis_cfg.get("do_plot", False):
        print("\n=== Visualization ===")
        
        bin_file = dyson_cfg.get("output_bin", "dyson.bin")
        if not os.path.exists(bin_file):
             print(f"Error: Binary file '{bin_file}' not found.")
             sys.exit(1)
             
        vis_cmd = [sys.executable, visualize_script, bin_file]
        
        iso = vis_cfg.get("isovalue", 0.02)
        vis_cmd.extend(["--isovalue", str(iso)])
        
        axis = vis_cfg.get("view_axis")
        if axis is not None:
             vis_cmd.append("--slice")
             vis_cmd.extend(["--axis", str(axis)])
             
        # Save Output
        save_path = vis_cfg.get("output_image")
        if save_path:
             vis_cmd.extend(["--save", save_path])
             print(f"Generating visualization to {save_path}...")
        else:
             print("Launching visualization window...")
        
        # Add atoms information to the visualization command
        try:
            data = load_qchem(qchem_out)
            atoms = data.atoms
            # Center the molecule same as dyson_io does
            coords = np.array([a.center_bohr for a in atoms])
            centroid = np.mean(coords, axis=0)
            atoms_list = []
            for a in atoms:
                atoms_list.append({
                    "symbol": a.symbol,
                    "x": float(a.center_bohr[0] - centroid[0]),
                    "y": float(a.center_bohr[1] - centroid[1]),
                    "z": float(a.center_bohr[2] - centroid[2])
                })
            
            # Use a temporary file to pass the atoms data to avoid shell argument length limits
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
                json.dump(atoms_list, tf)
                temp_atoms_file = tf.name
            
            vis_cmd.extend(["--atoms", temp_atoms_file])
            
            run_command(vis_cmd)
            
            # Clean up temp file
            if os.path.exists(temp_atoms_file):
                os.remove(temp_atoms_file)
        except Exception as e:
            print(f"Warning: Failed to extract atom labels for visualization: {e}")
            run_command(vis_cmd)

    # 5. Beta Plot Step
    beta_plot_cfg = config.get("beta_plot", {})
    if beta_plot_cfg.get("plot", False):
        print("\n=== Beta Parameter Plotting ===")
        
        # Determine which CSV to plot
        csv_file = beta_plot_cfg.get("input_csv")
        if not csv_file:
            # Use the calculation output CSV if not specified
            csv_file = calc_cfg.get("output_csv", "results.csv")
        
        if not os.path.exists(csv_file):
            print(f"Error: Beta CSV file '{csv_file}' not found.")
            sys.exit(1)
        
        plot_cmd = [sys.executable, beta_plot_script, csv_file]
        
        # Output image
        output_image = beta_plot_cfg.get("output_image", "beta_plot.png")
        plot_cmd.extend(["--output", output_image])
        
        # Title (optional)
        title = beta_plot_cfg.get("title")
        if title:
            plot_cmd.extend(["--title", title])
        
        # Show window
        if beta_plot_cfg.get("show", True):
            plot_cmd.append("--show")
        
        run_command(plot_cmd)

if __name__ == "__main__":
    main()
