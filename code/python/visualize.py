import struct
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import argparse
import json
import os

def read_binary_grid(filename):
    with open(filename, "rb") as f:
        nx, ny, nz = struct.unpack("iii", f.read(12))
        x0, y0, z0 = struct.unpack("ddd", f.read(24))
        dx, dy, dz = struct.unpack("ddd", f.read(24))
        
        n_total = nx * ny * nz
        # Check size
        f.seek(0, 2)
        size = f.tell()
        expected = 12 + 24 + 24 + n_total * 8
        if size != expected:
            print(f"Warning: File size mismatch. Expected {expected}, got {size}")
            
        f.seek(60) # Header size
        data = np.frombuffer(f.read(), dtype=np.float64)
        
    return data.reshape((nx, ny, nz), order='C'), (nx, ny, nz), (x0, y0, z0), (dx, dy, dz)

def plot_slice(data, axis, index, title, save_file=None):
    if axis == 0: slice_data = data[index, :, :]
    elif axis == 1: slice_data = data[:, index, :]
    else: slice_data = data[:, :, index]
    
    plt.figure(figsize=(8,6))
    plt.imshow(slice_data.T, origin='lower', cmap='seismic', vmin=-np.max(np.abs(data)), vmax=np.max(np.abs(data)))
    plt.colorbar(label="Amplitude")
    plt.title(title)
    if save_file:
        plt.savefig(save_file)
        print(f"Saved to {save_file}")
    else:
        plt.show()

def plot_isosurface(data, spacing, origin, isovalue=0.02, save_file=None, atoms=None):
    # Data is (nx, ny, nz)
    # Origin is (x0, y0, z0)
    # Spacing is (dx, dy, dz)
    
    verts_pos, faces_pos, _, _ = measure.marching_cubes(data, isovalue, spacing=spacing)
    verts_neg, faces_neg, _, _ = measure.marching_cubes(data, -isovalue, spacing=spacing)
    
    # Adjust vertices by origin
    verts_pos += np.array(origin)
    verts_neg += np.array(origin)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Positive isosurface (Blue)
    mesh_pos = Poly3DCollection(verts_pos[faces_pos], alpha=0.5)
    mesh_pos.set_facecolor('blue')
    ax.add_collection3d(mesh_pos)
    
    # Negative isosurface (Red)
    mesh_neg = Poly3DCollection(verts_neg[faces_neg], alpha=0.5)
    mesh_neg.set_facecolor('red')
    ax.add_collection3d(mesh_neg)
    
    # Auto-scaling
    all_verts = np.vstack([verts_pos, verts_neg])
    
    ax.set_xlim(np.min(all_verts[:,0]), np.max(all_verts[:,0]))
    ax.set_ylim(np.min(all_verts[:,1]), np.max(all_verts[:,1]))
    ax.set_zlim(np.min(all_verts[:,2]), np.max(all_verts[:,2]))
    
    ax.set_xlabel("X (Bohr)")
    ax.set_ylabel("Y (Bohr)")
    ax.set_zlabel("Z (Bohr)")
    
    # Plot atoms if provided
    if atoms:
        for atom in atoms:
            x, y, z = atom["x"], atom["y"], atom["z"]
            symbol = atom["symbol"]
            # Draw a small sphere or just a point
            ax.scatter([x], [y], [z], color='black', s=50)
            ax.text(x, y, z, symbol, color='black', fontsize=12, fontweight='bold')

    ax.set_title(f"Dyson Orbital Isosurface (+/- {isovalue})")
    
    if save_file:
        plt.savefig(save_file)
        print(f"Saved to {save_file}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Dyson Orbital Grid")
    parser.add_argument("file", help="Binary grid file")
    parser.add_argument("--slice", action="store_true", help="Show slice instead of 3D")
    parser.add_argument("--axis", type=int, default=2, help="Slice axis (0=X, 1=Y, 2=Z)")
    parser.add_argument("--index", type=int, default=-1, help="Slice index (default: middle)")
    parser.add_argument("--isovalue", type=float, default=0.02, help="Isosurface value (default: 0.02)")
    parser.add_argument("--save", type=str, help="Save plot to file instead of showing")
    parser.add_argument("--atoms", type=str, help="JSON file containing atom positions and symbols")
    
    args = parser.parse_args()
    
    atoms_data = None
    if args.atoms:
        if os.path.exists(args.atoms):
            with open(args.atoms, "r") as f:
                atoms_data = json.load(f)
        else:
            try:
                atoms_data = json.loads(args.atoms)
            except:
                print(f"Warning: Could not parse atoms data: {args.atoms}")
    
    print(f"Reading {args.file}...")
    data, dims, origin, spacing = read_binary_grid(args.file)
    print(f"Grid: {dims}")
    print(f"Range: [{np.min(data)}, {np.max(data)}]")
    
    if args.slice:
        idx = args.index if args.index >= 0 else dims[args.axis] // 2
        plot_slice(data, args.axis, idx, f"Slice Axis {args.axis} Index {idx}", args.save)
    else:
        print(f"Generating 3D isosurface at +/- {args.isovalue}...")
        try:
            plot_isosurface(data, spacing, origin, args.isovalue, args.save, atoms_data)
        except Exception as e:
            print(f"Error plotting isosurface: {e}")
            print("Try adjusting the isovalue or grid step.")
