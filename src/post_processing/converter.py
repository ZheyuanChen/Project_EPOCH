#!/usr/bin/env python3
import os
import h5py
import argparse
import numpy as np
import sdf_helper as sh
import sys

VARIABLES = {
    "Bz": ("Magnetic_Field_Bz", "Bz.hdf5"),
    "Ex": ("Electric_Field_Ex", "Ex.hdf5"),
    "Ey": ("Electric_Field_Ey", "Ey.hdf5"),
    "Jx": ("Current_Jx", "Jx.hdf5"),
    "Jy": ("Current_Jy", "Jy.hdf5"),
    "ne": ("Derived_Number_Density_Electron", "n_e.hdf5"),
    "n_photon": ("Derived_Number_Density_Photon", "n_photon.hdf5"),
    "poynt_x": ("Derived_Poynting_Flux_x", "poynt_x.hdf5"),
    "xy_Ekin": ("Derived_Average_Particle_Energy_Electron", "x_y_Ekin.hdf5"),
}

def get_args():
    parser = argparse.ArgumentParser(description="EPOCH SDF to HDF5 Converter")
    
    parser.add_argument(
        "--dir",
        dest="input",
        required=True, 
        help="Input directory containing SDF files (e.g., .../sth/sdf_files)"
    )
    parser.add_argument(
        "-o", "--output", 
        help="Output directory. Default: .../sth/hdf5_output"
    )
    parser.add_argument(
        "--max-steps", type=int, default=5000,
        help="Max timestep index to scan (default: 5000)"
    )

    args = parser.parse_args()
    
    # 1. Resolve Absolute Path for Input
    args.input = os.path.abspath(args.input)
    
    # 2. Logic for default output: .../sth/hdf5_output
    if args.output is None:
        # parent_dir is '.../sth', then join with 'hdf5_output'
        parent_dir = os.path.dirname(args.input)
        args.output = os.path.join(parent_dir, "hdf5_output")
    else:
        args.output = os.path.abspath(args.output)
        
    return args

def main():
    args = get_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory {args.input} does not exist.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Find files 0000.sdf, 0001.sdf...
    timesteps = []
    for j in range(args.max_steps):
        if os.path.isfile(os.path.join(args.input, f"{j:04d}.sdf")):
            timesteps.append(j)

    if not timesteps:
        print(f"No SDF files found in {args.input}")
        return

    print(f"Processing {len(timesteps)} files from: {args.input}")
    print(f"Saving results to: {args.output}")

    collected_data = {var: [] for var in VARIABLES}

    for j in timesteps:
        fname = f"{j:04d}.sdf"
        data = sh.getdata(os.path.join(args.input, fname), verbose=False)

        for key, (var_name, _) in VARIABLES.items():
            if hasattr(data, var_name):
                collected_data[key].append(getattr(data, var_name).data)

    for key, (var_name, base_name) in VARIABLES.items():
        if not collected_data[key]:
            continue

        array = np.stack(collected_data[key], axis=0)
        out_path = os.path.join(args.output, base_name)

        with h5py.File(out_path, "w") as f:
            f.create_dataset(key, data=array, chunks=True, compression="gzip")
        
        print(f"  ✔ Created {base_name} with shape {array.shape}")

if __name__ == "__main__":
    main()