#!/usr/bin/env python3

import os
import h5py
import argparse
import numpy as np
import sdf_helper as sh
import sys

# Mapping of EPOCH internal names to our desired HDF5 output filenames
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
    parser = argparse.ArgumentParser(description="EPOCH SDF to HDF5 Converter with Metadata")
    
    parser.add_argument(
        "--dir",
        dest="input",
        required=True, 
        help="Input directory containing SDF files"
    )
    parser.add_argument(
        "-o", "--output", 
        help="Output directory. Default: .../hdf5_output"
    )
    parser.add_argument(
        "--max-steps", type=int, default=5000,
        help="Max timestep index to scan (default: 5000)"
    )

    args = parser.parse_args()
    args.input = os.path.abspath(args.input)
    
    if args.output is None:
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

    # 1. Identify existing SDF files
    timesteps = []
    for j in range(args.max_steps):
        if os.path.isfile(os.path.join(args.input, f"{j:04d}.sdf")):
            timesteps.append(j)

    if not timesteps:
        print(f"No SDF files found in {args.input}")
        return

    print(f"Processing {len(timesteps)} files from: {args.input}")
    print(f"Saving results to: {args.output}")

    # 2. Prepare storage lists
    collected_data = {var: [] for var in VARIABLES}
    times_list = []
    extents_list = []
    grid_dims = None

    # 3. Main Data Extraction Loop (One pass through SDF files)
    for j in timesteps:
        fname = f"{j:04d}.sdf"
        fpath = os.path.join(args.input, fname)
        
        # Load using sdf_helper
        data = sh.getdata(fpath, verbose=False)

        # --- Extract Metadata ---
        times_list.append(data.Header['time'])
        
        # Try to get extents from the main grid
        if hasattr(data, 'Grid_Grid'):
            extents_list.append(data.Grid_Grid.extents)
            if grid_dims is None:
                grid_dims = data.Grid_Grid.dims # Store for the HDF5 attributes
        else:
            # Fallback for 1D or differently named grids
            extents_list.append([0, 0, 0, 0])

        # --- Extract Physical Variables ---
        for key, (var_name, _) in VARIABLES.items():
            if hasattr(data, var_name):
                # .data gets the raw numpy array from the sdf object
                collected_data[key].append(getattr(data, var_name).data)
        
        if j % 10 == 0:
            print(f"  Processed {fname}...")

    # 4. Save Physical Variables to separate HDF5 files
    print("\n--- Saving Variable Stacks ---")
    for key, (var_name, base_name) in VARIABLES.items():
        if not collected_data[key]:
            continue

        array = np.stack(collected_data[key], axis=0)
        out_path = os.path.join(args.output, base_name)

        with h5py.File(out_path, "w") as f:
            f.create_dataset(key, data=array, chunks=True, compression="gzip")
        
        print(f"  ✔ Created {base_name} (Shape: {array.shape})")

    # 5. Save Metadata to a single file
    print("\n--- Saving Metadata ---")
    meta_path = os.path.join(args.output, "metadata.hdf5")
    with h5py.File(meta_path, "w") as f:
        f.create_dataset("times", data=np.array(times_list))
        f.create_dataset("extents", data=np.array(extents_list))
        if grid_dims is not None:
            f.attrs['grid_dims'] = grid_dims
    
    print(f"  ✔ Created metadata.hdf5 with {len(times_list)} frames.")
    print("\nAll conversions complete!")

if __name__ == "__main__":
    main()