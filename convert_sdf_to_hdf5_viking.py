#!/usr/bin/env python3
"""
Convert sdf files to hdf5 format for visualisation. 
This script is to be used on Viking so I didn't add CLI command line arguments, but this can also be run locally.
The important part is that the input and output directories are set correctly in the USER SETTINGS section.

"""
import os
import h5py
import numpy as np
import sdf_helper as sh

# ============================================================
# USER SETTINGS
# ============================================================

# Directory containing SDF files
#INPUT_DIR = os.path.abspath(os.path.dirname(__file__))
INPUT_DIR = "/home/pnd531/Desktop/Project_EPOCH/test_hdf5/sdf_files"


# Directory to write HDF5 files
#OUTPUT_DIR = os.path.join(INPUT_DIR, "hdf5_output")
OUTPUT_DIR = "/home/pnd531/Desktop/Project_EPOCH/test_hdf5/hdf5_output"

# Maximum timestep index to scan. i.e. number of SDF files to look for. Typically shouldn't have more than 200 files.
MAX_STEPS = 5000





VARIABLES = {
    "Bz": ("Magnetic_Field_Bz", "Bz.hdf5"),
    "Ex": ("Electric_Field_Ex", "Ex.hdf5"),
    "Ey": ("Electric_Field_Ey", "Ey.hdf5"),
    "Jx": ("Current_Jx", "Jx.hdf5"),
    "Jy": ("Current_Jy", "Jy.hdf5"),
    "ne": ("Derived_Number_Density_Electron", "n_e.hdf5"),
    "n_photon": ("Derived_Number_Density_Photon", "n_photon.hdf5"),
    #"xy_Ekin": ("Derived_Electron_Kinetic_Energy", "x_y_Ekin.hdf5"),
    "poynt_x": ("Derived_Poynting_Flux_x", "poynt_x.hdf5"),
    "xy_Ekin": ("Derived_Average_Particle_Energy_Electron", "x_y_Ekin.hdf5"),
    
    # future additions go here
}

# storage for variables that actually exist


# ============================================================
# Utilities
# ============================================================

def next_available_filename(directory, base_name):
    """
    If base_name exists, returns base_1.hdf5, base_2.hdf5, ...
    In case the output file already exists, to avoid overwriting.
    """
    name, ext = os.path.splitext(base_name)
    candidate = base_name
    i = 1
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{name}_{i}{ext}"
        i += 1
    return candidate

def probe_variable(data, var_name, sdf_filename, verbose=True):
    """
    Safely extract a variable from an SDF BlockList.

    Parameters
    ----------
    data : sdf.BlockList
        Loaded SDF data object
    var_name : str
        Name of the variable to extract
    sdf_filename : str
        For logging purposes
    verbose : bool

    Returns
    -------
    np.ndarray or None
        Variable data if present, otherwise None
    """
    if hasattr(data, var_name):
        return getattr(data, var_name).data
    else:
        if verbose:
            print(f"[WARN] {var_name} not found in {sdf_filename}, skipping.")
        return None


def collect_sdf_indices(directory, max_steps):
    """Return sorted list of available SDF timestep indices."""
    indices = []
    for j in range(max_steps):
        if os.path.isfile(os.path.join(directory, f"{j:04d}.sdf")):
            indices.append(j)
    return indices


def write_hdf5(output_path, dataset_name, data_array):
    """Write full 3D array (t, x, y) to HDF5."""
    with h5py.File(output_path, "w") as f:
        f.create_dataset(dataset_name, data=data_array, chunks=True)
    print(f"  → wrote {output_path}")

# ============================================================
# Main conversion
# ============================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timesteps = collect_sdf_indices(INPUT_DIR, MAX_STEPS)
    if not timesteps:
        raise RuntimeError("No SDF files found.")

    print(f"Found {len(timesteps)} SDF files")

    # Storage dict: one list per variable
    collected_data = {var: [] for var in VARIABLES}


    # --------------------------------------------------------
    # Read SDF files
    # --------------------------------------------------------
    for j in timesteps:
        fname = f"{j:04d}.sdf"
        print(f"Reading {fname}...")

        data = sh.getdata(os.path.join(INPUT_DIR, fname), verbose = False)

        for key, (var_name,_) in VARIABLES.items():
            var_data = probe_variable(data, var_name, fname)
            if var_data is not None:
                collected_data[key].append(var_data)


    # --------------------------------------------------------
    # Write output files (only if data exists)
    # --------------------------------------------------------
    print("Writing HDF5 files:")

    for key, (var_name, base_name) in VARIABLES.items():
        if not collected_data[key]:
            print(f"  ⚠ Skipping {var_name} (not found in any SDF files)")
            continue

        array = np.stack(collected_data[key], axis=0)

        fname = next_available_filename(OUTPUT_DIR, base_name)
        write_hdf5(
            os.path.join(OUTPUT_DIR, fname),
            key,
            array
        )

        print(f"  ✔ Wrote {fname}")

    print("Conversion complete.")



# ============================================================

if __name__ == "__main__":
    main()
