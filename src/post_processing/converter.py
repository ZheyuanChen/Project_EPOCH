import os
import h5py
import argparse
import numpy as np
import sdf_helper as sh
import sys

# Mapping of EPOCH internal names to our desired HDF5 output filenames
VARIABLES = {
    "Bx": ("Magnetic_Field_Bx", "Bx.hdf5"),
    "By": ("Magnetic_Field_By", "By.hdf5"),
    "Bz": ("Magnetic_Field_Bz", "Bz.hdf5"),
    "Ex": ("Electric_Field_Ex", "Ex.hdf5"),
    "Ey": ("Electric_Field_Ey", "Ey.hdf5"),
    "Ez": ("Electric_Field_Ez", "Ez.hdf5"),
    "Jx": ("Current_Jx", "Jx.hdf5"),
    "Jy": ("Current_Jy", "Jy.hdf5"),
    "Jz": ("Current_Jz", "Jz.hdf5"),
    "ne": ("Derived_Number_Density_Electron", "n_e.hdf5"),
    "n_photon": ("Derived_Number_Density_Photon", "n_photon.hdf5"),
    "poynt_x": ("Derived_Poynting_Flux_x", "poynt_x.hdf5"),
    "poynt_y": ("Derived_Poynting_Flux_y", "poynt_y.hdf5"),
    "poynt_z": ("Derived_Poynting_Flux_z", "poynt_z.hdf5"),
    "ekbar": ("Derived_Average_Particle_Energy", "ekbar.hdf5"),
    "ekbar_electron": (
        "Derived_Average_Particle_Energy_Electron",
        "ekbar_electron.hdf5",
    ),
    "ekbar_ion": ("Derived_Average_Particle_Energy_Ion", "ekbar_ion.hdf5"),
    "ekbar_photon": ("Derived_Average_Particle_Energy_Photon", "ekbar_photon.hdf5"),
    "ekbar_positron": (
        "Derived_Average_Particle_Energy_Positron",
        "ekbar_positron.hdf5",
    ),
    # below requires a distribution function to be named as "spatial_energy" in the input.deck which contains dir_x, dir_y, and dir_en. 
    "dist_electron": ("dist_fn_spatial_energy_Electron", "dist_electron.hdf5"),
    "dist_photon": ("dist_fn_spatial_energy_Photon", "dist_photon.hdf5"),
    "dist_ion": ("dist_fn_spatial_energy_Ion", "dist_ion.hdf5"),
    "dist_positron": ("dist_fn_spatial_energy_Positron", "dist_positron.hdf5"),
    
    # --- NEW: xy_Angle-Energy Distributions ---
    # This requires a distribution function named as "xy_energy" in the input.deck which contains dir_xy_angle and dir_en.
    "dist_xy_en_electron": ("dist_fn_xy_energy_Electron", "dist_fn_xy_energy_Electron.hdf5"),
    "dist_xy_en_photon": ("dist_fn_xy_energy_Photon", "dist_fn_xy_energy_Photon.hdf5"),
    "dist_xy_en_ion": ("dist_fn_xy_energy_Ion", "dist_fn_xy_energy_Ion.hdf5"),
    "dist_xy_en_positron": ("dist_fn_xy_energy_Positron", "dist_fn_xy_energy_Positron.hdf5"),
}


def get_args():
    """
    Parses command line arguments for the converter. The most important one is the input directory containing the SDF files.
    The output directory is optional and defaults to a subfolder named "hdf5_output" in the same location as the input.
    The max-steps argument allows users to limit how many timesteps to scan for SDF files, which can speed up processing if they know their data range.
    """
    parser = argparse.ArgumentParser(description="EPOCH SDF to HDF5 Converter")
    parser.add_argument(
        "--dir", dest="input", required=True, help="Input directory (SDF files)"
    )
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument(
        "--max-steps", type=int, default=10000, help="Max timestep to scan"
    )
    args = parser.parse_args()
    args.input = os.path.abspath(args.input)
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "hdf5_output")
    else:
        args.output = os.path.abspath(args.output)
    return args


def main():
    args = get_args()
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Find available SDF files
    timesteps = [
        j
        for j in range(args.max_steps)
        if os.path.isfile(os.path.join(args.input, f"{j:04d}.sdf"))
    ]
    if not timesteps:
        print("No SDF files found.")
        return

    print(f"Processing {len(timesteps)} files...")

    collected_data = {var: [] for var in VARIABLES}
    times, extents = [], []
    scalars = {
        "laser_en_total": [],
        "abs_frac": [],
        "total_energy_particle": [],
        "total_energy_field": [],
    }
    species_energies = {"electron": [], "ion": [], "photon": [], "positron": []}

    grid_dims = None
    dist_extents = None
    dist_xy_extents = None # NEW: For Angle-Energy extents
    dist_xy_dims = None    # NEW: For Angle-Energy resolutions

    for j in timesteps:
        fpath = os.path.join(args.input, f"{j:04d}.sdf")
        data = sh.getdata(fpath, verbose=False)

        times.append(data.Header["time"])

        # Grid Extents
        if hasattr(data, "Grid_Grid"):
            extents.append(data.Grid_Grid.extents)
            if grid_dims is None:
                grid_dims = data.Grid_Grid.dims

        # Distribution Metadata (Energy axes) - ORIGINAL UNTOUCHED
        if dist_extents is None:
            for key in VARIABLES:
                if "dist_" in key:
                    sdf_var_name = VARIABLES[key][0]
                    grid_obj_name = sdf_var_name.replace("dist_fn_", "Grid_")

                    if hasattr(data, grid_obj_name):
                        full_extents = getattr(data, grid_obj_name).extents
                        # sdf_helper orders 3D extents as [x_min, y_min, E_min, x_max, y_max, E_max]
                        if len(full_extents) >= 6:
                            dist_extents = [full_extents[2], full_extents[5]]
                        else:
                            dist_extents = full_extents
                        print(f"  Found Original Energy Extents: {dist_extents}")
                        break

        # NEW: Angle-Energy Metadata Extraction
        if dist_xy_extents is None:
            for sp in ["Electron", "Photon", "Ion", "Positron"]:
                grid_obj_name = f"Grid_xy_energy_{sp}"
                if hasattr(data, grid_obj_name):
                    grid_obj = getattr(data, grid_obj_name)
                    # For a 2D grid, extents are [dim1_min, dim2_min, dim1_max, dim2_max]
                    dist_xy_extents = grid_obj.extents
                    # Dims are [resolution1, resolution2]
                    dist_xy_dims = grid_obj.dims
                    print(f"  Found Angle-Energy Extents: {dist_xy_extents} | Dims: {dist_xy_dims}")
                    break

        # Scalars (Extracting as pure floats using .item())
        if hasattr(data, "Absorption_Total_Laser_Energy_Injected__J_"):
            scalars["laser_en_total"].append(
                np.array(data.Absorption_Total_Laser_Energy_Injected__J_.data).item()
            )
            scalars["abs_frac"].append(
                np.array(
                    data.Absorption_Fraction_of_Laser_Energy_Absorbed____.data
                ).item()
            )

        if hasattr(data, "Total_Particle_Energy_in_Simulation__J_"):
            scalars["total_energy_particle"].append(
                np.array(data.Total_Particle_Energy_in_Simulation__J_.data).item()
            )
            scalars["total_energy_field"].append(
                np.array(data.Total_Field_Energy_in_Simulation__J_.data).item()
            )

        # Species Scalars
        for sp in species_energies.keys():
            sdf_name = f"Total_Particle_Energy_{sp.capitalize()}__J_"
            if hasattr(data, sdf_name):
                species_energies[sp].append(
                    np.array(getattr(data, sdf_name).data).item()
                )

        # Physical Variables (REVERTED: No transpose needed)
        for key, (sdf_name, _) in VARIABLES.items():
            if hasattr(data, sdf_name):
                collected_data[key].append(getattr(data, sdf_name).data)

        if j % 20 == 0:
            print(f"  Step {j} processed...")

    # Save Variable Stacks
    for key, (sdf_name, filename) in VARIABLES.items():
        if collected_data[key]:
            out_path = os.path.join(args.output, filename)
            with h5py.File(out_path, "w") as f:
                # Compression remains for large 4D files
                f.create_dataset(
                    key,
                    data=np.stack(collected_data[key]),
                    chunks=True,
                    compression="gzip",
                    compression_opts=4,
                )
            print(f"  ✔ Saved {filename}")

    # Save Metadata with keys matching the DianaInteractive viewer
    meta_path = os.path.join(args.output, "metadata.hdf5")
    with h5py.File(meta_path, "w") as f:
        f.create_dataset("times", data=np.array(times))
        f.create_dataset("extents", data=np.array(extents))

        for k, v in scalars.items():
            if v:
                f.create_dataset(k, data=np.array(v))

        for sp, en_list in species_energies.items():
            if en_list:
                f.create_dataset(f"total_energy_{sp}", data=np.array(en_list))

        if grid_dims is not None:
            f.attrs["grid_dims"] = grid_dims
            
        # Old distribution metadata
        if dist_extents is not None:
            f.create_dataset("dist_extents", data=np.array(dist_extents))
            
        # NEW: Angle-Energy metadata
        if dist_xy_extents is not None:
            f.create_dataset("dist_xy_energy_extents", data=np.array(dist_xy_extents))
        if dist_xy_dims is not None:
            f.attrs["dist_xy_energy_dims"] = dist_xy_dims

    print(f"\nFinished! All files saved to: {args.output}")


if __name__ == "__main__":
    main()