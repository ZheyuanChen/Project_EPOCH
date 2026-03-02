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
    "ekbar_electron": ("Derived_Average_Particle_Energy_Electron", "ekbar_electron.hdf5"),
    "ekbar_ion": ("Derived_Average_Particle_Energy_Ion", "ekbar_ion.hdf5"),
    "ekbar_photon": ("Derived_Average_Particle_Energy_Photon", "ekbar_photon.hdf5"),
    "ekbar_positron": ("Derived_Average_Particle_Energy_Positron", "ekbar_positron.hdf5"),
    "dist_electron": ("dist_fn_spatial_energy_Electron", "dist_electron.hdf5"),
    "dist_photon": ("dist_fn_spatial_energy_Photon", "dist_photon.hdf5"),
    "dist_ion": ("dist_fn_spatial_energy_Ion", "dist_ion.hdf5"),
    "dist_positron": ("dist_fn_spatial_energy_Positron", "dist_positron.hdf5"),
}

def get_args():
    parser = argparse.ArgumentParser(description="EPOCH SDF to HDF5 Converter")
    parser.add_argument("--dir", dest="input", required=True, help="Input directory (SDF files)")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max timestep to scan")
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
        print(f"Error: {args.input} not found."); sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Find available SDF files
    timesteps = [j for j in range(args.max_steps) if os.path.isfile(os.path.join(args.input, f"{j:04d}.sdf"))]
    if not timesteps:
        print("No SDF files found."); return

    print(f"Processing {len(timesteps)} files...")
    
    collected_data = {var: [] for var in VARIABLES}
    times, extents = [], []
    scalars = {"laser_en": [], "abs_frac": [], "part_en": [], "field_en": []}
    species_energies = {"electron": [], "ion": [], "photon": [], "positron": []}
    
    grid_dims = None
    dist_extents = None 

    for j in timesteps:
        fpath = os.path.join(args.input, f"{j:04d}.sdf")
        data = sh.getdata(fpath, verbose=False)

        times.append(data.Header['time'])
        
        # Grid Extents
        if hasattr(data, 'Grid_Grid'):
            extents.append(data.Grid_Grid.extents)
            if grid_dims is None: grid_dims = data.Grid_Grid.dims
        
        # Distribution Metadata (Energy/Angle axes)
        if dist_extents is None:
            for key in VARIABLES:
                if "dist_" in key:
                    sdf_var_name = VARIABLES[key][0]
                    # EPOCH prepends 'Grid_' to the distribution variable name
                    grid_obj_name = f"Grid_{sdf_var_name}"
                    if hasattr(data, grid_obj_name):
                        dist_extents = getattr(data, grid_obj_name).extents
                        print(f"  Found Dist Extents from {grid_obj_name}")
                        break

        # Scalars
        if hasattr(data, 'Absorption_Total_Laser_Energy_Injected__J_'):
            scalars["laser_en"].append(data.Absorption_Total_Laser_Energy_Injected__J_.data)
            scalars["abs_frac"].append(data.Absorption_Fraction_of_Laser_Energy_Absorbed____.data)
        
        if hasattr(data, 'Total_Particle_Energy_in_Simulation__J_'):
            scalars["part_en"].append(data.Total_Particle_Energy_in_Simulation__J_.data)
            scalars["field_en"].append(data.Total_Field_Energy_in_Simulation__J_.data)

        # Species Scalars (matching viewer keys: total_energy_electron, etc.)
        for sp in species_energies.keys():
            sdf_name = f"Total_Particle_Energy_{sp.capitalize()}__J_"
            if hasattr(data, sdf_name):
                species_energies[sp].append(getattr(data, sdf_name).data)

        # Physical Variables (Stacks)
        for key, (sdf_name, _) in VARIABLES.items():
            if hasattr(data, sdf_name):
                collected_data[key].append(getattr(data, sdf_name).data)
        
        if j % 20 == 0: print(f"  Step {j} processed...")

    # Save Variable Stacks
    for key, (sdf_name, filename) in VARIABLES.items():
        if collected_data[key]:
            out_path = os.path.join(args.output, filename)
            with h5py.File(out_path, "w") as f:
                # Compression is key for large 4D distributions
                f.create_dataset(key, data=np.stack(collected_data[key]), chunks=True, compression="gzip", compression_opts=4)
            print(f"  ✔ Saved {filename}")

    # Save Metadata
    meta_path = os.path.join(args.output, "metadata.hdf5")
    with h5py.File(meta_path, "w") as f:
        f.create_dataset("times", data=np.array(times))
        f.create_dataset("extents", data=np.array(extents))
        
        if scalars["laser_en"]:
            f.create_dataset("laser_en_total", data=np.array(scalars["laser_en"]).flatten())
            f.create_dataset("abs_frac", data=np.array(scalars["abs_frac"]).flatten())
        
        f.create_dataset("total_energy_particle", data=np.array(scalars["part_en"]).flatten())
        f.create_dataset("total_energy_field", data=np.array(scalars["field_en"]).flatten())
        
        for sp, en_list in species_energies.items():
            if en_list:
                f.create_dataset(f"total_energy_{sp}", data=np.array(en_list).flatten())
        
        if grid_dims is not None: f.attrs['grid_dims'] = grid_dims
        if dist_extents is not None: f.create_dataset("dist_extents", data=np.array(dist_extents))
    
    print(f"\nFinished! All files saved to: {args.output}")

if __name__ == "__main__":
    main()