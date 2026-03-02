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
    "ekbar": ("Derived_Average_Particle_Energy", "ekbar.hdf5"),
    "ekbar_electron": ("Derived_Average_Particle_Energy_Electron", "ekbar_electron.hdf5"),
    "ekbar_ion": ("Derived_Average_Particle_Energy_Ion", "ekbar_ion.hdf5"),
    "ekbar_photon": ("Derived_Average_Particle_Energy_Photon", "ekbar_photon.hdf5"),
    "ekbar_positron": ("Derived_Average_Particle_Energy_Positron", "ekbar_positron.hdf5"),
}

def get_args():
    parser = argparse.ArgumentParser(description="EPOCH SDF to HDF5 Converter with Metadata")
    parser.add_argument("--dir", dest="input", required=True, help="Input directory containing SDF files")
    parser.add_argument("-o", "--output", help="Output directory. Default: .../hdf5_output")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max timestep index to scan")
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

    timesteps = []
    for j in range(args.max_steps):
        if os.path.isfile(os.path.join(args.input, f"{j:04d}.sdf")):
            timesteps.append(j)

    if not timesteps:
        print(f"No SDF files found in {args.input}")
        return

    print(f"Processing {len(timesteps)} files from: {args.input}")
    
    collected_data = {var: [] for var in VARIABLES}
    times_list = []
    extents_list = []
    
    # Scalar Storage
    laser_en_list = []
    abs_frac_list = []
    part_en_list = []
    field_en_list = []
    
    # Species-specific energy storage
    species_energies = {
        "Electron": [], "Ion": [], "Photon": [], "Positron": []
    }
    
    grid_dims = None

    for j in timesteps:
        fname = f"{j:04d}.sdf"
        fpath = os.path.join(args.input, fname)
        data = sh.getdata(fpath, verbose=False)

        times_list.append(data.Header['time'])
        
        if hasattr(data, 'Grid_Grid'):
            extents_list.append(data.Grid_Grid.extents)
            if grid_dims is None:
                grid_dims = data.Grid_Grid.dims
        else:
            extents_list.append([0, 0, 0, 0])

        # Extract Scalars
        if hasattr(data, 'Absorption_Total_Laser_Energy_Injected__J_'):
            laser_en_list.append(data.Absorption_Total_Laser_Energy_Injected__J_.data)
            abs_frac_list.append(data.Absorption_Fraction_of_Laser_Energy_Absorbed____.data)
        
        if hasattr(data, 'Total_Particle_Energy_in_Simulation__J_'):
            part_en_list.append(data.Total_Particle_Energy_in_Simulation__J_.data)
            field_en_list.append(data.Total_Field_Energy_in_Simulation__J_.data)

        # Extract Species Energies
        for species in species_energies.keys():
            var_name = f"Total_Particle_Energy_{species}__J_"
            if hasattr(data, var_name):
                species_energies[species].append(getattr(data, var_name).data)

        # Physical Variables
        for key, (var_name, _) in VARIABLES.items():
            if hasattr(data, var_name):
                collected_data[key].append(getattr(data, var_name).data)
        
        if j % 10 == 0:
            print(f"  Processed {fname}...")

    # Save Physical Variables
    print("\n--- Saving Variable Stacks ---")
    for key, (var_name, base_name) in VARIABLES.items():
        if not collected_data[key]:
            continue
        array = np.stack(collected_data[key], axis=0)
        out_path = os.path.join(args.output, base_name)
        with h5py.File(out_path, "w") as f:
            f.create_dataset(key, data=array, chunks=True, compression="gzip")
        print(f"  ✔ Created {base_name}")

    # Save Metadata
    print("\n--- Saving Metadata ---")
    meta_path = os.path.join(args.output, "metadata.hdf5")
    with h5py.File(meta_path, "w") as f:
        f.create_dataset("times", data=np.array(times_list))
        f.create_dataset("extents", data=np.array(extents_list))
        
        if laser_en_list:
            f.create_dataset("laser_en_total", data=np.array(laser_en_list).flatten())
            f.create_dataset("abs_frac", data=np.array(abs_frac_list).flatten())
        if part_en_list:
            f.create_dataset("total_energy_particle", data=np.array(part_en_list).flatten())
            f.create_dataset("total_energy_field", data=np.array(field_en_list).flatten())
        
        # Save Species Energies
        for species, energy_data in species_energies.items():
            if energy_data:
                f.create_dataset(f"total_energy_{species.lower()}", data=np.array(energy_data).flatten())
        
        if grid_dims is not None:
            f.attrs['grid_dims'] = grid_dims
    
    print(f"  ✔ Created metadata.hdf5 with scalars and {len(times_list)} frames.")

if __name__ == "__main__":
    main()