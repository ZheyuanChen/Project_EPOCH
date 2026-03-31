import argparse
import os
import sdf_xarray as sdfxr
import xarray as xr
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Define the alias mapping dynamically
ALIASES = {}
for comp in ['x', 'y', 'z']:
    ALIASES[f'E{comp}'] = f'Electric_Field_E{comp}'
    ALIASES[f'B{comp}'] = f'Magnetic_Field_B{comp}'
    ALIASES[f'J{comp}'] = f'Current_J{comp}'
    ALIASES[f'S{comp}'] = f'Derived_Poynting_Flux_{comp}'
    
for species in ['Electron', 'Ion', 'Photon', 'Positron']:
    ALIASES[f'n_{species}'] = f'Derived_Number_Density_{species}'

def main():
    # 2. Set up the argument parser
    parser = argparse.ArgumentParser(description="Generate GIF animations from EPOCH SDF files.")
    parser.add_argument("input_dir", type=str, 
                        help="Path to the directory containing .sdf files")
    parser.add_argument("-v", "--vars", nargs="*", type=str, 
                        help="Variables to plot (use aliases like Ex, Bz, n_Electron). "
                             "If omitted, plots all available variables.")
    args = parser.parse_args()

    # Construct the search path for SDF files
    sdf_path = os.path.join(args.input_dir, "*.sdf")
    print(f"Loading data from: {sdf_path}")
    
    ds = sdfxr.open_mfdataset(sdf_path)
    
    # 3. Determine which variables to plot
    vars_to_plot = []
    if args.vars:
        # Map requested aliases to official EPOCH names
        for v in args.vars:
            official_name = ALIASES.get(v, v) # Fallback to input if not in alias dict
            if official_name in ds.data_vars:
                vars_to_plot.append(official_name)
            else:
                print(f"Warning: '{official_name}' not found in the dataset. Skipping.")
    else:
        # Default to all available data variables
        vars_to_plot = list(ds.data_vars.keys())
        print("No variables specified. Plotting all available variables.")

    # 4. Loop through the selected variables and create animations
    for var_name in vars_to_plot:
        print(f"Animating: {var_name}...")
        da = ds[var_name]
        
        fig, ax = plt.subplots()
        anim = da.epoch.animate(ax=ax)
        
        # Save the GIF in the same directory as the input files
        save_path = os.path.join(args.input_dir, f"{var_name}.gif")
        anim.save(save_path)
        
        plt.close(fig) # Crucial: frees up memory
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()