import os
import glob
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sdf_helper as sh

JOULES_TO_EV = 6.241509e18
JOULES_TO_MEV = JOULES_TO_EV * 1e-6
JOULES_TO_GEV = JOULES_TO_EV * 1e-9
RADIANS_TO_DEGREES = 180 / np.pi

def parse_math_string(val_str):
    """Safely evaluates strings containing 'pi' or basic math from the deck."""
    val_str = val_str.strip().lower()
    return float(eval(val_str, {"__builtins__": None}, {"pi": np.pi}))

def parse_input_deck(parent_dir, sdf_dir):
    """Parses the input.deck to find ranges for the xy_energy distribution."""
    deck_paths = [
        os.path.join(parent_dir, 'input.deck'),
        os.path.join(sdf_dir, 'input.deck')
    ]
    
    deck_file = None
    for path in deck_paths:
        if os.path.exists(path):
            deck_file = path
            break
            
    if not deck_file:
        raise FileNotFoundError("Could not find input.deck in parent_dir or sdf_dir.")
        
    print(f"Parsing metadata from {deck_file}...")
    
    with open(deck_file, 'r') as f:
        content = f.read()

    # FIX: Split the file into distinct blocks first so regex doesn't swallow multiple blocks!
    blocks = content.split('begin:dist_fn')
    block_text = None
    
    for block in blocks:
        if re.search(r'name\s*=\s*xy_energy', block, re.IGNORECASE):
            block_text = block
            break
            
    if not block_text:
        raise ValueError("Could not find 'name = xy_energy' dist_fn block in input.deck.")
    
    # Extract range1 (Angle)
    range1_match = re.search(r'range1\s*=\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', block_text)
    if not range1_match:
        raise ValueError("Could not find 'range1 = (...)' in the xy_energy block.")
    
    # Extract range2 (Energy in Joules)
    range2_match = re.search(r'range2\s*=\s*\(\s*([0-9\.eE\+\-]+)\s*,\s*([0-9\.eE\+\-]+)\s*\)', block_text)
    if not range2_match:
        raise ValueError("Could not find 'range2 = (...)' in the xy_energy block.")
        
    emin_j, emax_j = float(range2_match.group(1)), float(range2_match.group(2))
    emin_mev, emax_mev = emin_j * JOULES_TO_MEV, emax_j * JOULES_TO_MEV
    
    # Safely evaluate -pi and pi
    angle_min = parse_math_string(range1_match.group(1))
    angle_max = parse_math_string(range1_match.group(2))
    
    angle_min_deg = angle_min * RADIANS_TO_DEGREES
    angle_max_deg = angle_max * RADIANS_TO_DEGREES
    
    # Extract resolutions just to have them
    res1_match = re.search(r'resolution1\s*=\s*(\d+)', block_text)
    res2_match = re.search(r'resolution2\s*=\s*(\d+)', block_text)
    
    res1 = int(res1_match.group(1)) if res1_match else None
    res2 = int(res2_match.group(1)) if res2_match else None
    
    return emin_mev, emax_mev, angle_min_deg, angle_max_deg, res1, res2

def main():
    parser = argparse.ArgumentParser(description="Plot Angle-Energy Heatmaps directly from SDF files.")
    parser.add_argument('--dir', dest='input_dir', required=True, help="Input directory containing the .sdf files")
    parser.add_argument('--out', type=str, default=None, help="Output directory for heatmaps")
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    parent_dir = os.path.dirname(input_dir)
    
    if args.out:
        output_dir = os.path.abspath(args.out)
    else:
        output_dir = os.path.join(parent_dir, 'xy_angle_energy_heatmap')
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Heatmaps will be saved to: {output_dir}")

    # Parse boundaries from input deck
    try:
        emin_mev, emax_mev, angle_min_deg, angle_max_deg, res1, res2 = parse_input_deck(parent_dir, input_dir)
        print(f"Parsed Energy Range: {emin_mev:.2e} MeV to {emax_mev:.2e} MeV")
        print(f"Parsed Angle Range: {angle_min_deg:.2f}° to {angle_max_deg:.2f}°")
    except Exception as e:
        print(f"Warning: {e}")
        print("Will attempt to rely purely on SDF grid limits.")
        emin_mev, emax_mev, angle_min_deg, angle_max_deg = None, None, None, None

    sdf_files = sorted(glob.glob(os.path.join(input_dir, '[0-9][0-9][0-9][0-9].sdf')))
    if not sdf_files:
        print(f"No SDF files found in {input_dir}")
        return

    for sdf_file in sdf_files:
        file_num = os.path.basename(sdf_file).split('.')[0]
        data = sh.getdata(sdf_file, verbose=False)
        sim_time_fs = data.Header['time'] * 1e15
        
        # We will create a figure with 2 subplots (1 for electrons, 1 for photons)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        plotted_anything = False

        species_list = [
            ('Electron', 'dist_fn_xy_energy_Electron', axes[0]),
            ('Photon', 'dist_fn_xy_energy_Photon', axes[1])
        ]

        for sp_name, var_name, ax in species_list:
            if hasattr(data, var_name):
                dist_obj = getattr(data, var_name)
                z_data = dist_obj.data 
                
                grid_obj_name = f"Grid_xy_energy_{sp_name}"
                grid = getattr(data, grid_obj_name).data
                
                # grid[0] is Angle, grid[1] is Energy (in Joules)
                angle_grid = grid[0] * RADIANS_TO_DEGREES
                energy_grid_mev = grid[1] * JOULES_TO_MEV
                
                # --- NEW: Convert particle count to Energy Density ---
                # z_data is shape (N_angle, N_energy). 
                # Multiplying by energy_grid_mev weights each bin by its energy.
                z_energy_density = z_data * energy_grid_mev
                
                max_val = np.max(z_energy_density)
                
                # If the array is empty or zero, skip plotting to avoid LogNorm errors
                if max_val <= 0:
                    ax.set_title(f"{sp_name} (No particles yet)")
                    ax.axis('off')
                    continue
                
                # --- NEW: Dynamic Range for Colorbar ---
                # Set the minimum cutoff to 5 orders of magnitude below the peak.
                # Adjust '1e5' to '1e4' or '1e6' if you want more or less contrast.
                min_val = max_val / 1e5
                
                # Create coordinate meshes
                X, Y = np.meshgrid(energy_grid_mev, angle_grid)
                
                # Plot the heatmap using the new dynamic bounds and energy density data
                pcm = ax.pcolormesh(X, Y, z_energy_density, norm=LogNorm(vmin=min_val, vmax=max_val), cmap='viridis', shading='auto')
                
                ax.set_title(f'{sp_name} Energy Phase Space')
                ax.set_xlabel('Energy (MeV)')
                ax.set_ylabel('XY Angle (Degrees)')
                ax.set_xscale('log')
                
                # Apply limits if parsed from deck
                if emin_mev and emax_mev:
                    ax.set_xlim(emin_mev, emax_mev)
                if angle_min_deg is not None and angle_max_deg is not None:
                    ax.set_ylim(angle_min_deg, angle_max_deg)
                else:
                    ax.set_ylim(-180, 180)
                
                # Update the colorbar label to reflect the new physics
                fig.colorbar(pcm, ax=ax, label='Energy per Bin (MeV)')
                plotted_anything = True
            else:
                ax.set_title(f"{sp_name} (Not found in SDF)")
                ax.axis('off')

        if plotted_anything:
            fig.suptitle(f'Angle_Energy Heatmap at t = {sim_time_fs:.1f} fs', fontsize=16)
            plt.tight_layout()
            out_path = os.path.join(output_dir, f"xy_angle_energy_heatmap_{file_num}.png")
            plt.savefig(out_path, dpi=300)
            print(f"  Saved {out_path}")
            
        plt.close(fig)

    print("Finished plotting all heatmaps!")

if __name__ == "__main__":
    main()