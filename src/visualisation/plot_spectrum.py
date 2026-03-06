import os
import glob
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import sdf_helper as sh

# Constants
JOULES_TO_EV = 6.241509e18
JOULES_TO_GEV = JOULES_TO_EV * 1e-9

def parse_input_deck(parent_dir, sdf_dir):
    """Parses the input.deck to find range3 and resolution3 for spatial_energy."""
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

    # Find the dist_fn block containing 'name = spatial_energy'
    # We use regex to isolate the specific block
    block_pattern = re.compile(r'begin:dist_fn.*?name\s*=\s*spatial_energy.*?end:dist_fn', re.DOTALL | re.IGNORECASE)
    match = block_pattern.search(content)
    
    if not match:
        raise ValueError("Could not find 'name = spatial_energy' dist_fn block in input.deck.")
        
    block_text = match.group(0)
    
    # Extract range3 (min_joules, max_joules)
    range_match = re.search(r'range3\s*=\s*\(\s*([0-9\.eE\+\-]+)\s*,\s*([0-9\.eE\+\-]+)\s*\)', block_text)
    if not range_match:
        raise ValueError("Could not find 'range3 = (min, max)' in the spatial_energy block.")
    
    min_j, max_j = float(range_match.group(1)), float(range_match.group(2))
    emin_gev, emax_gev = min_j * JOULES_TO_GEV, max_j * JOULES_TO_GEV
    
    # Extract resolution3
    res_match = re.search(r'resolution3\s*=\s*(\d+)', block_text)
    if not res_match:
        raise ValueError("Could not find 'resolution3 = ...' in the spatial_energy block.")
        
    resolution = int(res_match.group(1))
    
    return emin_gev, emax_gev, resolution

def main():
    parser = argparse.ArgumentParser(description="Plot 1D energy spectra from 3D EPOCH SDF distributions.")
    parser.add_argument('--dir', dest='input_dir', type=str, help="Directory containing the .sdf files (e.g., .../parent_dir/sdf_files)")
    parser.add_argument('--out', type=str, default=None, help="Output directory for plots. Defaults to parent_dir/energy_spectrum")
    parser.add_argument('--emin', type=float, default=None, help="Minimum energy in GeV")
    parser.add_argument('--emax', type=float, default=None, help="Maximum energy in GeV")
    parser.add_argument('--res', type=int, default=None, help="Energy resolution (number of bins)")
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    parent_dir = os.path.dirname(input_dir)
    
    # Resolve output directory
    if args.out:
        output_dir = os.path.abspath(args.out)
    else:
        output_dir = os.path.join(parent_dir, 'energy_spectrum')
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Fallback to input.deck parsing if any parameter is missing
    emin, emax, res = args.emin, args.emax, args.res
    if None in (emin, emax, res):
        print("Missing range or resolution arguments. Falling back to input.deck parsing...")
        deck_emin, deck_emax, deck_res = parse_input_deck(parent_dir, input_dir)
        emin = emin if emin is not None else deck_emin
        emax = emax if emax is not None else deck_emax
        res = res if res is not None else deck_res
        
    print(f"Plotting Parameters -> Range: {emin:.3e} to {emax:.3e} GeV | Resolution: {res} bins")

    # Get all SDF files and sort them
    sdf_files = sorted(glob.glob(os.path.join(input_dir, '[0-9][0-9][0-9][0-9].sdf')))
    if not sdf_files:
        print(f"No SDF files found in {input_dir}")
        return

    # Loop through and plot
    for sdf_file in sdf_files:
        file_basename = os.path.basename(sdf_file)
        file_num = file_basename.split('.')[0] # Extracts 'xxxx' from 'xxxx.sdf'
        
        print(f"Processing {file_basename}...")
        data = sh.getdata(sdf_file, verbose=False)
        
        plt.figure(figsize=(10, 6))
        
        # 1. Process Electrons
        if hasattr(data, 'dist_fn_spatial_energy_Electron'):
            elec_data = data.dist_fn_spatial_energy_Electron.data
            spec_elec = elec_data.sum(axis=(0, 1)) # Integrate over X and Y
            
            # The SDF grid is in Joules. Convert directly to eV for the X-axis
            en_grid_joules = data.Grid_spatial_energy_Electron.data[2]
            en_grid_ev = en_grid_joules * JOULES_TO_EV
            
            plt.plot(en_grid_ev, spec_elec, label='Electrons', color='blue', linewidth=2)
            
        # 2. Process Photons (if they exist in this dump)
        if hasattr(data, 'dist_fn_spatial_energy_Photon'):
            phot_data = data.dist_fn_spatial_energy_Photon.data
            spec_phot = phot_data.sum(axis=(0, 1))
            plt.plot(en_grid_ev, spec_phot, label='Photons', color='red', linewidth=2, linestyle='--')

        # Formatting (Log-Log scale, X-axis in eV)
        sim_time_fs = data.Header['time'] * 1e15
        
        plt.title(f'Energy Spectrum at t = {sim_time_fs:.1f} fs')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Number of Particles (a.u.)')
        
        plt.xscale('log')
        plt.yscale('log')
        
        # Set limits based on the parsed/provided GeV parameters (converted to eV)
        plt.xlim(emin * 1e9, emax * 1e9)
        
        # Add grid and legend
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(fontsize=12)
        
        # Save plot
        out_filename = f"energy_spectrum_{file_num}.png"
        out_path = os.path.join(output_dir, out_filename)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close() # Close the figure to free memory!

    print(f"Done! All spectra saved to {output_dir}")

if __name__ == "__main__":
    main()