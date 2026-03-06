import numpy as np
import matplotlib.pyplot as plt
import sdf_helper as sh

# Choose the output file you want to inspect (e.g., at the end of the simulation)
filename = "/home/pnd531/Desktop/Project_EPOCH/dev_test/test_dist_fn/sdf_files/0010.sdf"
print(f"Loading {filename} using sdf_helper...")


data = sh.getdata(filename, verbose=False)

# 1. Extract Electron Data (Using your exact variable name)
elec_obj = data.dist_fn_spatial_energy_Electron
elec_3d = elec_obj.data
# Integrate over X (axis 0) and Y (axis 1) to get the 1D energy array
spec_elec = elec_3d.sum(axis=(0, 1))

# 2. Extract the Grid
# For distribution functions, EPOCH's grid array matches the data size directly
en_centers_joules = data.Grid_spatial_energy_Electron.data[2]

# Convert Joules to GeV
joules_to_gev = 1.0 / (1.60218e-19 * 1e9)
en_centers_gev = en_centers_joules * joules_to_gev

# 3. Setup Plot
plt.figure(figsize=(10, 6))
plt.plot(en_centers_gev, spec_elec, label='Electrons', color='blue', linewidth=2)

# 4. Safely Extract Photon Data
if hasattr(data, 'dist_fn_spatial_energy_Photon'):
    phot_obj = data.dist_fn_spatial_energy_Photon
    phot_3d = phot_obj.data
    spec_phot = phot_3d.sum(axis=(0, 1))
    plt.plot(en_centers_gev, spec_phot, label='Photons', color='red', linewidth=2, linestyle='--')
else:
    print("Note: No photon distribution found in this file (normal for early timesteps).")

# 5. Formatting
sim_time_fs = data.Header['time'] * 1e15

plt.title(f'Particle Energy Spectra at t = {sim_time_fs:.1f} fs')
plt.xlabel('Energy (GeV)')
plt.ylabel('Number of Particles (a.u.)')
plt.yscale('log')
plt.xlim(0, 6) 
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('energy_spectra_corrected.png', dpi=300)
print("Plot saved as 'energy_spectra_corrected.png'")