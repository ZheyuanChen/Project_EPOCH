import sdf_xarray as sdfxr
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from matplotlib.animation import FuncAnimation

def read_multiple_sdf(file_parent_path, read_range=None, data_vars=None, convert_units=True, **kwargs):
    '''
    Read multiple SDF files into a single xarray Dataset using sdf_xarray's open_mfdataset.
    Return ds, which is an xarray Dataset containing all the data from the specified SDF files.
    Parameters:
    - file_parent_path: The directory containing the SDF files.
    - read_range: Optional tuple (lower, upper) to specify a range of files to read (e.g., (0, 100) to read files 0000.sdf to 0099.sdf). If None, all .sdf files in the directory will be read.
    - data_vars: Optional list of variable names to load. If None, all variables will be loaded.
    - convert_units: If True, automatically rescale time to femtoseconds and spatial coordinates to micrometers for easier interpretation. Set to False to keep original units.
    - **kwargs: Additional keyword arguments to pass to open_mfdataset (e.g., chunks={'time': 10} for Dask parallelisation).
    '''
    
    print("Starting to load files...")
    
    if read_range is None:
        file_pattern = os.path.join(file_parent_path, "*.sdf")
        files_to_load = sorted(glob.glob(file_pattern))
    else:
        lower, upper = read_range
        # Build the list and filter out non-existent files in one go
        files_to_load = [
            os.path.join(file_parent_path, f"{i:04d}.sdf") 
            for i in range(lower, upper)
        ]
        files_to_load = [f for f in files_to_load if os.path.exists(f)]

    if not files_to_load:
        raise FileNotFoundError("No files found matching your criteria. Check your paths.")
    
    print(f"Attempting to load {len(files_to_load)} files...")
    ds = sdfxr.open_mfdataset(files_to_load, data_vars=data_vars, **kwargs)
    
    if convert_units:
        ds = ds.epoch.rescale_coords(1e15, "fs", "time")
        ds = ds.epoch.rescale_coords(1e6, "µm", ["X_Grid_mid", "Y_Grid_mid"])
    
    print("Files loaded successfully!")
    return ds



def simple_animation(ds, variable_name, save_path=None, convert_units=False):
    '''
    Create a simple animation of a 2D variable across time using xarray's built-in animation capabilities.
    Parameters:
    - ds: The xarray Dataset containing the data.
    - variable_name: The name of the variable to animate.
    - save_path: Optional path to save the animation. If None, the animation will be displayed.
    - convert_units: If True, automatically rescale time to femtoseconds and spatial coordinates to micrometers for easier interpretation. Set to False to keep original units.
    '''
    if convert_units:
        # Chain the rescales cleanly
        ds_plot = ds.epoch.rescale_coords(1e15, "fs", "time")
        ds_plot = ds_plot.epoch.rescale_coords(1e6, "µm", ["X_Grid_mid", "Y_Grid_mid"])
        da = ds_plot[variable_name]
    else:
        da = ds[variable_name]
    
    anim = da.epoch.animate()
    
    if save_path:
        anim.save(save_path)
        print(f"Animation saved to {save_path}")
    else:
        anim.show() # plt.show() might be better?
        
    return anim # Returning the object is highly recommended if you use Jupyter Notebooks


def check_lbw_positron(ds, lbw_pos_name = None):
    '''
    Check the consistency of positron data in the LBW dataset by comparing the 2D density grid with the raw particle weights across time.
    '''

    times = ds["time"].values
    if lbw_pos_name is not None:
        number_density_name = "Derived_Number_Density_"+lbw_pos_name
        particle_weight_name = "Particles_Weight_"+lbw_pos_name
    else:
        number_density_name = "Derived_Number_Density_pos_lbw"
        particle_weight_name = "Particles_Weight_pos_lbw"


    for i, t in enumerate(times):
        t_fs = t * 1e15
        print(f"--- Time: {t_fs:5.1f} fs ---")

        # 1. Check the 2D density grid
        if number_density_name in ds.data_vars:
            # FIX: Use .isel(time=i) to grab only the 2D array for this specific timestep
            density_array = ds[number_density_name].isel(time=i).values
            
            total_density_sum = np.sum(density_array)
            max_density = np.max(density_array)
            
            print(f"  Grid  -> Max Density: {max_density:.2e} m^-3 | Grid Sum: {total_density_sum:.2e}")
        else:
            print(f"  Grid  -> '{number_density_name}' DNE")

        # 2. Check the raw particle weights
        if particle_weight_name in ds.data_vars:
            # FIX: Slice by time, and strip out the NaNs padded by mfdataset
            weights = ds[particle_weight_name].isel(time=i).values
            weights = weights[~np.isnan(weights)]
            
            physical_count = np.sum(weights)
            macro_count = len(weights)
            print(f"  Parts -> Macro-Positrons: {macro_count:5d} | Physical Positrons: {physical_count:.2e}")
        else:
            print(f"  Parts -> Variable '{particle_weight_name}' DNE")


def check_photon_threshold(file_parent_path, convert_units=False):
    '''
    This function scans through all SDF files in the specified directory, checks for photon particle data, 
    and calculates the maximum photon energy at each timestep. It then compares the maximum energy to the 0.511 MeV threshold 
    to determine if any photons exceed this limit, which is relevant for LBW pair production analysis.
    '''

    # Pass a glob string directly (e.g., "data/*.sdf") instead of needing the mfdataset 'ds'
    file_pattern = os.path.join(file_parent_path, "*.sdf")
    files = sorted(glob.glob(file_pattern))

    if not files:
        print("No files found.")
        return

    print("Scanning for high-energy photons...")
    
    # Use exact physics constants
    c = 299792458.0 
    joules_to_mev = 1.0 / 1.602176634e-13

    for f in files:
        try:
            # Load individual file to handle ragged particle arrays safely
            ds = xr.open_dataset(f)
            t = ds["time"].values
            if convert_units:
                t_fs = t * 1e15 
            else:
                t_fs = t

            # Photons can be named with different capitalizations, so we check for both possibilities
            # If no momenta data is found, skip the energy calculation for this file and move on to the next one
            if "Particles_Px_photon" in ds.data_vars:
                px = ds["Particles_Px_photon"].values
                py = ds["Particles_Py_photon"].values
                pz = ds["Particles_Pz_photon"].values
            elif "Particles_Px_Photon" in ds.data_vars:
                px = ds["Particles_Px_Photon"].values
                py = ds["Particles_Py_Photon"].values
                pz = ds["Particles_Pz_Photon"].values
            else:
                print(f"Time: {t_fs:5.1f} fs | No photon particle data found in {f}. Skipping energy calculation.")
                continue


            p = np.sqrt(px**2 + py**2 + pz**2)
            energy_mev = (p * c) * joules_to_mev
                
            max_energy = np.max(energy_mev)
            num_photons = len(px) 
                
            print(f"Time: {t_fs:5.1f} fs | Total number of photons: {num_photons:7d} | Max Energy: {max_energy:7.2f} MeV")
                
            if max_energy > 0.511:
                 print("  -> STATUS: Photons cross the 0.511 MeV rest-mass threshold!")


        except Exception as e:
            print(f"Error reading particles from {f}: {e}")
        finally:
            # Always close individual files in a loop to prevent memory/file-descriptor leaks
            if 'ds' in locals():
                ds.close()

def plot_laser_abs_frac(ds, save_parent_path=None, save_name="laser_absorption_fraction.png",does_save=True):

    ds["Laser_Absorption_Fraction_in_Simulation"] = (
        (ds["Total_Particle_Energy_in_Simulation"] - ds["Total_Particle_Energy_in_Simulation"][0])
        / ds["Absorption_Total_Laser_Energy_Injected"]
        ) 

    # We can also manipulate the units and other attributes
    ds["Laser_Absorption_Fraction_in_Simulation"].attrs["units"] = "%"
    ds["Laser_Absorption_Fraction_in_Simulation"].attrs["long_name"] = "Laser Absorption Fraction"

    ds["Laser_Absorption_Fraction_in_Simulation"].epoch.plot()
    plt.title("Laser absorption fraction in simulation")
    plt.show()

    if does_save:
        if save_parent_path is not None:
            save_path = os.path.join(save_parent_path, save_name)
        else:
            save_path = os.path.join(os.getcwd(), save_name)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")


def plot_absorption_rates(ds, lbw_pos_name=None, convert_units=False, data_var_suffix=None):
    '''
    Plot the absorption rates of lbw_positrons, lbw_electrons, electrons, and ions in the LBW dataset across time. This function calculates the absorption rate 
    by comparing the number of positrons in the grid (from the density variable) to the number of positrons represented 
    by the particle weights, and then plots this rate over time.
    '''

