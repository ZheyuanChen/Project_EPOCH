import scipy.constants as spc
import numpy as np
import xarray as xr
import argparse


def calculate_a_0(intensity_in_W_per_cm2, wavelength_in_microns):
    '''
    Calculate the normalised vector potential a_0 for a given laser intensity and wavelength.
    Parameters:
    - intensity_in_W_per_cm2: Laser intensity in W/cm^2
    - wavelength_in_microns: Laser wavelength in microns (µm)
    Returns:
    - a_0: The normalised vector potential (dimensionless)
    '''
    # The formula a_0 = 0.854 * sqrt(I * λ^2 / 1e18) expects practical units.
    # No conversion to W/m^2 or meters is needed!
    
    a_0 = 0.854 * np.sqrt(intensity_in_W_per_cm2 * (wavelength_in_microns ** 2) / 1e18)
    
    return a_0


def print_a_0():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--intensity", type=float, 
        help="Laser intensity in W/cm^2 (default: 1e22)",
        default=1e22
    )
    parser.add_argument(
        "-w", "--wavelength", type=float, 
        help="Laser wavelength in microns (µm) (default: 1.0)",
        default=1.0
    )
    args = parser.parse_args()
    a_0 = calculate_a_0(args.intensity, args.wavelength)
    print(f"Wavelength (µm): {args.wavelength} | Intensity (W/cm^2): {args.intensity} | Calculated a_0: {a_0:.3f}")

def calculate_chi_e(energy_in_MeV, intensity_in_W_per_cm2):
    '''
    Calculate the quantum parameter chi_e for a given electron energy and laser intensity.
    
    Parameters:
    - energy_in_MeV: Electron beam initial energy in MeV
    - intensity_in_W_per_cm2: Laser intensity in W/cm^2
    
    Returns:
    - chi_e: The quantum parameter (dimensionless)
    '''
    # Convert energy from MeV to GeV
    energy_in_GeV = energy_in_MeV / 1000.0
    
    # Scale intensity to units of 10^21 W/cm^2
    intensity_scaled = intensity_in_W_per_cm2 / 1e21
    
    # Calculate chi_e using the formula: chi_e = 0.18 * E_0[GeV] * I_0^{1/2}[10^21 W/cm^2]
    chi_e = 0.18 * energy_in_GeV * np.sqrt(intensity_scaled)
    
    return chi_e

def print_chi_e():
    parser = argparse.ArgumentParser(description="Calculate the quantum parameter chi_e")
    
    # Set up optional flags with sensible defaults
    parser.add_argument(
        "-e", "--energy", type=float, 
        help="Electron initial energy in MeV (default: 1000.0)",
        default=1000.0
    )
    parser.add_argument(
        "-i", "--intensity", type=float, 
        help="Laser intensity in W/cm^2 (default: 1e22)",
        default=1e22
    )

    args = parser.parse_args()
    
    chi_e = calculate_chi_e(args.energy, args.intensity)
    
    print(f"Energy (MeV): {args.energy} | Intensity (W/cm^2): {args.intensity:.2e} | Calculated chi_e: {chi_e:.3f}")

def calculate_critical_density(wavelength_in_microns):
    '''
    Calculate the critical plasma density for a given laser wavelength.
    Returns the density in cm^-3.
    '''
    # Formula: n_c = 1.11e21 / lambda^2
    n_c = 1.11e21 / (wavelength_in_microns ** 2)
    return n_c

def print_critical_density():
    parser = argparse.ArgumentParser(description="Calculate the critical plasma density for a given laser wavelength.")
    parser.add_argument(
        "-w", "--wavelength", type=float, 
        help="Laser wavelength in microns (µm) (default: 1.0)",
        default=1.0
    )
    args = parser.parse_args()
    
    n_c = calculate_critical_density(args.wavelength)
    
    print(f"Wavelength (µm): {args.wavelength} | Critical Density (cm^-3): {n_c:.2e}")

def calculate_ang_frequency_from_wavelength(wavelength_in_microns):
    '''
    Calculate the laser frequency from its wavelength.
    Returns the frequency in Hz.
    '''
    # Convert wavelength from microns to meters
    wavelength_in_meters = wavelength_in_microns * 1e-6
    ang_frequency = 2 * np.pi * spc.c / wavelength_in_meters
    return ang_frequency

def print_ang_frequency():
    parser = argparse.ArgumentParser(description="Calculate the angular frequency of a laser from its wavelength.")
    parser.add_argument(
        "-w", "--wavelength", type=float, 
        help="Laser wavelength in microns (µm) (default: 1.0)",
        default=1.0
    )
    args = parser.parse_args()
    
    ang_freq = calculate_ang_frequency_from_wavelength(args.wavelength)
    
    print(f"Wavelength (µm): {args.wavelength} | Angular Frequency (rad/s): {ang_freq:.3e}")

def calculate_power(intensity_in_W_per_cm2, waist_size_in_microns):
    '''
    Calculate the laser power from its intensity and spot size.
    Returns the power in Watts.
    '''
    # Convert waist size from microns to meters
    waist_size_in_meters = waist_size_in_microns * 1e-6
    area = np.pi * (waist_size_in_meters) ** 2
    power = intensity_in_W_per_cm2 * 1e4 * area /2 # Convert W/cm^2 to W/m^2
    return power

def print_power():
    parser = argparse.ArgumentParser(description="Calculate the laser power from its intensity and spot size.")
    parser.add_argument(
        "-i", "--intensity", type=float, 
        help="Laser intensity in W/cm^2 (default: 1e22)",
        default=1e22
    )
    parser.add_argument(
        "-w0", "--waist", type=float, 
        help="Laser waist size in microns (µm) (default: 5.0)",
        default=5.0
    )
    args = parser.parse_args()
    
    power = calculate_power(args.intensity, args.waist)
    
    print(f"Intensity (W/cm^2): {args.intensity} | Waist Size (µm): {args.waist} | Calculated Power (W): {power:.3e}")