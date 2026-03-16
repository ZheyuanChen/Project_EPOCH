import numpy as np
from scipy import constants as const
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "~/Desktop/Project_EPOCH/src/field_propagator")  # add Folder_2 path to search list
import thiele_injector # type: ignore



# Define laser parameters
w_0 = 4e-6  # Spot size (4 microns)
lambda_0 = 1e-6  # Wavelength (1 micron)
z_rayleigh = np.pi * w_0**2 / (lambda_0)  # Rayleigh length
pulse_duration = 25e-15
laser_frequency = 2 * np.pi * const.c / lambda_0
t_centre = 100e-15  # Center of the pulse in time (100 fs)


# Now we can create a test grid and compute the electric field on it.
dt = 0.146e-15  # Time step (0.146 fs)
dy = 0.1172e-6  # Spatial step in y (0.117 microns)
dx = 0.1172e-6  # Spatial step in x (0.117 microns)
Nx = 512  # Number of points in x (512 points * 0.117 microns ≈ 60 microns total width)
Ny = 512  # Number of points in y (512 points * 0.117 microns ≈ 60 microns total width)
Nt = 2054 # Number of time steps (2054 steps * 0.146 fs ≈ 300 fs total duration)
z_0 = 0.0  # Initial position of the pulse (z=0)
z_B = 20e-6 # Propagate forward by 10 microns

def initialise_grids(Nx, Ny, Nt, dx, dy, dt):
    x = (np.arange(Nx) - Nx//2) * dx
    y = (np.arange(Ny) - Ny//2) * dy
    t = np.arange(Nt) * dt
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    return X, Y, T, x,y,t

def initialise_electric_field(X, Y, T):   
    # Initialize the electric field at z=0

    def gouy_phase(z):
        return np.arctan(z / z_rayleigh)

    def beam_width(z):
        return w_0 * np.sqrt(1 + (z / z_rayleigh)**2)

    def radius_of_curvature(z):
        if z == 0:
            return np.inf
        else:
            return z * (1 + (z_rayleigh / z)**2)

    # Define the electric field of a Gaussian beam. Note that the field propagates in the +z direction.
    def electric_field_spatial(x, y, z):
        r_squared = x**2 + y**2
        w_z = beam_width(z)
        R = radius_of_curvature(z)
        amplitude = np.exp(-r_squared / w_z**2 - 1j * laser_frequency * r_squared/(2*const.c * R) + 1j * gouy_phase(z))
        return amplitude

    def electric_field_temporal(t): 
        amplitude = np.exp(-1j * laser_frequency * t) * np.exp(- 4*np.log(2)*(t-t_centre)**2/pulse_duration**2) 
        return amplitude
    def electric_field(x, y, z, t):
        return electric_field_spatial(x, y, z) * electric_field_temporal(t)

    Ex_0 = np.zeros_like(X)  # No x-polarization
    Ey_0 = electric_field(X, Y, 0, T)  # y-polarized Gaussian beam at z=0
    return Ex_0, Ey_0

X, Y, T, x, y, t = initialise_grids(Nx, Ny, Nt, dx, dy, dt)
Ex_0, Ey_0 = initialise_electric_field(X, Y, T)
E_out, B_out = thiele_injector.propagate_thiele(Ex_0, Ey_0, dx, dy, dt, z_0, z_B)

y_um = y * 1e6
x_um = x * 1e6
t_fs = t * 1e15
    
    # Slices at x = center (Nx//2)
center_x = Nx // 2
Ey_in_yt = Ey_0[center_x, :, :]
Ey_out_yt = E_out['y'][center_x, :, :]
    
# 1 & 2. Calculate I(y, omega) using 1D FFT along the time axis (axis=1)
Ey_in_yw = np.fft.fft(Ey_in_yt, axis=1)
Ey_out_yw = np.fft.fft(Ey_out_yt, axis=1)
    
I_in_yw = np.abs(Ey_in_yw)**2
I_out_yw = np.abs(Ey_out_yw)**2
    
# Get positive frequencies and convert to PHz
omega = np.fft.fftfreq(Nt, d=dt) * 2 * np.pi
pos_mask = omega > 0
omega_pos_PHz = omega[pos_mask] / 1e15
    
I_in_plot = I_in_yw[:, pos_mask]
I_out_plot = I_out_yw[:, pos_mask]
    
# Find the time index where the output pulse reaches its peak for plot 3
#t_peak_idx = np.unravel_index(np.argmax(np.abs(E_out['y'])), E_out['y'].shape)[2]
t_peak_idx = np.unravel_index(np.argmax(E_out['y']), E_out['y'].shape)[2]
# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extents for imshow (left, right, bottom, top)
extent_yw = [omega_pos_PHz[0], omega_pos_PHz[-1], y_um[0], y_um[-1]]
extent_xy = [x_um[0], x_um[-1], y_um[0], y_um[-1]]
extent_ty = [t_fs[0], t_fs[-1], y_um[0], y_um[-1]]

# 1. Input I(y, omega)
im1 = axes[0, 0].imshow(I_in_plot, aspect='auto', origin='lower', extent=extent_yw, cmap='inferno')
axes[0, 0].set_title(r'Input $I(y, \omega)$ at $x=0$')
axes[0, 0].set_xlabel(r'$\omega$ (PHz)')
axes[0, 0].set_ylabel(r'$y$ ($\mu$m)')
axes[0, 0].set_xlim(1.5, 3.5) # Zoom in around 800nm (which is ~2.35 PHz)
fig.colorbar(im1, ax=axes[0, 0], label='Intensity')
# 2. Output I(y, omega)
im2 = axes[0, 1].imshow(I_out_plot, aspect='auto', origin='lower', extent=extent_yw, cmap='inferno')
axes[0, 1].set_title(r'Output $I(y, \omega)$ at $x=0$')
axes[0, 1].set_xlabel(r'$\omega$ (PHz)')
axes[0, 1].set_ylabel(r'$y$ ($\mu$m)')
axes[0, 1].set_xlim(1.5, 3.5)
fig.colorbar(im2, ax=axes[0, 1], label='Intensity')

# 3. Output E(x, y) at peak time
E_xy = E_out['y'][:, :, t_peak_idx].T # Transpose for proper orientation
im3 = axes[1, 0].imshow(E_xy, aspect='auto', origin='lower', extent=extent_xy, cmap='RdBu_r')
axes[1, 0].set_title(rf'Output $E_y(x, y)$ at $t={t_fs[t_peak_idx]:.1f}$ fs')
axes[1, 0].set_xlabel(r'$x$ ($\mu$m)')
axes[1, 0].set_ylabel(r'$y$ ($\mu$m)')
fig.colorbar(im3, ax=axes[1, 0], label='Electric Field')

# 4. Output E(t, y) at x = center
E_ty = Ey_out_yt
im4 = axes[1, 1].imshow(E_ty, aspect='auto', origin='lower', extent=extent_ty, cmap='RdBu_r')
axes[1, 1].set_title(r'Output $E_y(t, y)$ at $x=0$')
axes[1, 1].set_xlabel(r'$t$ (fs)')
axes[1, 1].set_ylabel(r'$y$ ($\mu$m)')
fig.colorbar(im4, ax=axes[1, 1], label='Electric Field')

plt.tight_layout()
plt.show()