
import sys
sys.path.insert(0, "~/Desktop/Project_EPOCH/src/field_propagator")  # add Folder_2 path to search list

import thiele_injector # type: ignore

import numpy as np
from scipy import constants as const
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Testing Thiele propagation with a dummy Gaussian pulse...")
    
    # Define grid
    Nx, Ny, Nt = 64, 64, 256
    dx, dy = 1e-6, 1e-6     # 1 um resolution
    dt = 0.5e-15            # 0.5 fs resolution to capture high frequencies
    
    x = np.arange(-Nx/2, Nx/2) * dx
    y = np.arange(-Ny/2, Ny/2) * dy
    t = np.arange(Nt) * dt
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    
    # Create a dummy laser pulse (Y-polarized, propagating in Z)
    omega_L = 2 * np.pi * const.c / 800e-9 # 800 nm laser
    pulse_duration = 10e-15
    spot_size = 5e-6
    
    # Input field at z=0; electric field is normalised to 1 (maximum amplitude = 1).
    Ey_0 = np.exp(- (X**2 + Y**2) / spot_size**2) * \
           np.exp(- (T - Nt*dt/2)**2 / pulse_duration**2) * \
           np.cos(omega_L * T)
    Ex_0 = np.zeros_like(Ey_0)
    
    z_0 = 0.0
    z_B = 10e-6 # Propagate forward by 10 microns
    
    print("Running Thiele algorithm...")
    E_out, B_out = thiele_injector.propagate_thiele(Ex_0, Ey_0, dx, dy, dt, z_0, z_B)
    print("Propagation complete. Generating plots...")

    # --- Data Preparation for Plotting ---
    
    # Coordinates in convenient units
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