import numpy as np
import scipy.constants as const

def propagate_thiele(Ex_0, Ey_0, dx, dy, dt, z_0, z_B):
    """
    Propagates transverse electric fields from z_0 to z_B using Thiele's algorithm.
    
    Parameters:
        Ex_0, Ey_0 : 3D numpy arrays (Nx, Ny, Nt) of transverse electric field (i.e. Ex(x,y,t) and Ey(x,y,t) with grid- and time-discretisation) at z=z_0
        dx, dy     : Spatial grid steps
        dt         : Time step
        z_0, z_B   : Initial and boundary z-coordinates
        
    Returns:
        E_B, B_B   : Dictionaries containing 3D arrays of the 3 components of E and B at z_B
    """
    Nx, Ny, Nt = Ex_0.shape
    c = const.c
    dz = z_B - z_0
    
    # ---------------------------------------------------------
    # Steps 1 & 2: 1D Time DFT and 2D Spatial DFT (Eqs 15 - 18)
    # ---------------------------------------------------------
    # We can do this in one step using a 3D N-dimensional FFT.
    # Note: Physics convention e^(i(wt - kx)) vs NumPy convention. 
    # NumPy handles the normalisations internally so IFFT(FFT(x)) = x perfectly.
    Ex_k_omega = np.fft.fftn(Ex_0, axes=(0, 1, 2))
    Ey_k_omega = np.fft.fftn(Ey_0, axes=(0, 1, 2))
    
    # Generate frequency and wavenumber grids
    kx_1d = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    ky_1d = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    omega_1d = np.fft.fftfreq(Nt, d=dt) * 2 * np.pi
    
    # Create 3D meshgrids for vectorized calculations
    kx, ky, omega = np.meshgrid(kx_1d, ky_1d, omega_1d, indexing='ij')
    
    # Avoid division by zero at DC component (omega=0, k=0)
    omega_safe = np.copy(omega)
    omega_safe[omega == 0] = 1e-15
    
    # ---------------------------------------------------------
    # Step 3: Transverse E-field at the boundary (Eqs 19 & 20)
    # ---------------------------------------------------------
    # Calculate kz squared. 
    kz_sq = (omega / c)**2 - kx**2 - ky**2
    
    # Mask to identify propagating waves (kz_sq > 0). Suppresses evanescent waves.
    prop_mask = kz_sq > 0
    kz = np.zeros_like(kz_sq)
    #kz[prop_mask] = np.sqrt(kz_sq[prop_mask])
    kz[prop_mask] = np.sign(omega[prop_mask]) * np.sqrt(kz_sq[prop_mask]) # Force kz to have the same sign as omega to ensure forward propagation
    # Apply phase shift (Eq 20)
    phase_shift = np.zeros_like(Ex_k_omega, dtype=complex)
    phase_shift[prop_mask] = np.exp(-1j * kz[prop_mask] * dz) # be careful about the phase convertion (i omega t or -i omega t). This convention is obtained by trial and error
    
    Ex_B_k = Ex_k_omega * phase_shift
    Ey_B_k = Ey_k_omega * phase_shift
    
    # ---------------------------------------------------------
    # Step 4: Longitudinal E-field at boundary (Eq 21)
    # ---------------------------------------------------------
    Ez_B_k = np.zeros_like(Ex_k_omega, dtype=complex)
    kz_safe = np.copy(kz)
    kz_safe[~prop_mask] = 1e-15 # Avoid div by zero, mask will zero it out anyway
    
    Ez_B_k[prop_mask] = -(kx[prop_mask] * Ex_B_k[prop_mask] + 
                          ky[prop_mask] * Ey_B_k[prop_mask]) / kz_safe[prop_mask]
    
    # ---------------------------------------------------------
    # Step 5: Magnetic field at boundary (Eqs 22 & 23)
    # ---------------------------------------------------------
    Bx_B_k = np.zeros_like(Ex_k_omega, dtype=complex)
    By_B_k = np.zeros_like(Ex_k_omega, dtype=complex)
    Bz_B_k = np.zeros_like(Ex_k_omega, dtype=complex)
    
    # Pre-calculate terms from the R matrix (Eq 22)
    R11 = -kx * ky
    R12 = kx**2 - (omega/c)**2
    R21 = (omega/c)**2 - ky**2
    R22 = kx * ky
    R31 = -ky * kz
    R32 = kx * kz
    
    prefactor = 1.0 / (omega_safe * kz_safe)
    
    # Apply Eq 23
    Bx_B_k[prop_mask] = prefactor[prop_mask] * (R11[prop_mask] * Ex_B_k[prop_mask] + 
                                                R12[prop_mask] * Ey_B_k[prop_mask])
    
    By_B_k[prop_mask] = prefactor[prop_mask] * (R21[prop_mask] * Ex_B_k[prop_mask] + 
                                                R22[prop_mask] * Ey_B_k[prop_mask])
                                                
    Bz_B_k[prop_mask] = prefactor[prop_mask] * (R31[prop_mask] * Ex_B_k[prop_mask] + 
                                                R32[prop_mask] * Ey_B_k[prop_mask])

    # ---------------------------------------------------------
    # Steps 6 & 7: Inverse 2D Spatial & 1D Time DFTs (Eqs 24 - 27)
    # ---------------------------------------------------------
    # Again, a 3D IFFT handles this perfectly. We take the real part 
    # because the physical fields must be purely real.
    E_B = {
        'x': np.real(np.fft.ifftn(Ex_B_k, axes=(0, 1, 2))),
        'y': np.real(np.fft.ifftn(Ey_B_k, axes=(0, 1, 2))),
        'z': np.real(np.fft.ifftn(Ez_B_k, axes=(0, 1, 2)))
    }
    
    B_B = {
        'x': np.real(np.fft.ifftn(Bx_B_k, axes=(0, 1, 2))),
        'y': np.real(np.fft.ifftn(By_B_k, axes=(0, 1, 2))),
        'z': np.real(np.fft.ifftn(Bz_B_k, axes=(0, 1, 2)))
    }
    
    return E_B, B_B

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
    
    # Input field at z=0
    Ey_0 = np.exp(- (X**2 + Y**2) / spot_size**2) * \
           np.exp(- (T - Nt*dt/2)**2 / pulse_duration**2) * \
           np.cos(omega_L * T)
    Ex_0 = np.zeros_like(Ey_0)
    
    z_0 = 0.0
    z_B = 10e-6 # Propagate forward by 10 microns
    
    print("Running Thiele algorithm...")
    E_out, B_out = propagate_thiele(Ex_0, Ey_0, dx, dy, dt, z_0, z_B)
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