import os
import glob
import argparse
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import sdf_helper as sh


def main():
    parser = argparse.ArgumentParser(
        description="Plot Laser Intensity and Frequency Spectrum."
    )
    parser.add_argument(
        "--dir",
        dest="input_dir",
        required=True,
        help="Input directory containing .sdf files",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Output directory for plots"
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    parent_dir = os.path.dirname(input_dir)

    # Put these in a dedicated subdirectory to play nice with the viewer
    output_dir = (
        os.path.abspath(args.out)
        if args.out
        else os.path.join(parent_dir, "laser_intensity_spectrum")
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir}")

    sdf_files = sorted(glob.glob(os.path.join(input_dir, "[0-9][0-9][0-9][0-9].sdf")))
    if not sdf_files:
        print(f"No SDF files found in {input_dir}")
        return

    for sdf_file in sdf_files:
        file_num = os.path.basename(sdf_file).split(".")[0]
        data = sh.getdata(sdf_file, verbose=False)
        sim_time_fs = data.Header["time"] * 1e15

        # Check if Ey exists
        if not hasattr(data, "Electric_Field_Ey"):
            print(f"Skipping {file_num}: No Electric_Field_Ey found.")
            continue

        Ey_data = data.Electric_Field_Ey.data

        # --- FIXED: Yee Grid Extraction ---
        grid = data.Grid_Grid.data
        x_nodes = grid[0]
        # Calculate cell centers for field variables (averaging the edges)
        x_centers = 0.5 * (x_nodes[:-1] + x_nodes[1:])
        x_centers_um = x_centers * 1e6

        # Handle 1D, 2D, or 3D data by taking a slice down the middle
        if Ey_data.ndim == 1:
            Ey_1d = Ey_data
        elif Ey_data.ndim == 2:
            y_center_idx = Ey_data.shape[1] // 2
            Ey_1d = Ey_data[:, y_center_idx]
        elif Ey_data.ndim == 3:
            y_center = Ey_data.shape[1] // 2
            z_center = Ey_data.shape[2] // 2
            Ey_1d = Ey_data[:, y_center, z_center]

        # 1. Calculate Intensity vs X (W/m^2)
        # Instantaneous intensity: I = c * eps_0 * E^2
        intensity_x = const.c * const.epsilon_0 * (Ey_1d**2)  # type: ignore
        # Convert to W/cm^2
        intensity_x_wcm2 = intensity_x / 1e4

        # 2. Calculate Spectral Intensity vs Omega
        dx = x_nodes[1] - x_nodes[0]
        n_points = len(Ey_1d)  # type: ignore

        # Perform FFT on the Electric Field
        Ey_fft = np.fft.fft(Ey_1d)  # type: ignore
        k_space = np.fft.fftfreq(n_points, d=dx) * 2 * np.pi

        # Map k to omega (omega = c * k)
        omega = np.abs(k_space * const.c)
        spectral_intensity = np.abs(Ey_fft) ** 2

        # Shift arrays to center the zero-frequency component
        omega_shifted = np.fft.fftshift(omega)
        spectrum_shifted = np.fft.fftshift(spectral_intensity)

        # We only care about positive frequencies
        pos_mask = omega_shifted > 0
        omega_pos = omega_shifted[pos_mask]
        spectrum_pos = spectrum_shifted[pos_mask]

        # Normalize spectrum for cleaner plotting
        if np.max(spectrum_pos) > 0:
            spectrum_pos = spectrum_pos / np.max(spectrum_pos)

        # --- Plotting ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Spatial Plot
        axes[0].plot(x_centers_um, intensity_x_wcm2, color="firebrick")
        axes[0].set_title("Laser Intensity Profile")
        # FIXED: Added 'r' before strings to handle LaTeX escapes
        axes[0].set_xlabel(r"Position x ($\mu$m)")
        axes[0].set_ylabel(r"Intensity (W/cm$^2$)")
        axes[0].grid(True, alpha=0.3)

        # Frequency Plot
        axes[1].plot(omega_pos / 1e15, spectrum_pos, color="navy")
        axes[1].set_title("Frequency Spectrum")
        # FIXED: Added 'r' before string
        axes[1].set_xlabel(r"$\omega$ (PHz)")
        axes[1].set_ylabel("Normalized Spectral Intensity")
        axes[1].set_xlim(0, 10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale("log")
        axes[1].set_ylim(1e-6, 1.2)

        fig.suptitle(f"Laser Diagnostics at t = {sim_time_fs:.1f} fs", fontsize=14)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"laser_diagnostics_{file_num}.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"  Saved {out_path}")

    print("Finished plotting laser diagnostics!")


if __name__ == "__main__":
    main()
