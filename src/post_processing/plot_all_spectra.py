import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Run all EPOCH spectra plotting scripts.")
    parser.add_argument('--dir', required=True, help="Input directory containing .sdf files")
    parser.add_argument('--out', default=None, help="Base output directory for plots")
    args = parser.parse_args()

    # Get the absolute path of the directory where THIS script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Make sure these match your actual filenames exactly!
    scripts = [
        "plot_energy_spectrum.py",
        "plot_xy_energy_heatmap.py",
        "plot_laser_intensity_spectrum.py"  # <-- Updated to match your actual filename
    ]

    for script_name in scripts:
        # Construct the full absolute path to the script
        script_path = os.path.join(base_dir, script_name)
        
        if not os.path.exists(script_path):
            print(f"Warning: Could not find {script_path}. Skipping...")
            continue
            
        print(f"\n[{script_name}] Starting...")
        
        # Build the command using the absolute path
        cmd = [sys.executable, script_path, "--dir", args.dir]
        if args.out:
            cmd.extend(["--out", args.out])
        
        try:
            # Run the script and wait for it to finish
            subprocess.run(cmd, check=True)
            print(f"[{script_name}] Finished successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[{script_name}] FAILED with error code {e.returncode}.")

    print("\nAll plotting routines completed!")

if __name__ == "__main__":
    main()