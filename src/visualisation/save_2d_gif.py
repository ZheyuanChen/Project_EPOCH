import os
import sys
import argparse
import numpy
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import sdf_helper as sh
import matplotlib.ticker as ticker  # <-- NEEDED FOR TICK FORMATTING

def read_sdffiles_from_directory(directory_path, verbose=False):
    """Read all SDF files from a directory and return a list of data objects."""
    files = [f for f in os.listdir(directory_path) if f.endswith(".sdf")]
    
    if len(files) == 0:
        raise ValueError(f"No .sdf files found in directory: {directory_path}")

    def sdf_number(f):
        return int("".join(filter(str.isdigit, f)))

    files.sort(key=sdf_number)
    
    data_list = []
    for f in files:
        number = sdf_number(f)
        data_list.append(sh.getdata(number, directory_path, verbose=verbose))
    return data_list

def general_parser():
    parser = argparse.ArgumentParser(
        prog="SDF Animation Tool",
        description="Animate SDF files in a directory using sdf_helper visualisation functions.",
    )
    parser.add_argument(
        "--gif-filename",
        type=str,
        default="animation.gif",
        help="Output filename for the gif.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Input directory containing SDF files.",
    )
    parser.add_argument(
        "--variable-name-to-be-animated",
        type=str,
        default=None,
        help="Variable name to be animated.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.1,
        help="Duration between frames in seconds.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output during SDF file loading.",
    )
    return parser

def save_2d_animation_to_gif_fixed_colour_bar():
    parser = general_parser()
    args = parser.parse_args()
    
    directory_path = args.dir
    gif_filename = args.gif_filename
    verbose = args.verbose
    variable_name = args.variable_name_to_be_animated
    duration = args.duration

    # 1. Collect all SDF filenames
    try:
        data_list = read_sdffiles_from_directory(directory_path, verbose=verbose)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # 2. Ask user for variable to plot if not provided
    if variable_name is None:
        print('Choose a variable to plot from the following list:')
        sh.list_variables(data_list[0])
        variable_name = input("Enter the variable name to plot: ").strip()

    print("Creating animation...")

    if gif_filename == "animation.gif":
        gif_filename = f"{variable_name}_animation.gif"
        print(f"No gif filename provided, using default: {gif_filename}")

    # ----------------------------------------------------------
    # 3. Set up figure and initial frame (REVERTED TO YOUR ORIGINAL)
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    var0 = getattr(data_list[0], variable_name)
    sh.plot2d(var0, figure=fig, subplot=ax)
    #sh.plot_auto(var0, figure=fig, subplot=ax) # Plot_auto does not work.

    # ----------------------------------------------------------
    # 4. Animation over all SDF files (REVERTED TO YOUR ORIGINAL)
    # ----------------------------------------------------------

    print("Scanning data to lock color scale (this prevents the flashing)...")
    global_min = float('inf')
    global_max = float('-inf')
    for data in data_list:
        if hasattr(data, variable_name):
            val = getattr(data, variable_name).data
            global_min = min(global_min, val.min())
            global_max = max(global_max, val.max())

    writer = PillowWriter(fps=int(1 / duration))

    with writer.saving(fig, gif_filename, dpi=150):
        for i, data in enumerate(data_list):
            plt.clf()
            var = getattr(data, variable_name)
            #sh.plot_auto(var, figure=fig, subplot=ax) # Plot_auto does not work.
            sh.plot2d(var, interpolation='bicubic', vmin=global_min, vmax=global_max)
            plt.title(f"Frame {i}")

            # --- THE FIX ---
            # Lock the margins so the plot box stays exactly the same size 
            # and position on every frame, regardless of colorbar text width.
            #plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.90)

            writer.grab_frame()
            
            # Just printing progress so you know it hasn't frozen
            if i % 10 == 0:
                print(f"  Frame {i}/{len(data_list)} added...")

    print(f"\nSuccessfully saved to: {os.path.abspath(gif_filename)}")

def save_2d_animation_to_gif_unstable():
    parser = general_parser()
    args = parser.parse_args()
    
    directory_path = args.dir
    gif_filename = args.gif_filename
    verbose = args.verbose
    variable_name = args.variable_name_to_be_animated
    duration = args.duration
    
    # 1. Load Data
    try:
        data_list = read_sdffiles_from_directory(directory_path, verbose=verbose)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 2. Handle Variable Selection
    
    if variable_name is None:
        print('\nAvailable variables in first file:')
        sh.list_variables(data_list[0])
        variable_name = input("\nEnter variable name to animate: ").strip()

    # 3. Handle Output Filename
    if gif_filename == "animation.gif":
        gif_filename = f"{variable_name}_animation.gif"
        print(f"No gif filename provided, using default: {gif_filename}")

    print(f"Creating animation for '{variable_name}'...")
        # 3. Set up figure and initial frame (REVERTED TO YOUR ORIGINAL)
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    var0 = getattr(data_list[0], variable_name)
    sh.plot2d(var0, figure=fig, subplot=ax)
    #sh.plot_auto(var0, figure=fig, subplot=ax) # Plot_auto does not work.

    # ----------------------------------------------------------
    # 4. Animation over all SDF files (REVERTED TO YOUR ORIGINAL)
    # ----------------------------------------------------------

    print("Scanning data to lock color scale (this prevents the flashing)...")
    global_min = float('inf')
    global_max = float('-inf')
    for data in data_list:
        if hasattr(data, variable_name):
            val = getattr(data, variable_name).data
            global_min = min(global_min, val.min())
            global_max = max(global_max, val.max())

    writer = PillowWriter(fps=int(1 / duration))

    with writer.saving(fig, gif_filename, dpi=150):
        for i, data in enumerate(data_list):
            plt.clf()
            var = getattr(data, variable_name)
            #sh.plot_auto(var, figure=fig, subplot=ax) # Plot_auto does not work.
            sh.plot2d(var, interpolation='bicubic')
            plt.title(f"Frame {i}")

            # --- THE FIX ---
            # Lock the margins so the plot box stays exactly the same size 
            # and position on every frame, regardless of colorbar text width.
            #plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.90)

            writer.grab_frame()
            
            # Just printing progress so you know it hasn't frozen
            if i % 10 == 0:
                print(f"  Frame {i}/{len(data_list)} added...")

    print(f"\nSuccessfully saved to: {os.path.abspath(gif_filename)}")

def save_2d_animation_to_gif():
    parser = general_parser()
    args = parser.parse_args()
    
    directory_path = args.dir
    gif_filename = args.gif_filename
    verbose = args.verbose
    variable_name = args.variable_name_to_be_animated
    duration = args.duration
    
    # 1. Load Data
    try:
        data_list = read_sdffiles_from_directory(directory_path, verbose=verbose)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 2. Handle Variable Selection
    if variable_name is None:
        print('\nAvailable variables in first file:')
        sh.list_variables(data_list[0])
        variable_name = input("\nEnter variable name to animate: ").strip()

    # 3. Handle Output Filename
    if gif_filename == "animation.gif":
        gif_filename = f"{variable_name}_animation.gif"
        print(f"No gif filename provided, using default: {gif_filename}")

    print(f"Creating animation for '{variable_name}'...")
    
    # ----------------------------------------------------------
    # 3. Set up figure
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    # ----------------------------------------------------------
    # 4. Animation over all SDF files
    # ----------------------------------------------------------
    writer = PillowWriter(fps=int(1 / duration))

    with writer.saving(fig, gif_filename, dpi=150):
        for i, data in enumerate(data_list):
            plt.clf()
            var = getattr(data, variable_name)
            
            # Plot dynamically (no fixed vmin/vmax)
            sh.plot2d(var, interpolation='bicubic')
            plt.title(f"Frame {i}")

            # --- THE FRIEND'S FIX ---
            # sdf_helper usually places the colorbar in the second axes object
            if len(fig.axes) > 1:
                cax = fig.axes[1] 
                
                # 1. Force exactly 5 tick marks so the scale doesn't jump around
                cax.yaxis.set_major_locator(ticker.LinearLocator(numticks=5))
                
                # 2. Force scientific notation with exactly 2 decimal places.
                # PRO TIP: The space in '% .2e' is critical! It adds a blank space 
                # in front of positive numbers so they are exactly the same width 
                # as negative numbers (which have a '-').
                cax.yaxis.set_major_formatter(ticker.FormatStrFormatter('% .2e'))

            writer.grab_frame()
            
            if i % 10 == 0:
                print(f"  Frame {i}/{len(data_list)} added...")

    print(f"\nSuccessfully saved to: {os.path.abspath(gif_filename)}")


if __name__ == "__main__":
    sys.exit(save_2d_animation_to_gif())