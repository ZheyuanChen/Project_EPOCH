import xarray as xr
import sdf_xarray as sdfxr
import matplotlib.pyplot as plt
import sdf_helper as sh
from matplotlib.animation import PillowWriter
import sdf
from matplotlib import animation

import os
from matplotlib.colors import LogNorm
from matplotlib.backend_bases import KeyEvent
from matplotlib.widgets import Slider
def read_sdffiles_from_directory(directory_path, verbose=False):
    """Read all SDF files from a directory and return a list of data objects.
    Change verbose to be True to see loading messages. Useful to debug loading issues(broken files etc).
    """

    # Collect all SDF filenames. os.listdir lists all files; if statement filters for those ending with .sdf
    files = [f for f in os.listdir(directory_path) if f.endswith(".sdf")]
    # Raise error if no files found
    if len(files) == 0:
        raise ValueError(f"No .sdf files found in directory: {directory_path}")
    # Sort files numerically by SDF number (allows for number jumping)
    def sdf_number(f):
        return int("".join(filter(str.isdigit, f)))

    files.sort(key=sdf_number)
    # Load all SDFs in the directory into a list called data_list
    data_list = []
    for f in files:
        number = sdf_number(f)
        data_list.append(sh.getdata(number, directory_path, verbose=verbose))
    return data_list

def use_plot_auto_to_visualise_a_single_sdf_data(data_sh):
    """Use sdf_helper's plot_auto function to visualise a single SDF data set.
    This function presents what stuff is available to plot and asks the user to input the variable name they want to plot.
    It then retrieves that variable from the data and calls plot_auto on it.
    """
    print('Choose a variable to plot from the following list:')
    # list available variables
    sh.list_variables(data_sh)
    # ask user for variable name
    variable_name = input("Enter the variable name to plot: ").strip() # strip() removes any leading/trailing whitespace
    # Get the variable inside `data_sh` using its string name
    variable = getattr(data_sh, variable_name)
    # Now call the auto plot function on that variable
    sh.plot_auto(variable)
    # Show the plot. This is blocking to allow user interaction.
    plt.show(block=True)

def animate_plot_auto_from_directory(directory_path, duration=0.1, verbose=False):
    """
    Create an animation using sh.plot_auto over all SDF
    files in a directory.
    Ask the user for the variable to plot.
    Autoplay. Can specify duration between frames.
    Change verbose to be True to see loading messages. Useful to debug loading issues(broken files etc).
    """

    # ----------------------------------------------------------
    # 1. Collect all SDF filenames
    # ----------------------------------------------------------
    # Gives a full list of .sdf files in the directory
    # os.listdir lists all files; if statement filters for those ending with .sdf;  os.path.join combines directory path with filename
    # in this case it joins directory_path with f (the filename) e.g. test_2d/0001.sdf
    # looping over all files in the directory makes 'files' a list (of all .sdf files paths). 
    
    # Alternative simpler version without full paths
    #files = [os.path.join(directory_path, f)
    #         for f in os.listdir(directory_path)
    #         if f.endswith(".sdf")]

    data_list = read_sdffiles_from_directory(directory_path, verbose=verbose)


    # ----------------------------------------------------------
    # 2. Ask user for variable to plot
    # ----------------------------------------------------------
    print('Choose a variable to plot from the following list:')
    sh.list_variables(data_list[0])       # get variable list from first SDF file.
    variable_name = input("Enter the variable name to plot: ").strip() # strip() removes any leading/trailing whitespace

    # ----------------------------------------------------------
    # 3. Set up figure and initial frame
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5)) # create figure and axis with specified size

    # get first variable
    var0 = getattr(data_list[0], variable_name)
    sh.plot_auto(var0, figure=fig, subplot=ax)

    # ----------------------------------------------------------
    # 4. Animation over all SDF files
    # ----------------------------------------------------------
    plt.ion()
    for i, data in enumerate(data_list):
        var = getattr(data, variable_name)
        plt.clf()
        sh.plot_auto(var)
        plt.pause(duration)   # adjust speed
    plt.ioff()
    plt.show()

def animate_plot2d_from_directory(directory_path, duration=0.1, verbose=False):
    """
    Create an animation using sh.plot_auto over all SDF
    files in a directory.
    Ask the user for the variable to plot.
    Autoplay. Can specify duration between frames.
    Change verbose to be True to see loading messages. Useful to debug loading issues(broken files etc).
    """

    # ----------------------------------------------------------
    # 1. Collect all SDF filenames
    # ----------------------------------------------------------
    # Gives a full list of .sdf files in the directory
    # os.listdir lists all files; if statement filters for those ending with .sdf;  os.path.join combines directory path with filename
    # in this case it joins directory_path with f (the filename) e.g. test_2d/0001.sdf
    # looping over all files in the directory makes 'files' a list (of all .sdf files paths). 
    
    # Alternative simpler version without full paths
    #files = [os.path.join(directory_path, f)
    #         for f in os.listdir(directory_path)
    #         if f.endswith(".sdf")]

    data_list = read_sdffiles_from_directory(directory_path, verbose=verbose)
    # ----------------------------------------------------------
    # 2. Ask user for variable to plot
    # ----------------------------------------------------------
    print('Choose a variable to plot from the following list:')
    sh.list_variables(data_list[0])       # get variable list from first SDF file.
    variable_name = input("Enter the variable name to plot: ").strip() # strip() removes any leading/trailing whitespace

    # ----------------------------------------------------------
    # 3. Set up figure and initial frame
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5)) # create figure and axis with specified size

    # get first variable
    var0 = getattr(data_list[0], variable_name)
    sh.plot2d(var0, figure=fig, subplot=ax)

    # ----------------------------------------------------------
# ----------------------------------------------------------
# 4. Animation over all SDF files (saved to GIF)
# ----------------------------------------------------------

    writer = PillowWriter(fps=int(1 / duration))

    with writer.saving(fig, "animation.gif", dpi=150):
        for i, data in enumerate(data_list):
            plt.clf()
            var = getattr(data, variable_name)
            sh.plot2d(var, interpolation='bicubic')
            plt.title(f"Frame {i}")
            writer.grab_frame()

def animate_plot2d_from_directory_manual_control(directory_path,verbose=False):
    """
    Manual-control animation using sh.plot2d over all SDF files in a directory.
    Use ← and → to step frames; 'q' to quit.
    This version preserves/recreates the colorbar correctly.
    """

    data_list = read_sdffiles_from_directory(directory_path, verbose=verbose)
    # ----------------------------------------------------------
    # 3. Ask user for variable
    # ----------------------------------------------------------
    print("Choose a variable to plot:")
    sh.list_variables(data_list[0])
    variable_name = input("Variable name: ").strip()

    # ----------------------------------------------------------
    # 4. Prepare figure and initial plot
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    current_index = {"i": 0}  # mutable holder

    # I don't understand this. But this way the plots are rendered correctly with colorbars and labels.
    def safe_remove_colorbar(ax_obj):
        """Remove colorbar axes created by sh.plot2d if present."""
        if hasattr(ax_obj, "colorbar"):
            try:
                # .colorbar is the axes object used for the cbar (per sdf_helper)
                ax_obj.colorbar.remove()
            except Exception:
                try:
                    # fallback: search figure axes and remove one that's not the main ax
                    for a in list(fig.axes):
                        if a is not ax_obj:
                            fig.delaxes(a)
                except Exception:
                    pass
            # clean up attribute to avoid repeated calls
            try:
                delattr(ax_obj, "colorbar")
            except Exception:
                # if hasattr not supported, ignore
                pass

    def update_frame(idx, redraw=True):
        """Update plot for frame idx. redraw=False will only update title (if needed)."""
        # Remove previous colorbar safely (so it doesn't get orphaned)
        safe_remove_colorbar(ax)

        # Clear only the plotting axes (not the whole figure)
        ax.cla()

        # Get variable object from data_list (handle blocklist if needed)
        data = data_list[idx]
        if hasattr(data, variable_name):
            var = getattr(data, variable_name)
        elif hasattr(data, "blocklist") and variable_name in data.blocklist:
            var = data.blocklist[variable_name]
        else:
            raise ValueError(f"Variable '{variable_name}' not found in dataset.")

        # Plot into the same figure & axes; force hold=False so colorbar is created fresh
        # pass interpolation or other kwargs here
        sh.plot2d(var, figure=fig, subplot=ax, interpolation="bicubic", hold=False)

        # Optional: set a frame-specific title
        # try to display time if available
        #try:
        #    tval = getattr(data, "t", None)
        #    if tval is not None:
        #        ax.set_title(f"{variable_name} — frame {idx} — t = {float(tval):.3e}")
        #    else:
        #        ax.set_title(f"{variable_name} — frame {idx}")
        #except Exception:
        #    ax.set_title(f"{variable_name} — frame {idx}")

        # Make sure the canvas is updated
        fig.canvas.draw_idle()

    # draw first frame
    update_frame(0)

    # ----------------------------------------------------------
    # 5. Keyboard handler (left/right)
    # ----------------------------------------------------------
    def on_key(event: KeyEvent):
        if event.key == "right":
            if current_index["i"] < len(data_list) - 1:
                current_index["i"] += 1
                update_frame(current_index["i"])
        elif event.key == "left":
            if current_index["i"] > 0:
                current_index["i"] -= 1
                update_frame(current_index["i"])
        elif event.key == "q":
            plt.close(fig)

    # Keep the connection alive on the figure
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Use ← and → to move between frames. Press 'q' to quit.")
    plt.show()

def animate_plot2d_with_slider(directory_path):
    """
    Manual-control slider animation for SDF files using sh.plot2d.
    Slider persists; colorbar is removed/recreated per-frame to keep it correct.
    """

    # ----------------------------------------------------------
    # 1. Collect SDF files and read (streaming)
    # ----------------------------------------------------------
    files = [f for f in os.listdir(directory_path) if f.endswith(".sdf")]
    if len(files) == 0:
        raise ValueError(f"No .sdf files found in directory: {directory_path}")

    def sdf_number(f):
        return int("".join(filter(str.isdigit, f)))
    files.sort(key=sdf_number)

    # read data into list (you can change this to stream if memory is a problem)
    data_list = [sh.getdata(sdf_number(f), directory_path, verbose=False) for f in files]

    # ----------------------------------------------------------
    # 2. Ask user for variable
    # ----------------------------------------------------------
    print("Choose a variable to plot:")
    sh.list_variables(data_list[0])
    variable_name = input("Variable name: ").strip()

    # ----------------------------------------------------------
    # 3. Setup figure & slider axes (create once, never clear figure)
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    plt.subplots_adjust(bottom=0.20)  # reserve space for slider; do this once

    # slider axes: create once and keep it
    ax_slider = plt.axes([0.15, 0.05, 0.70, 0.05])
    slider = Slider(ax=ax_slider, label='Frame', valmin=0, valmax=len(data_list)-1,
                    valinit=0, valstep=1)

    # helper to remove previously created colorbar axes (created by sh.plot2d)
    def safe_remove_colorbar(ax_obj, fig_obj):
        # remove attribute-held colorbar axis if present
        if hasattr(ax_obj, "colorbar"):
            try:
                ax_obj.colorbar.remove()
            except Exception:
                # fallback: search for extra axes in the figure and remove them if not the main ax or slider
                try:
                    for a in list(fig_obj.axes):
                        if a is not ax_obj and a is not ax_slider:
                            try:
                                fig_obj.delaxes(a)
                            except Exception:
                                pass
                except Exception:
                    pass
            # remove the attribute if possible
            try:
                delattr(ax_obj, "colorbar")
            except Exception:
                pass
        else:
            # no ax.colorbar attribute — still attempt to remove any extra axes that look like a colorbar
            try:
                extra_axes = [a for a in fig_obj.axes if a is not ax_obj and a is not ax_slider]
                for a in extra_axes:
                    try:
                        fig_obj.delaxes(a)
                    except Exception:
                        pass
            except Exception:
                pass

    # ----------------------------------------------------------
    # 4. Initial plot (frame 0)
    # ----------------------------------------------------------
    def get_var_from_data(d):
        if hasattr(d, variable_name):
            return getattr(d, variable_name)
        if hasattr(d, "blocklist") and variable_name in d.blocklist:
            return d.blocklist[variable_name]
        raise ValueError(f"Variable '{variable_name}' not found in dataset.")

    # plot initial frame
    var0 = get_var_from_data(data_list[0])
    # ensure no stale colorbar present
    safe_remove_colorbar(ax, fig)
    sh.plot2d(var0, figure=fig, subplot=ax, hold=False, interpolation='bicubic')
    #ax.set_title(f"{variable_name} — Frame 0/{len(data_list)-1}")
    fig.canvas.draw_idle()

    # ----------------------------------------------------------
    # 5. Update function called by slider
    # ----------------------------------------------------------
    def update(val):
        idx = int(slider.val)

        # remove previous colorbar axes so it doesn't get orphaned
        safe_remove_colorbar(ax, fig)

        # clear only the plotting axes (not the slider)
        ax.cla()

        # get variable object
        data = data_list[idx]
        var = get_var_from_data(data)

        # plot into same figure & axes (plot2d will recreate colorbar)
        sh.plot2d(var, figure=fig, subplot=ax, hold=False, interpolation='bicubic')

        

    slider.on_changed(update)

    # ----------------------------------------------------------
    # 6. Keyboard support uses the same slider (and thus same update)
    # ----------------------------------------------------------
    def on_key(event):
        if event.key == "right":
            if slider.val < len(data_list)-1:
                slider.set_val(slider.val + 1)
        elif event.key == "left":
            if slider.val > 0:
                slider.set_val(slider.val - 1)
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Use slider or ← → keys. Press 'q' to quit.")
    plt.show()

# Load a single SDF file using xarray, sdf_helper, or sdf
#ds = xr.open_dataset("./test_2d/0001.sdf")
#data_sh = sh.getdata(57,'./mini_project_plasma_slab/data_10pc/data_10pc_crit_density')
#data_sdf = sdf.read("./test_2d/0001.sdf")


# List variables using sdf_helper. This one is more comprehensive and contains everything
#sh.list_variables(data_sh)
#data_sh.

#plt.set_cmap('jet') # set default colormap to 'jet'

#plt.ion() # turn on interactive mode for plotting

#use_plot_auto_to_visualise_a_single_sdf_data(data_sh) # uncomment to use single file visualisation

#animate_plot_auto_from_directory('./mini_project_plasma_slab/data_10pc/data_10pc_crit_density') # uncomment to use animation over directory

# sh.plot_auto(data_sh.Derived_Average_Particle_Energy_Electron) # example of plotting a specific variable


animate_plot2d_from_directory('./test_2d', duration=0.05) # uncomment to use animation over directory

#animate_plot2d_from_directory_manual_control('./test_2d') # uncomment to use manual control animation over directory

#animate_plot2d_with_slider('./test_2d')
plt.show(block=True) # keep plots open

# List variables using sdf. For some reason this does not show all variables
#data_sdf.__dict__


