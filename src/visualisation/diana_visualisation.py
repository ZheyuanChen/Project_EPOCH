#!/usr/bin/env python3
"""
diana_visualisation.py
Interactive plasma data visualization CLI tool
"""

# ----------------------------
# Imports
# ----------------------------
import os
import sys
import argparse
from collections import deque

import numpy as np
import h5py

import matplotlib
matplotlib.use("TkAgg")  # must come before pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as clr


# ----------------------------
# Available variables to be displayed
# ----------------------------
VARIABLES = {
    "Bz": {
        "file": "Bz.hdf5",
        "dataset": "Bz",
        "pool": 1,
        "range": [-0.2, 0.2],
    },
    "Ex": {
        "file": "Ex.hdf5",
        "dataset": "Ex",
        "pool": 1,
        "range": [-0.2, 0.2],
    },
    "Ey": {
        "file": "Ey.hdf5",
        "dataset": "Ey",
        "pool": 1,
        "range": [-0.2, 0.2],
    },
    "ne": {
        "file": "x_y.hdf5",
        "dataset": "xy",
        "pool": 2,
        "range": [0.01, 0.25],
    },
    "xye": {
        "file": "x_y_Ekin.hdf5",
        "dataset": "xy_Ekin",
        "pool": 3,
        "range": [0.00004, 0.004],
        "postprocess": "xye",
    },
}

# ----------------------------
# Helper functions
# ----------------------------

def load_hdf5_file(filepath: str, dataset_name: str) -> np.ndarray:
    """Load dataset from HDF5 file safely."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"{filepath} not found")
    with h5py.File(filepath, "r") as f:
        if dataset_name not in f:
            raise KeyError(f"{dataset_name} not found in {filepath}")
        data = f[dataset_name][:] # type: ignore
    return np.array(data)

def probe_hdf5_variable(directory, file_name, dataset, label, verbose=True):
    path = os.path.join(directory, file_name)

    if not os.path.isfile(path):
        if verbose:
            print(f"⚠ Missing file: {file_name} ({label})")
        return None

    try:
        with h5py.File(path, "r") as f:
            if dataset not in f:
                if verbose:
                    print(f"⚠ Missing dataset '{dataset}' in {file_name}")
                return None
            return np.array(f[dataset][:])
    except Exception as e:
        if verbose:
            print(f"⚠ Failed loading {label}: {e}")
        return None


def prepare_xye_arrays(xye: np.ndarray):
    """Transform x_y_Ekin array into per-energy arrays and combine E>=2 MeV."""
    labels = [f"{i} MeV" for i in range(xye.shape[-1])]
    arrays = [xye[:, :, :, i] for i in range(xye.shape[-1])]
    
    # Combine energies >= 2 MeV
    arrays[0] = sum(arrays[2:])
    labels[0] = "E>=2 MeV"
    return arrays, labels


def setup_colormaps():
    """Return the three custom colormaps."""
    # Grey semi-transparent
    cmap1 = plt.cm.gist_yarg
    my_cmap = cmap1(np.arange(cmap1.N))
    my_cmap[:, -1] = 0.8
    cmap_new = ListedColormap(my_cmap)

    # Green
    colors2 = [(0.2, 1, 0.01, (c+1)*0.5) for c in np.linspace(-1, 1, 200)]
    cmap_new2 = clr.LinearSegmentedColormap.from_list('mycmap2', colors2, N=200)

    # Red/Blue
    cmapB = plt.cm.coolwarm
    my_cmapB = cmapB(np.arange(cmapB.N))
    aB = [abs(i-128)/128 for i in range(cmapB.N)]
    my_cmapB[:, -1] = aB
    cmap_newB = ListedColormap(my_cmapB)

    return cmap_new, cmap_new2, cmap_newB


# ----------------------------
# Main class
# ----------------------------

class DianaInteractive:
    def __init__(self, Bz, Ex, Ey, ne, xye_arrays, xye_labels, time_step=0):
        self.Bz = Bz
        self.Ex = Ex
        self.Ey = Ey
        self.ne = ne
        self.xye_arrays = xye_arrays
        self.xye_labels = xye_labels

        self.time_step = time_step

        # Pools for rotation
        self.pool = deque()
        self.pool_labels = deque()
        self.pool_ranges = deque()

        if self.Bz is not None:
            self.pool.append(self.Bz)
            self.pool_labels.append("Bz")
            self.pool_ranges.append([-0.2, 0.2])

        if self.Ex is not None:
            self.pool.append(self.Ex)
            self.pool_labels.append("Ex")
            self.pool_ranges.append([-0.2, 0.2])

        if self.Ey is not None:
            self.pool.append(self.Ey)
            self.pool_labels.append("Ey")
            self.pool_ranges.append([-0.2, 0.2])


        self.pool2 = deque()
        self.pool2_labels = deque()
        self.pool2_ranges = deque()

        if self.ne is not None:
            self.pool2.append(self.ne)
            self.pool2_labels.append("ne")
            self.pool2_ranges.append([0.01, 0.25])

        self.pool3 = deque()
        self.pool3_labels = deque()
        self.pool3_ranges = deque()

        if self.xye_arrays:
            for arr, label in zip(self.xye_arrays, self.xye_labels):
                if arr.ndim == 3:  # (t, x, y)
                    self.pool3.append(arr)
                    self.pool3_labels.append(label)
                    self.pool3_ranges.append([0.00004, 0.004])

        self.has_pool3 = len(self.pool3) > 0
        if not self.has_pool3:
            self.show_value3 = None
            self.show_label3 = None
            self.show_range3 = None

        #self.pool3_ranges = deque([[0.00004, 0.004]] * len(self.xye_arrays))

        # Current displayed arrays
        self.show_value = self.pool[0]
        self.show_label = self.pool_labels[0]
        self.show_range = self.pool_ranges[0]

        self.show_value2 = self.pool2[0]
        self.show_label2 = self.pool2_labels[0]
        self.show_range2 = self.pool2_ranges[0]
        if self.has_pool3:
                self.show_value3 = self.pool3[0]
                self.show_label3 = self.pool3_labels[0]
                self.show_range3 = self.pool3_ranges[0]

        # Colormaps
        self.cmap_new, self.cmap_new2, self.cmap_newB = setup_colormaps()
        self.borders = (0.0001, 200, 0.0001, 24)

        # Setup figure
        self.fig, self.ax = plt.subplots()
        self.img2 = self.ax.imshow(
            np.transpose(self.show_value2[self.time_step]),
            cmap=self.cmap_new,
            vmin=self.show_range2[0],
            vmax=self.show_range2[1],
            extent=self.borders
        )
        self.img1 = self.ax.imshow(
            np.transpose(self.show_value[self.time_step]),
            cmap=self.cmap_newB,
            vmin=self.show_range[0],
            vmax=self.show_range[1],
            extent=self.borders
        )
        if self.has_pool3:
            self.show_value3 = self.pool3[0]
            self.show_label3 = self.pool3_labels[0]
            self.show_range3 = self.pool3_ranges[0]

            self.img3 = self.ax.imshow(
                np.transpose(self.show_value3[self.time_step]),
                cmap=self.cmap_new2,
                vmin=self.show_range3[0],
                vmax=self.show_range3[1],
                extent=self.borders
            )
        else:
            self.img3 = None


        # Colorbars
        self.cbar1 = plt.colorbar(self.img1, fraction=0.02, pad=0.1)
        self.cbar1.ax.set_title('$B_z / a_0$')

        self.cbar2 = plt.colorbar(self.img2, fraction=0.02, pad=0.04)
        self.cbar2.ax.set_title('$n_e / n_{cr}$')

        if self.has_pool3:
            self.cbar3 = plt.colorbar(self.img3, fraction=0.02, pad=0.04)
            self.cbar3.ax.set_title(f"${self.show_label3}$")
        else:
            self.cbar3 = None


        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.onclick)

        # Show figure
        plt.show()

    # ----------------------------
    # Plot refresh
    # ----------------------------
    #def refresh_plot(self):
        # Update Pool 1
        self.img1.set_data(np.transpose(self.show_value[self.time_step]))
        self.img1.set_clim(vmin=self.show_range[0], vmax=self.show_range[1])
        self.cbar1.set_clim(vmin=self.show_range[0], vmax=self.show_range[1])
        self.cbar1.ax.set_title(f"${self.show_label}$")  # update title
        self.cbar1.draw_all()  # refresh

        # Update Pool 2
        # Uncomment when we have more than ne in pool2
        self.img2.set_data(np.transpose(self.show_value2[self.time_step]))
        self.img2.set_clim(vmin=self.show_range2[0], vmax=self.show_range2[1])
        self.cbar2.set_clim(vmin=self.show_range2[0], vmax=self.show_range2[1])
        self.cbar2.ax.set_title(f"${self.show_label2}$")  # update title
        self.cbar2.draw_all()  # refresh

        if self.img3 is not None:
            self.img3.set_data(np.transpose(self.show_value3[self.time_step]))
            self.img3.set_clim(vmin=self.show_range3[0],vmax=self.show_range3[1])
            self.cbar3.set_clim(vmin=self.show_range3[0], vmax=self.show_range3[1])
            self.cbar3.ax.set_title(f"${self.show_label3}$")
            self.cbar3.draw_all()

        #self.img1.set_clim(vmin=self.show_range[0], vmax=self.show_range[1])
        #self.img2.set_clim(vmin=self.show_range2[0], vmax=self.show_range2[1])
        #self.img3.set_clim(vmin=self.show_range3[0], vmax=self.show_range3[1])

        #self.ax.set_title(
        #    f'{self.show_label3}  Ne={np.sum(self.show_value3[self.time_step]):.3f}  timestep={self.time_step*10:.1f}T'
        #)
        if self.has_pool3:
            title = (
                f'{self.show_label3}  '
                f'Ne={np.sum(self.show_value3[self.time_step]):.3f}  '
                f'timestep={self.time_step*10:.1f}T'
            )
        else:
            title = f'timestep={self.time_step*10:.1f}T'

        self.ax.set_title(title)

        self.cbar1.set_ticks(np.linspace(self.show_range[0], self.show_range[1], 5))
        self.cbar2.set_ticks(np.linspace(self.show_range2[0], self.show_range2[1], 5))

        self.fig.canvas.draw()

    def refresh_plot(self):
        # Pool 1
        self.img1.set_data(np.transpose(self.show_value[self.time_step]))
        self.img1.set_clim(vmin=self.show_range[0], vmax=self.show_range[1])
        self.cbar1.ax.set_title(f"${self.show_label}$")
        self.cbar1.set_ticks(np.linspace(self.show_range[0], self.show_range[1], 5))

        # Pool 2
        self.img2.set_data(np.transpose(self.show_value2[self.time_step]))
        self.img2.set_clim(vmin=self.show_range2[0], vmax=self.show_range2[1])
        self.cbar2.ax.set_title(f"${self.show_label2}$")
        self.cbar2.set_ticks(np.linspace(self.show_range2[0], self.show_range2[1], 5))

        # Pool 3 (if exists)
        if self.img3 is not None:
            self.img3.set_data(np.transpose(self.show_value3[self.time_step]))
            self.img3.set_clim(vmin=self.show_range3[0], vmax=self.show_range3[1])
            self.cbar3.ax.set_title(f"${self.show_label3}$")
            self.cbar3.set_ticks(np.linspace(self.show_range3[0], self.show_range3[1], 5))

        # Title
        if self.has_pool3:
            title = (
                f'{self.show_label3}  '
                f'Ne={np.sum(self.show_value3[self.time_step]):.3f}  '
                f'timestep={self.time_step*10:.1f}T'
            )
        else:
            title = f'timestep={self.time_step*10:.1f}T'
        self.ax.set_title(title)

        # Refresh the figure
        self.fig.canvas.draw_idle()


    # ----------------------------
    # Key press handler
    # ----------------------------
    def onclick(self, event):
        if event.key == 'right': # right arrow: next timestep
            self.time_step = min(self.time_step + 1, self.show_value.shape[0]-1)
        elif event.key == 'left': # left arrow: previous timestep
            self.time_step = max(self.time_step - 1, 0)
        elif event.key == 'shift+right': # fast forward
            self.time_step = min(self.time_step + 10, self.show_value.shape[0]-1)
        elif event.key == 'shift+left': # fast backward
            self.time_step = max(self.time_step - 10, 0)
        elif event.key == 'm': # rotate main pool
            self.pool.rotate(-1)
            self.pool_labels.rotate(-1)
            self.pool_ranges.rotate(-1)
            self.show_value = self.pool[0]
            self.show_label = self.pool_labels[0]
            self.show_range = self.pool_ranges[0]
        elif event.key == 'n':
            self.pool.rotate(1)
            self.pool_labels.rotate(1)
            self.pool_ranges.rotate(1)
            self.show_value = self.pool[0]
            self.show_label = self.pool_labels[0]
            self.show_range = self.pool_ranges[0]
        elif event.key == 'M':
            self.pool2.rotate(-1)
            self.pool2_labels.rotate(-1)
            self.pool2_ranges.rotate(-1)
            self.show_value2 = self.pool2[0]
            self.show_label2 = self.pool2_labels[0]
            self.show_range2 = self.pool2_ranges[0]
        elif event.key == 'N':
            self.pool2.rotate(1)
            self.pool2_labels.rotate(1)
            self.pool2_ranges.rotate(1)
            self.show_value2 = self.pool2[0]
            self.show_label2 = self.pool2_labels[0]
            self.show_range2 = self.pool2_ranges[0]
        elif event.key == 'ctrl+m' and self.has_pool3:
            self.pool3.rotate(-1)
            self.pool3_labels.rotate(-1)
            self.pool3_ranges.rotate(-1)
            self.show_value3 = self.pool3[0]
            self.show_label3 = self.pool3_labels[0]
            self.show_range3 = self.pool3_ranges[0]
        elif event.key == 'ctrl+n' and self.has_pool3:
            self.pool3.rotate(1)
            self.pool3_labels.rotate(1)
            self.pool3_ranges.rotate(1)
            self.show_value3 = self.pool3[0]
            self.show_label3 = self.pool3_labels[0]
            self.show_range3 = self.pool3_ranges[0]

        self.refresh_plot()


# ----------------------------
# CLI entry point
# ----------------------------
def run_cli():
    parser = argparse.ArgumentParser(description="Interactive plasma data visualisation CLI")
    
    parser.add_argument("--dir", type=str, default=".", help="Directory with HDF5 files")
    parser.add_argument("--time_step", type=int, default=0)
    #parser.add_argument("--Bz_file", type=str, default="Bz.hdf5")
    #parser.add_argument("--Ex_file", type=str, default="Ex.hdf5")
    #parser.add_argument("--Ey_file", type=str, default="Ey.hdf5")
    #parser.add_argument("--ne_file", type=str, default="x_y.hdf5")
    #parser.add_argument("--xye_file", type=str, default="x_y_Ekin.hdf5")
    
    args = parser.parse_args()
    
    
    # Load data
    loaded = {}
    print("Probing HDF5 variables:")
    for key, meta in VARIABLES.items():
        data = probe_hdf5_variable(
            args.dir,
            meta["file"],
            meta["dataset"],
            key,
            verbose=True
        )
        if data is not None:
            loaded[key] = data

    if not loaded:
        raise RuntimeError("No valid HDF5 variables found.")
    
    #Bz = load_hdf5_file(f"{args.dir}/{args.Bz_file}", "Bz")
    #Ex = load_hdf5_file(f"{args.dir}/{args.Ex_file}", "Ex")
    #Ey = load_hdf5_file(f"{args.dir}/{args.Ey_file}", "Ey")
    #ne = load_hdf5_file(f"{args.dir}/{args.ne_file}", "xy")
    #xye = load_hdf5_file(f"{args.dir}/{args.xye_file}", "xy_Ekin")


    # --------------------------------------------------------
    # Special post-processing
    # --------------------------------------------------------
    xye_arrays = []
    xye_labels = []

    if "xye" in loaded:
        xye_arrays, xye_labels = prepare_xye_arrays(loaded["xye"])
    #xye_arrays, xye_labels = prepare_xye_arrays(xye)

    # --------------------------------------------------------
    # Launch interactive viewer
    # --------------------------------------------------------
    DianaInteractive(
        Bz=loaded.get("Bz"),
        Ex=loaded.get("Ex"),
        Ey=loaded.get("Ey"),
        ne=loaded.get("ne"),
        xye_arrays=xye_arrays,
        xye_labels=xye_labels,
        time_step=args.time_step,
    )


# ----------------------------
# If run directly
# ----------------------------
if __name__ == "__main__":
    run_cli()
