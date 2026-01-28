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




me=9.1e-31
c=3e8
e=1.6e-19
llambda=1e-6
e0=8.85e-12
omega=2*np.pi*c/llambda

norm=(2*np.pi*me*c)/(llambda*e)
#Bzz=Bz/norm
#Exx=Ex/(norm*c)
#Eyy=Ey/(norm*c)
#Ezz=Ez/(norm*c)

critical_density = e0*me*omega**2/e**2 #in 1/m3

#ne_beam_norm = ne_beam/critical_density

# ----------------------------
# Available variables to be displayed
# ----------------------------
VARIABLES = {
    "Jx": {
        "file": "Jx.hdf5",
        "dataset": "Jx",
        "pool": 1,
        "range": [-50, 50],
    },
    "Jy": {
        "file": "Jy.hdf5",
        "dataset": "Jy",
        "pool": 1,
        "range": [-50, 50],
    },
    "Bz": {
        "file": "Bz.hdf5",
        "dataset": "Bz",
        "pool": 1,
        "range": [-50, 50],
    },
    "Ex": {
        "file": "Ex.hdf5",
        "dataset": "Ex",
        "pool": 1,
        "range": [-0.5, 0.5],
    },
    "Ey": {
        "file": "Ey.hdf5",
        "dataset": "Ey",
        "pool": 1,
        "range": [-0.5, 0.5],
    },
    "ne": {
        "file": "x_y.hdf5",
        "dataset": "xy",
        "pool": 2,
        "range": [0.01, 0.25],
    },
    "n_photon": {
        "file": "n_photon.hdf5",
        "dataset": "n_photon",
        "pool": 2,
        "range": [0.01, 0.25],
    },
    "poynt_x": {
        "file": "poynt_x.hdf5",
        "dataset": "poynt_x",
        "pool": 3,
        "range": [0.00001, 0.002],
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
        data = f[dataset_name][:]  # type: ignore
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
            return np.array(f[dataset][:], dtype=np.float32)
    except Exception as e:
        if verbose:
            print(f"⚠ Failed loading {label}: {e}")
        return None

def prepare_xye_arrays(xye: np.ndarray):
    """Transform x_y_Ekin array into per-energy arrays and combine E>=2 MeV."""
    labels = [f"{i} MeV" for i in range(xye.shape[-1])]
    arrays = [xye[:, :, :, i] for i in range(xye.shape[-1])]
    arrays[0] = sum(arrays[2:])
    labels[0] = "E>=2 MeV"
    return arrays, labels

def setup_colormaps():
    cmap1 = plt.cm.gist_yarg
    my_cmap = cmap1(np.arange(cmap1.N))
    my_cmap[:, -1] = 0.8
    cmap_new = ListedColormap(my_cmap)

    colors2 = [(0.2, 1, 0.01, (c+1)*0.5) for c in np.linspace(-1, 1, 200)]
    cmap_new2 = clr.LinearSegmentedColormap.from_list('mycmap2', colors2, N=200)

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
    def __init__(self, Jx, Jy, Bz, Ex, Ey, ne, n_photon, poynt_x, xye_arrays, xye_labels, time_step=0):
        self.Jx = Jx
        self.Jy = Jy
        self.Bz = Bz
        self.Ex = Ex
        self.Ey = Ey
        self.ne = ne
        self.n_photon = n_photon
        self.poynt_x = poynt_x
        self.xye_arrays = xye_arrays
        self.xye_labels = xye_labels
        self.time_step = time_step

        from collections import deque
        # ----------------------------
        # Pools for rotation
        # ----------------------------
        self.pool = deque()
        self.pool_labels = deque()
        self.pool_ranges = deque()
        for arr, label, rng in [(self.Jx, "Jx", [-0.2, 0.2]),
                                (self.Jy, "Jy", [-0.2, 0.2]),
                                (self.Bz, "Bz", [-0.2, 0.2]),
                                (self.Ex, "Ex", [-0.5, 0.5]),
                                (self.Ey, "Ey", [-0.5, 0.5])]:
            if arr is not None:
                self.pool.append(arr)
                self.pool_labels.append(label)
                self.pool_ranges.append(rng)

        self.pool2 = deque()
        self.pool2_labels = deque()
        self.pool2_ranges = deque()
        for arr, label, rng in [(self.ne, "ne", [0.01, 0.25]),
                                (self.n_photon, "n_photon", [0.01, 0.25])]:
            if arr is not None:
                self.pool2.append(arr)
                self.pool2_labels.append(label)
                self.pool2_ranges.append(rng)

        self.pool3 = deque()
        self.pool3_labels = deque()
        self.pool3_ranges = deque()
        for arr, label, rng in [(self.poynt_x, "poynt_x", [0.00001, 0.002])]:
            if arr is not None:
                self.pool3.append(arr)
                self.pool3_labels.append(label)
                self.pool3_ranges.append(rng)
        if self.xye_arrays:
            for arr, label in zip(self.xye_arrays, self.xye_labels):
                if arr.ndim == 3:
                    self.pool3.append(arr)
                    self.pool3_labels.append(label)
                    self.pool3_ranges.append([0.00004, 0.004])

        self.has_pool3 = len(self.pool3) > 0

        # ----------------------------
        # Current displayed arrays
        # ----------------------------
        self.show_value = self.pool[0] if self.pool else None
        self.show_label = self.pool_labels[0] if self.pool else ""
        self.show_range = self.pool_ranges[0] if self.pool else [0, 1]

        self.show_value2 = self.pool2[0] if self.pool2 else None
        self.show_label2 = self.pool2_labels[0] if self.pool2 else ""
        self.show_range2 = self.pool2_ranges[0] if self.pool2 else [0, 1]

        if self.has_pool3:
            self.show_value3 = self.pool3[0]
            self.show_label3 = self.pool3_labels[0]
            self.show_range3 = self.pool3_ranges[0]
        else:
            self.show_value3 = None
            self.show_label3 = None
            self.show_range3 = None

        # ----------------------------
        # Colormaps
        # ----------------------------
        self.cmap_new, self.cmap_new2, self.cmap_newB = setup_colormaps()
        self.borders = (0.0001, 200, 0.0001, 24)

        # ----------------------------
        # Setup figure
        # ----------------------------
        self.fig, self.ax = plt.subplots()

        # Pool 1
        self.img1 = None
        self.cbar1 = None
        if self.pool:
            self.img1 = self.ax.imshow(
                np.transpose(self.show_value[self.time_step]),
                cmap=self.cmap_newB,
                vmin=self.show_range[0],
                vmax=self.show_range[1],
                extent=self.borders
            )
            self.cbar1 = plt.colorbar(self.img1, fraction=0.02, pad=0.1)
            self.cbar1.ax.set_title(f"${self.show_label}$")

        # Pool 2
        self.img2 = None
        self.cbar2 = None
        if self.pool2:
            self.img2 = self.ax.imshow(
                np.transpose(self.show_value2[self.time_step]),
                cmap=self.cmap_new,
                vmin=self.show_range2[0],
                vmax=self.show_range2[1],
                extent=self.borders
            )
            self.cbar2 = plt.colorbar(self.img2, fraction=0.02, pad=0.04)
            self.cbar2.ax.set_title(f"${self.show_label2}$")

        # Pool 3
        self.img3 = None
        self.cbar3 = None
        if self.has_pool3:
            self.img3 = self.ax.imshow(
                np.transpose(self.show_value3[self.time_step]),
                cmap=self.cmap_new2,
                vmin=self.show_range3[0],
                vmax=self.show_range3[1],
                extent=self.borders
            )
            self.cbar3 = plt.colorbar(self.img3, fraction=0.02, pad=0.06)
            self.cbar3.ax.set_title(f"${self.show_label3}$")

        # ----------------------------
        # Connect events
        # ----------------------------
        self.fig.canvas.mpl_connect('key_press_event', self.onclick)

        # Show figure
        plt.show()

    # ----------------------------
    # Refresh plot
    # ----------------------------
    def refresh_plot(self):
        # Pool 1
        if self.img1 is not None:
            self.img1.set_data(np.transpose(self.show_value[self.time_step]))
            self.img1.set_clim(vmin=self.show_range[0], vmax=self.show_range[1])
            if self.cbar1 is not None:
                self.cbar1.update_normal(self.img1)
                self.cbar1.ax.set_title(f"${self.show_label}$")
                self.cbar1.set_ticks(np.linspace(self.show_range[0], self.show_range[1], 5))

        # Pool 2
        if self.img2 is not None:
            self.img2.set_data(np.transpose(self.show_value2[self.time_step]))
            self.img2.set_clim(vmin=self.show_range2[0], vmax=self.show_range2[1])
            if self.cbar2 is not None:
                self.cbar2.update_normal(self.img2)
                self.cbar2.ax.set_title(f"${self.show_label2}$")
                self.cbar2.set_ticks(np.linspace(self.show_range2[0], self.show_range2[1], 5))

        # Pool 3
        if self.img3 is not None:
            self.img3.set_data(np.transpose(self.show_value3[self.time_step]))
            self.img3.set_clim(vmin=self.show_range3[0], vmax=self.show_range3[1])
            if self.cbar3 is not None:
                self.cbar3.update_normal(self.img3)
                self.cbar3.ax.set_title(f"${self.show_label3}$")
                self.cbar3.set_ticks(np.linspace(self.show_range3[0], self.show_range3[1], 5))

        # Title
        if self.has_pool3:
            title = f'{self.show_label3}  Ne={np.sum(self.show_value3[self.time_step]):.3f}  timestep={self.time_step*10:.1f}T'
        else:
            title = f'timestep={self.time_step*10:.1f}T'

        self.ax.set_title(title)
        self.fig.canvas.draw_idle()

    # ----------------------------
    # Key press handler
    # ----------------------------
    def onclick(self, event):
        # Time navigation
        if event.key == 'right':
            self.time_step = min(self.time_step + 1, self.show_value.shape[0]-1)
        elif event.key == 'left':
            self.time_step = max(self.time_step - 1, 0)
        elif event.key == 'shift+right':
            self.time_step = min(self.time_step + 10, self.show_value.shape[0]-1)
        elif event.key == 'shift+left':
            self.time_step = max(self.time_step - 10, 0)

        # Pool 1 rotation
        elif event.key == 'm':
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

        # Pool 2 rotation
        elif event.key == 'a':
            self.pool2.rotate(-1)
            self.pool2_labels.rotate(-1)
            self.pool2_ranges.rotate(-1)
            self.show_value2 = self.pool2[0]
            self.show_label2 = self.pool2_labels[0]
            self.show_range2 = self.pool2_ranges[0]
        elif event.key == 'd':
            self.pool2.rotate(1)
            self.pool2_labels.rotate(1)
            self.pool2_ranges.rotate(1)
            self.show_value2 = self.pool2[0]
            self.show_label2 = self.pool2_labels[0]
            self.show_range2 = self.pool2_ranges[0]

        # Pool 3 rotation
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

        # ----------------------------
        # Toggle pool visibility
        # ----------------------------
        elif event.key == '1' and self.img1 is not None:
            visible = not self.img1.get_visible()
            self.img1.set_visible(visible)
            if self.cbar1 is not None:
                self.cbar1.ax.set_visible(visible)

        elif event.key == '2' and self.img2 is not None:
            visible = not self.img2.get_visible()
            self.img2.set_visible(visible)
            if self.cbar2 is not None:
                self.cbar2.ax.set_visible(visible)

        elif event.key == '3' and self.img3 is not None:
            visible = not self.img3.get_visible()
            self.img3.set_visible(visible)
            if self.cbar3 is not None:
                self.cbar3.ax.set_visible(visible)

        # Refresh plot
        self.refresh_plot()


# ----------------------------
# CLI entry point
# ----------------------------
def run_cli():
    parser = argparse.ArgumentParser(description="Interactive plasma data visualisation CLI")
    parser.add_argument("--dir", type=str, default=".", help="Directory with HDF5 files")
    parser.add_argument("--time_step", type=int, default=0)
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

    # --------------------------------------------------------
    # Special post-processing
    # --------------------------------------------------------
    xye_arrays = []
    xye_labels = []
    if "xye" in loaded:
        xye_arrays, xye_labels = prepare_xye_arrays(loaded["xye"])

    # --------------------------------------------------------
    # Launch interactive viewer
    # --------------------------------------------------------
    DianaInteractive(
        Jx=loaded.get("Jx"),
        Jy=loaded.get("Jy"),
        Bz=loaded.get("Bz"),
        Ex=loaded.get("Ex")/norm/c,
        Ey=loaded.get("Ey")/norm/c,
        ne=loaded.get("ne"),
        n_photon=loaded.get("n_photon"),
        poynt_x=loaded.get("poynt_x"),
        xye_arrays=xye_arrays,
        xye_labels=xye_labels,
        time_step=args.time_step,
    )


# ----------------------------
# If run directly
# ----------------------------
if __name__ == "__main__":
    run_cli()
