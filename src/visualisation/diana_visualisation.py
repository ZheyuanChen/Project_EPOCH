#!/usr/bin/env python3
"""
diana_visualisation.py
Interactive plasma data visualization CLI tool
"""

# ----------------------------
# Imports
# ----------------------------
import os
import argparse
from collections import deque
import numpy as np
import h5py

import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# ----------------------------
# Global Physics State
# ----------------------------
# These are calculated dynamically based on user input (wavelength)
CONSTANTS = {} 

def setup_physics(lambda_nm):
    """
    Recalculate physics constants and normalization factors based on laser wavelength.
    """
    lambda_m = lambda_nm * 1.0e-9
    
    c = 299792458.0
    me = 9.10938356e-31
    e = 1.60217663e-19
    eps0 = 8.85418781e-12
    
    omega = 2 * np.pi * c / lambda_m
    nc = eps0 * me * omega**2 / e**2  # Critical Density
    
    print(f"Physics Setup: Lambda = {lambda_nm} nm")
    print(f"  -> Critical Density (nc) = {nc:.2e} m^-3")
    print(f"  -> Laser Frequency (w)   = {omega:.2e} rad/s")

    # Populate global constants dictionary
    CONSTANTS["NC"] = nc
    CONSTANTS["OMEGA"] = omega
    
    # Define Normalization Factors (SI -> Normalized)
    # This dictionary maps variable names to their divisor
    CONSTANTS["NORM_FACTORS"] = {
        # Density -> n / nc
        "ne": nc, 
        "n_photon": nc,
        
        # Fields -> a0 (Relativistic amplitude) = e E / (me omega c)
        "Ex": (me * omega * c) / e,
        "Ey": (me * omega * c) / e,
        "Ez": (me * omega * c) / e,
        
        # B field -> a0 = e B / (me omega)
        "Bz": (me * omega) / e, 
        "By": (me * omega) / e,
        "Bx": (me * omega) / e,

        # Current -> J / (e nc c)
        "Jx": e * nc * c,
        "Jy": e * nc * c,
        
        # Poynting -> Intensity / (me c^3 nc) 
        "poynt_x": me * c**3 * nc, 
        
        # Energy (raw eV/Joule usually requires specific handling, keeping 1.0 for now)
        "xye": 1.0 
    }

# ----------------------------
# Configuration
# ----------------------------
VARIABLES = {
    "Jx":       {"file": "Jx.hdf5",       "dataset": "Jx",        "pool": 1},
    "Jy":       {"file": "Jy.hdf5",       "dataset": "Jy",        "pool": 1},
    "Bz":       {"file": "Bz.hdf5",       "dataset": "Bz",        "pool": 1},
    "Ex":       {"file": "Ex.hdf5",       "dataset": "Ex",        "pool": 1},
    "Ey":       {"file": "Ey.hdf5",       "dataset": "Ey",        "pool": 1},
    "ne":       {"file": "n_e.hdf5",      "dataset": "ne",        "pool": 2}, 
    "n_photon": {"file": "n_photon.hdf5", "dataset": "n_photon",  "pool": 2},
    "poynt_x":  {"file": "poynt_x.hdf5",  "dataset": "poynt_x",   "pool": 3},
    "xye":      {"file": "x_y_Ekin.hdf5", "dataset": "xy_Ekin",   "pool": 3},
}

# ----------------------------
# Helper functions
# ----------------------------

def normalize_data(key, data):
    """Converts SI input data to normalized units using global constants."""
    factors = CONSTANTS.get("NORM_FACTORS", {})
    if key in factors:
        norm = factors[key]
        if norm != 0:
            return data / norm
    return data

def probe_hdf5_variable(directory, file_name, dataset, label, verbose=True):
    path = os.path.join(directory, file_name)
    if not os.path.isfile(path):
        if verbose: print(f"  [Skipped] {file_name} not found")
        return None
    try:
        with h5py.File(path, "r") as f:
            target_key = dataset
            if target_key not in f:
                keys = list(f.keys())
                if keys:
                    target_key = keys[0]
                else:
                    return None
            
            data = np.array(f[target_key][:], dtype=np.float32)
            
            # Normalize immediately using the calculated constants
            data = normalize_data(label, data)
            return data

    except Exception as e:
        if verbose: print(f"  [Error] Failed loading {label}: {e}")
        return None

def prepare_xye_arrays(xye: np.ndarray):
    """Transform x_y_Ekin array into per-energy arrays."""
    if xye.ndim != 4: 
        return [], []
        
    labels = [f"Energy Bin {i}" for i in range(xye.shape[-1])]
    arrays = [xye[:, :, :, i] for i in range(xye.shape[-1])]
    
    # Custom logic: If we have many bins, sum the high energy ones
    if len(arrays) > 2:
        arrays[0] = np.sum(xye[:,:,:,2:], axis=3)
        labels[0] = "High Energy Sum"
        
    return arrays, labels

def setup_colormaps():
    # 1. Dark background for fields
    cmap1 = plt.cm.gist_yarg
    my_cmap = cmap1(np.arange(cmap1.N))
    my_cmap[:, -1] = 0.8
    cmap_density = ListedColormap(my_cmap)

    # 2. Green transparent for overlays
    colors2 = [(0.2, 1, 0.01, (c+1)*0.5) for c in np.linspace(-1, 1, 200)]
    cmap_green = LinearSegmentedColormap.from_list('mycmap2', colors2, N=200)

    # 3. Diverging (Blue-Red) for Fields
    cmapB = plt.cm.coolwarm
    cmap_fields = ListedColormap(cmapB(np.arange(cmapB.N)))

    return cmap_density, cmap_green, cmap_fields

# ----------------------------
# Main class
# ----------------------------
class DianaInteractive:
    def __init__(self, data_dict, xye_arrays=None, xye_labels=None, time_step=0):
        
        self.data = data_dict
        self.xye_arrays = xye_arrays or []
        self.xye_labels = xye_labels or []
        self.time_step = time_step
        
        # --- Pools ---
        # Pool 1: Fields
        self.pool1 = deque()
        self.pool1_meta = deque()
        for key in ["Bz", "Ex", "Ey", "Jx", "Jy"]:
            if self.data.get(key) is not None:
                self.pool1.append(self.data[key])
                self.pool1_meta.append(key)

        # Pool 2: Densities
        self.pool2 = deque()
        self.pool2_meta = deque()
        for key in ["ne", "n_photon"]:
            if self.data.get(key) is not None:
                self.pool2.append(self.data[key])
                self.pool2_meta.append(key)

        # Pool 3: Overlays
        self.pool3 = deque()
        self.pool3_meta = deque()
        if self.data.get("poynt_x") is not None:
            self.pool3.append(self.data["poynt_x"])
            self.pool3_meta.append("poynt_x")
        for arr, lbl in zip(self.xye_arrays, self.xye_labels):
            self.pool3.append(arr)
            self.pool3_meta.append(lbl)

        # --- State ---
        self.state = {
            1: {"data": self.pool1[0] if self.pool1 else None, 
                "label": self.pool1_meta[0] if self.pool1 else ""},
            2: {"data": self.pool2[0] if self.pool2 else None, 
                "label": self.pool2_meta[0] if self.pool2 else ""},
            3: {"data": self.pool3[0] if self.pool3 else None, 
                "label": self.pool3_meta[0] if self.pool3 else ""},
        }

        # --- Figure ---
        self.cmap_density, self.cmap_green, self.cmap_fields = setup_colormaps()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        self.imgs = {1: None, 2: None, 3: None}
        self.cbars = {1: None, 2: None, 3: None}

        # Init Images
        if self.state[1]["data"] is not None:
            self.imgs[1] = self.ax.imshow(
                np.transpose(self.state[1]["data"][self.time_step]),
                cmap=self.cmap_fields, origin='lower', aspect='auto'
            )
            self.cbars[1] = plt.colorbar(self.imgs[1], ax=self.ax, fraction=0.046, pad=0.04)

        if self.state[2]["data"] is not None:
            self.imgs[2] = self.ax.imshow(
                np.transpose(self.state[2]["data"][self.time_step]),
                cmap=self.cmap_density, origin='lower', aspect='auto', alpha=0.6
            )
            self.cbars[2] = plt.colorbar(self.imgs[2], ax=self.ax, fraction=0.046, pad=0.04)

        if self.state[3]["data"] is not None:
            self.imgs[3] = self.ax.imshow(
                np.transpose(self.state[3]["data"][self.time_step]),
                cmap=self.cmap_green, origin='lower', aspect='auto', alpha=0.5
            )
            self.cbars[3] = plt.colorbar(self.imgs[3], ax=self.ax, fraction=0.046, pad=0.04)

        self.fig.canvas.mpl_connect('key_press_event', self.onclick)
        
        self.refresh_plot()
        plt.show()

    def get_auto_clim(self, data, symmetric=False):
        vmin, vmax = np.percentile(data, [1, 99])
        if symmetric:
            abs_max = max(abs(vmin), abs(vmax))
            return -abs_max, abs_max
        return vmin, vmax

    def refresh_plot(self):
        title_parts = [f"T={self.time_step}"]

        # Update Loop for all 3 pools
        for idx in [1, 2, 3]:
            if self.imgs[idx] is not None:
                data_vol = self.state[idx]["data"]
                t = min(self.time_step, data_vol.shape[0]-1)
                data_slice = np.transpose(data_vol[t])
                
                self.imgs[idx].set_data(data_slice)
                
                # Auto Scale
                is_sym = (idx == 1) # Field is symmetric
                vmin, vmax = self.get_auto_clim(data_slice, symmetric=is_sym)
                self.imgs[idx].set_clim(vmin, vmax)
                
                # Label
                unit = "a0" if idx == 1 else ("n/nc" if idx == 2 else "arb")
                self.cbars[idx].set_label(f"{self.state[idx]['label']} ({unit})")
                title_parts.append(self.state[idx]['label'])

        self.ax.set_title(" | ".join(title_parts))
        self.fig.canvas.draw_idle()

    def rotate_pool(self, pool_idx, direction):
        pool = getattr(self, f"pool{pool_idx}")
        meta = getattr(self, f"pool{pool_idx}_meta")
        
        if pool:
            pool.rotate(direction)
            meta.rotate(direction)
            self.state[pool_idx]["data"] = pool[0]
            self.state[pool_idx]["label"] = meta[0]
            self.refresh_plot()

    def onclick(self, event):
        # Time
        if event.key == 'right':
            self.time_step += 1
            self.refresh_plot()
        elif event.key == 'left':
            self.time_step = max(0, self.time_step - 1)
            self.refresh_plot()
        elif event.key == 'shift+right':
            self.time_step += 10
            self.refresh_plot()
        elif event.key == 'shift+left':
            self.time_step = max(0, self.time_step - 10)
            self.refresh_plot()

        # Rotation
        elif event.key == 'n': self.rotate_pool(1, 1)
        elif event.key == 'm': self.rotate_pool(1, -1)
        elif event.key == 'd': self.rotate_pool(2, 1)
        elif event.key == 'a': self.rotate_pool(2, -1)
        elif event.key == 'ctrl+n': self.rotate_pool(3, 1)
        elif event.key == 'ctrl+m': self.rotate_pool(3, -1)
        
        # Visibility
        elif event.key == '1' and self.imgs[1]:
            self.imgs[1].set_visible(not self.imgs[1].get_visible())
            self.fig.canvas.draw_idle()
        elif event.key == '2' and self.imgs[2]:
            self.imgs[2].set_visible(not self.imgs[2].get_visible())
            self.fig.canvas.draw_idle()
        elif event.key == '3' and self.imgs[3]:
            self.imgs[3].set_visible(not self.imgs[3].get_visible())
            self.fig.canvas.draw_idle()


# ----------------------------
# CLI entry point
# ----------------------------
def run_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=".", help="Directory with HDF5 files")
    parser.add_argument("--lambda", dest="lambda_nm", type=float, default=1000.0, 
                        help="Laser wavelength in nm (default: 1000)")
    args = parser.parse_args()
    
    # 1. Setup Physics with user wavelength
    setup_physics(args.lambda_nm)

    # 2. Load Data (Normalization happens here now)
    print(f"Loading data from {args.dir}...")
    loaded_data = {}
    
    for key, meta in VARIABLES.items():
        data = probe_hdf5_variable(
            args.dir, meta["file"], meta["dataset"], key, verbose=True
        )
        if data is not None:
            loaded_data[key] = data

    if not loaded_data:
        print("Error: No data found.")
        return

    # 3. Special Processing
    xye_arrays, xye_labels = [], []
    if "xye" in loaded_data:
        xye_arrays, xye_labels = prepare_xye_arrays(loaded_data["xye"])
        del loaded_data["xye"] 

    # 4. Launch
    DianaInteractive(loaded_data, xye_arrays, xye_labels)

if __name__ == "__main__":
    run_cli()