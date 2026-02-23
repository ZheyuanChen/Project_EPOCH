#!/usr/bin/env python3
"""
diana_visualisation.py
Fast Animation / Pre-Loaded Version
Uses spatial downsampling to fit large data into RAM.
"""

import os
import argparse
import sys
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
CONSTANTS = {} 

def setup_physics(lambda_nm):
    """
    Define constants and normalise stuff (EPOCH standard).
    Allows for user input for the wavelength (default 1000nm)
    The normalisation scheme is hardwired for each variable, so need to write normalisation for each newly-added variable 
    """
    lambda_m = lambda_nm * 1.0e-9
    c = 299792458.0
    me = 9.10938356e-31
    e = 1.60217663e-19
    eps0 = 8.85418781e-12
    
    omega = 2 * np.pi * c / lambda_m
    nc = eps0 * me * omega**2 / e**2
    
    print(f"Physics Setup: Lambda = {lambda_nm} nm")
    print(f"  -> Critical Density (nc) = {nc:.2e} m^-3")

    CONSTANTS["NORM_FACTORS"] = {
        "ne": nc, "n_photon": nc,
        "Ex": (me * omega * c) / e, "Ey": (me * omega * c) / e, "Ez": (me * omega * c) / e,
        "Bz": (me * omega) / e, "By": (me * omega) / e, "Bx": (me * omega) / e,
        "Jx": e * nc * c, "Jy": e * nc * c,
        "poynt_x": me * c**3 * nc, 
        "xye": 1.0 
    }

# ----------------------------
# Data Loading (The Magic Part)
# ----------------------------
def load_and_downsample(filepath, dataset_name, label, norm_key, stride):
    """
    Reads the ENTIRE dataset into RAM, but downsamples spatially to save memory.
    """
    try:
        with h5py.File(filepath, 'r') as f:
            if dataset_name in f:
                dset = f[dataset_name]
            else:
                dset = f[list(f.keys())[0]] # Fallback
            
            # shape is usually (Time, Y, X)
            # We apply stride to Y and X: [:, ::stride, ::stride]
            print(f"  ... Loading {label} ... ", end="", flush=True)
            
            # 1. Read from disk with slicing (Fast I/O)
            raw_data = dset[:, ::stride, ::stride]
            
            # 2. Convert to float32 (Saves 50% RAM compared to float64)
            data = raw_data.astype(np.float32)
            
            # 3. Normalize
            factors = CONSTANTS.get("NORM_FACTORS", {})
            if norm_key in factors:
                norm = factors[norm_key]
                if norm != 0:
                    data /= norm
            
            print(f"Done. Shape: {data.shape} ({data.nbytes / 1024**2:.1f} MB)")
            return data
            
    except Exception as e:
        print(f"\n  [Error] Failed loading {label}: {e}")
        return None

def load_xye_sum(filepath, dataset_name, stride):
    """Special loader for Kinetic Energy Sum."""
    try:
        with h5py.File(filepath, 'r') as f:
            dset = f[dataset_name]
            print(f"  ... Loading Kinetic Energy Sum ... ", end="", flush=True)
            
            # Shape: (Time, Y, X, EnergyBins)
            # Read all time, sliced space, high energy bins (idx 2 to end)
            # Note: We must sum AFTER reading to keep memory low, or read iteratively if massive.
            # For speed, we read the slice then sum.
            
            # Read: [All Times, Strided Y, Strided X, High Energy Bins]
            raw_data = dset[:, ::stride, ::stride, 2:] 
            
            # Sum over energy axis (last axis)
            data = np.sum(raw_data, axis=-1).astype(np.float32)
            
            print(f"Done. Shape: {data.shape} ({data.nbytes / 1024**2:.1f} MB)")
            return data
    except Exception as e:
        print(f"\n  [Error] Failed loading XYE: {e}")
        return None

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
# Main Visualization
# ----------------------------
class DianaInteractive:
    def __init__(self, pools, time_step=0):
        self.pools = pools 
        self.time_step = time_step
        
        # Meta lists for labels
        self.pool1_meta = deque(k for k in pools[1].keys())
        self.pool2_meta = deque(k for k in pools[2].keys())
        self.pool3_meta = deque(k for k in pools[3].keys())

        # Current Selection
        self.sel = {
            1: self.pool1_meta[0] if self.pool1_meta else None,
            2: self.pool2_meta[0] if self.pool2_meta else None,
            3: self.pool3_meta[0] if self.pool3_meta else None
        }

        # Setup Figure
        self.cmap_density, self.cmap_green, self.cmap_fields = setup_colormaps()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        self.imgs = {1: None, 2: None, 3: None}
        self.cbars = {1: None, 2: None, 3: None}

        # Initialize Image Objects with first frame
        # (We use random data just to init the object, then refresh)
        dummy = np.zeros((100,100))
        
        if self.sel[1]:
            self.imgs[1] = self.ax.imshow(dummy, cmap=self.cmap_fields, origin='lower', aspect='auto')
            self.cbars[1] = plt.colorbar(self.imgs[1], ax=self.ax, fraction=0.046, pad=0.04)

        if self.sel[2]:
            self.imgs[2] = self.ax.imshow(dummy, cmap=self.cmap_density, origin='lower', aspect='auto', alpha=0.6)
            self.cbars[2] = plt.colorbar(self.imgs[2], ax=self.ax, fraction=0.046, pad=0.04)

        if self.sel[3]:
            self.imgs[3] = self.ax.imshow(dummy, cmap=self.cmap_green, origin='lower', aspect='auto', alpha=0.5)
            self.cbars[3] = plt.colorbar(self.imgs[3], ax=self.ax, fraction=0.046, pad=0.04)

        self.fig.canvas.mpl_connect('key_press_event', self.onclick)
        
        self.refresh_plot()
        plt.show()

    def get_auto_clim(self, data, symmetric=False):
        # Since data is in RAM, this is fast now
        flat = data.flatten()
        if flat.size > 500000: flat = flat[::5] # Subsample for speed
        vmin, vmax = np.percentile(flat, [1, 99])
        
        if symmetric:
            abs_max = max(abs(vmin), abs(vmax))
            return -abs_max, abs_max
        return vmin, vmax

    def refresh_plot(self):
        title_parts = [f"T-idx={self.time_step}"]

        for idx in [1, 2, 3]:
            label = self.sel[idx]
            if self.imgs[idx] is not None and label is not None:
                # Retrieve from RAM
                full_data = self.pools[idx][label]
                
                # Bounds check
                t = min(self.time_step, full_data.shape[0]-1)
                
                # Slice and Transpose
                data_slice = np.transpose(full_data[t])
                
                self.imgs[idx].set_data(data_slice)
                
                # Auto Scale
                is_sym = (idx == 1)
                vmin, vmax = self.get_auto_clim(data_slice, symmetric=is_sym)
                self.imgs[idx].set_clim(vmin, vmax)
                
                unit = "a0" if idx == 1 else ("n/nc" if idx == 2 else "arb")
                self.cbars[idx].set_label(f"{label} ({unit})")
                title_parts.append(label)

        self.ax.set_title(" | ".join(title_parts))
        self.fig.canvas.draw_idle()

    def rotate_pool(self, idx, direction):
        meta = getattr(self, f"pool{idx}_meta")
        if meta:
            meta.rotate(direction)
            self.sel[idx] = meta[0]
            self.refresh_plot()

    def onclick(self, event):
        # Time Navigation (Fast RAM access!)
        if event.key == 'right': self.time_step += 1; self.refresh_plot()
        elif event.key == 'left': self.time_step = max(0, self.time_step - 1); self.refresh_plot()
        elif event.key == 'shift+right': self.time_step += 10; self.refresh_plot()
        elif event.key == 'shift+left': self.time_step = max(0, self.time_step - 10); self.refresh_plot()

        elif event.key == 'n': self.rotate_pool(1, 1)
        elif event.key == 'm': self.rotate_pool(1, -1)
        elif event.key == 'd': self.rotate_pool(2, 1)
        elif event.key == 'a': self.rotate_pool(2, -1)
        elif event.key == 'ctrl+n': self.rotate_pool(3, 1)
        elif event.key == 'ctrl+m': self.rotate_pool(3, -1)
        
        elif event.key == '1' and self.imgs[1]: self.imgs[1].set_visible(not self.imgs[1].get_visible()); self.fig.canvas.draw_idle()
        elif event.key == '2' and self.imgs[2]: self.imgs[2].set_visible(not self.imgs[2].get_visible()); self.fig.canvas.draw_idle()
        elif event.key == '3' and self.imgs[3]: self.imgs[3].set_visible(not self.imgs[3].get_visible()); self.fig.canvas.draw_idle()

# ----------------------------
# CLI Entry
# ----------------------------
def run_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=".", help="Directory with HDF5 files")
    parser.add_argument("--lambda", dest="lambda_nm", type=float, default=1000.0, help="Laser wavelength nm")
    parser.add_argument("--stride", type=int, default=1, help="Downsampling factor (1=Full, 2=Half, 4=Quarter). Higher=Faster/LessRAM.")
    args = parser.parse_args()
    
    setup_physics(args.lambda_nm)

    # Dictionary to hold Loaded RAM Data
    # structure: pools[pool_id][label] = numpy_array
    pools = {1: {}, 2: {}, 3: {}}
    
    print(f"--- Loading Data (Stride={args.stride}) ---")
    print("Please wait, pre-loading files into RAM...")

    # Load Standard Variables
    for key, meta in VARIABLES.items():
        if key == "xye": continue
        
        path = os.path.join(args.dir, meta["file"])
        if os.path.isfile(path):
            data = load_and_downsample(path, meta["dataset"], key, key, args.stride)
            if data is not None:
                pools[meta["pool"]][key] = data

    # Load XYE Special Case
    xye_meta = VARIABLES["xye"]
    path = os.path.join(args.dir, xye_meta["file"])
    if os.path.isfile(path):
        data = load_xye_sum(path, xye_meta["dataset"], args.stride)
        if data is not None:
            pools[3]["High_E_Sum"] = data

    if not any(pools[p] for p in pools):
        print("Error: No data loaded.")
        return

    print("--- Initialization Complete. Starting Viewer ---")
    DianaInteractive(pools)

if __name__ == "__main__":
    run_cli()