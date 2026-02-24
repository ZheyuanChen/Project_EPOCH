#!/usr/bin/env python3
"""
diana_visualisation.py
Fast Animation / Pre-Loaded Version with Auto-Deck Parsing & Moving Window
"""

import os
import argparse
import sys
import re
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
# Input Deck Parser
# ----------------------------
def eval_epoch_math(expr):
    """Safely evaluates EPOCH mathematical strings like '340 * femto' or 'c * 0.87'"""
    expr = expr.lower()
    replacements = {
        r'\bc\b': '299792458.0',
        r'\bfemto\b': '1e-15',
        r'\bpico\b': '1e-12',
        r'\bnano\b': '1e-9',
        r'\bmicro\b': '1e-6',
        r'\bmicron\b': '1e-6',
        r'\bmilli\b': '1e-3'
    }
    for key, val in replacements.items():
        expr = re.sub(key, val, expr)
    
    try:
        # Evaluate the math safely (prevents malicious code execution)
        return eval(expr, {"__builtins__": None}, {})
    except Exception:
        return None

def parse_input_deck(base_dir):
    """Looks for input.deck and extracts lambda, window velocity, and start time."""
    # Assuming dir is .../sth/hdf5_output, input.deck is in .../sth/sdf_files/input.deck
    parent_dir = os.path.dirname(os.path.abspath(base_dir))
    deck_path = os.path.join(parent_dir, "sdf_files", "input.deck")
    
    params = {}
    if not os.path.exists(deck_path):
        print(f"  [Parser] No input.deck found at {deck_path}")
        return params

    print(f"  [Parser] Found input.deck at {deck_path}. Scanning...")
    with open(deck_path, 'r') as f:
        for line in f:
            # Strip out comments and whitespace
            line = line.split('!')[0].strip()
            if '=' not in line:
                continue
                
            key, val = [x.strip().lower() for x in line.split('=', 1)]
            
            # Match keywords
            if key in ['lambda', 'window_v_x', 'window_start_time', 'win_start']:
                num_val = eval_epoch_math(val)
                if num_val is not None:
                    if key == 'lambda':
                        params['lambda_nm'] = num_val * 1e9 # Convert m to nm
                    elif key == 'window_v_x':
                        params['window_v_x'] = num_val
                    elif key in ['window_start_time', 'win_start']:
                        params['window_start'] = num_val
                        
    return params

# ----------------------------
# Data Loading 
# ----------------------------
def load_and_downsample(filepath, dataset_name, label, norm_key, stride):
    try:
        with h5py.File(filepath, 'r') as f:
            dset = f[dataset_name] if dataset_name in f else f[list(f.keys())[0]]
            print(f"  ... Loading {label} ... ", end="", flush=True)
            raw_data = dset[:, ::stride, ::stride]
            data = raw_data.astype(np.float32)
            
            factors = CONSTANTS.get("NORM_FACTORS", {})
            if norm_key in factors and factors[norm_key] != 0:
                data /= factors[norm_key]
            
            print(f"Done. Shape: {data.shape} ({data.nbytes / 1024**2:.1f} MB)")
            return data
    except Exception as e:
        print(f"\n  [Error] Failed loading {label}: {e}")
        return None

def load_xye_sum(filepath, dataset_name, stride):
    try:
        with h5py.File(filepath, 'r') as f:
            dset = f[dataset_name]
            print(f"  ... Loading Kinetic Energy Sum ... ", end="", flush=True)
            raw_data = dset[:, ::stride, ::stride, 2:] 
            data = np.sum(raw_data, axis=-1).astype(np.float32)
            print(f"Done. Shape: {data.shape} ({data.nbytes / 1024**2:.1f} MB)")
            return data
    except Exception as e:
        return None

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
    cmap1 = plt.cm.gist_yarg
    my_cmap = cmap1(np.arange(cmap1.N))
    my_cmap[:, -1] = 0.8
    cmap_density = ListedColormap(my_cmap)

    colors2 = [(0.2, 1, 0.01, (c+1)*0.5) for c in np.linspace(-1, 1, 200)]
    cmap_green = LinearSegmentedColormap.from_list('mycmap2', colors2, N=200)

    cmapB = plt.cm.coolwarm
    cmap_fields = ListedColormap(cmapB(np.arange(cmapB.N)))

    return cmap_density, cmap_green, cmap_fields

# ----------------------------
# Main Visualization
# ----------------------------
class DianaInteractive:
    def __init__(self, pools, grid_params, time_step=0):
        self.pools = pools 
        self.grid_params = grid_params
        self.time_step = time_step
        
        self.pool1_meta = deque(k for k in pools[1].keys())
        self.pool2_meta = deque(k for k in pools[2].keys())
        self.pool3_meta = deque(k for k in pools[3].keys())

        self.sel = {
            1: self.pool1_meta[0] if self.pool1_meta else None,
            2: self.pool2_meta[0] if self.pool2_meta else None,
            3: self.pool3_meta[0] if self.pool3_meta else None
        }

        self.cmap_density, self.cmap_green, self.cmap_fields = setup_colormaps()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        self.imgs = {1: None, 2: None, 3: None}
        self.cbars = {1: None, 2: None, 3: None}

        dummy = np.zeros((100,100))
        initial_extent = self.get_extent(0)
        
        if self.sel[1]:
            self.imgs[1] = self.ax.imshow(dummy, cmap=self.cmap_fields, origin='lower', aspect='auto', extent=initial_extent)
            self.cbars[1] = plt.colorbar(self.imgs[1], ax=self.ax, fraction=0.046, pad=0.04)
        if self.sel[2]:
            self.imgs[2] = self.ax.imshow(dummy, cmap=self.cmap_density, origin='lower', aspect='auto', alpha=0.6, extent=initial_extent)
            self.cbars[2] = plt.colorbar(self.imgs[2], ax=self.ax, fraction=0.046, pad=0.04)
        if self.sel[3]:
            self.imgs[3] = self.ax.imshow(dummy, cmap=self.cmap_green, origin='lower', aspect='auto', alpha=0.5, extent=initial_extent)
            self.cbars[3] = plt.colorbar(self.imgs[3], ax=self.ax, fraction=0.046, pad=0.04)

        self.fig.canvas.mpl_connect('key_press_event', self.onclick)
        self.refresh_plot()
        plt.show()

    def get_extent(self, time_idx):
        p = self.grid_params
        # Check if the grid was actually defined (not None)
        if p['dt'] is None or p['xmin'] is None or p['xmax'] is None: 
            return None
            
        t = time_idx * p['dt']
        shift = 0.0
        
        if t > p['t_start']:
            shift = p['v_x'] * (t - p['t_start'])
            
        scale = 1e6 # Convert to micrometers
        return [
            (p['xmin'] + shift) * scale, (p['xmax'] + shift) * scale,
            p['ymin'] * scale, p['ymax'] * scale
        ]

    def get_auto_clim(self, data, symmetric=False):
        flat = data.flatten()
        if flat.size > 500000: flat = flat[::5]
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
                full_data = self.pools[idx][label]
                t = min(self.time_step, full_data.shape[0]-1)
                
                data_slice = np.transpose(full_data[t])
                self.imgs[idx].set_data(data_slice)
                
                extent = self.get_extent(t)
                if extent is not None:
                    self.imgs[idx].set_extent(extent)
                
                is_sym = (idx == 1)
                vmin, vmax = self.get_auto_clim(data_slice, symmetric=is_sym)
                self.imgs[idx].set_clim(vmin, vmax)
                
                unit = "a0" if idx == 1 else ("n/nc" if idx == 2 else "arb")
                self.cbars[idx].set_label(f"{label} ({unit})")
                title_parts.append(label)

        # Update axes limits
        t_safe = min(self.time_step, self.pools[1][self.sel[1]].shape[0]-1)
        current_extent = self.get_extent(t_safe)
        
        if current_extent is not None:
            self.ax.set_xlim(current_extent[0], current_extent[1])
            self.ax.set_ylim(current_extent[2], current_extent[3])
            self.ax.set_xlabel("x [μm]")
            self.ax.set_ylabel("y [μm]")
        else:
            self.ax.set_xlabel("x [pixels]")
            self.ax.set_ylabel("y [pixels]")

        self.ax.set_title(" | ".join(title_parts))
        self.fig.canvas.draw_idle()

    def rotate_pool(self, idx, direction):
        meta = getattr(self, f"pool{idx}_meta")
        if meta:
            meta.rotate(direction)
            self.sel[idx] = meta[0]
            self.refresh_plot()

    def onclick(self, event):
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
# CLI Entry & Logic Hierarchy
# ----------------------------
def run_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=".", help="Directory with HDF5 files")
    parser.add_argument("--stride", type=int, default=1, help="Downsampling factor.")
    
    # We set default=None so we can check if the user actively provided them in the CLI
    parser.add_argument("--lambda", dest="lambda_nm", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--window-v-x", type=float, default=None)
    parser.add_argument("--window-start", type=float, default=None)
    args = parser.parse_args()
    
    # 1. Parse input.deck
    deck_params = parse_input_deck(args.dir)
    
    # 2. Hierarchy Logic (Deck -> CLI -> Prompt)
    
    # LAMBDA
    lambda_nm = args.lambda_nm if args.lambda_nm is not None else deck_params.get('lambda_nm')
    if lambda_nm is None:
        val = input("  [Manual Input] Wavelength (lambda) not found. Enter value in nm (default 1000): ").strip()
        lambda_nm = float(val) if val else 1000.0
    setup_physics(lambda_nm)
    
    # WINDOW PARAMETERS
    v_x = args.window_v_x if args.window_v_x is not None else deck_params.get('window_v_x')
    if v_x is None:
        val = input("  [Manual Input] Window velocity (window_v_x) not found. Enter value in m/s (default 0.0): ").strip()
        v_x = float(val) if val else 0.0

    t_start = args.window_start if args.window_start is not None else deck_params.get('window_start')
    if t_start is None and v_x > 0:
        val = input("  [Manual Input] Window start time (win_start) not found. Enter value in seconds (default 0.0): ").strip()
        t_start = float(val) if val else 0.0
    elif t_start is None:
        t_start = 0.0
        
    # GRID PARAMETERS (If moving window is active, we MUST have grid params to calculate extent)
    dt = args.dt
    xmin, xmax = args.x_min, args.x_max
    ymin, ymax = args.y_min, args.y_max
    
    if v_x > 0 and (dt is None or xmin is None):
        print("\n  [Warning] Moving window detected, but physical grid size/timestep missing.")
        val_dt = input("  [Manual Input] Enter timestep (dt) in seconds (e.g. 1e-14): ").strip()
        dt = float(val_dt) if val_dt else None
        if dt is not None:
            xmin = float(input("  [Manual Input] Enter x_min in meters (e.g. 0): ").strip() or "0")
            xmax = float(input("  [Manual Input] Enter x_max in meters: ").strip() or "0")
            ymin = float(input("  [Manual Input] Enter y_min in meters: ").strip() or "0")
            ymax = float(input("  [Manual Input] Enter y_max in meters: ").strip() or "0")

    grid_params = {
        'dt': dt, 'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax,
        'v_x': v_x, 't_start': t_start
    }

    pools = {1: {}, 2: {}, 3: {}}
    print(f"\n--- Loading Data (Stride={args.stride}) ---")
    for key, meta in VARIABLES.items():
        if key == "xye": continue
        path = os.path.join(args.dir, meta["file"])
        if os.path.isfile(path):
            data = load_and_downsample(path, meta["dataset"], key, key, args.stride)
            if data is not None: pools[meta["pool"]][key] = data

    xye_meta = VARIABLES["xye"]
    path = os.path.join(args.dir, xye_meta["file"])
    if os.path.isfile(path):
        data = load_xye_sum(path, xye_meta["dataset"], args.stride)
        if data is not None: pools[3]["High_E_Sum"] = data

    if not any(pools[p] for p in pools):
        print("Error: No data loaded.")
        return

    print("--- Initialization Complete. Starting Viewer ---")
    DianaInteractive(pools, grid_params)

if __name__ == "__main__":
    run_cli()