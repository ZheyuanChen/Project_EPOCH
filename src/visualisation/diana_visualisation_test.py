#!/usr/bin/env python3
"""
diana_visualisation.py
Auto-Deck Parsing & Dual Viewer (Static / Moving Window)
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
# Global Physics Setup
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
# Smart Deck Parser
# ----------------------------
def parse_input_deck(base_dir):
    parent_dir = os.path.dirname(os.path.abspath(base_dir))
    deck_path = os.path.join(parent_dir, "sdf_files", "input.deck")
    
    if not os.path.exists(deck_path):
        print(f"  [Parser] No input.deck found at {deck_path}")
        return {}

    print(f"  [Parser] Found input.deck at {deck_path}. Analyzing equations...")
    
    raw_vars = {}
    current_block = None
    is_normal_output = False
    normal_dt_raw = None

    with open(deck_path, 'r') as f:
        for line in f:
            line = line.split('!')[0].strip() 
            if not line: continue
            
            if line.lower().startswith('begin:'):
                current_block = line.split(':')[1].strip().lower()
                is_normal_output = False
                continue
            if line.lower().startswith('end:'):
                current_block = None
                is_normal_output = False
                continue

            if '=' not in line: continue
            key, val = [x.strip().lower() for x in line.split('=', 1)]
            
            if key == 'move_window':
                val_clean = val.strip('. "\'').lower()
                raw_vars['move_window'] = 'True' if val_clean in ['t', 'true'] else 'False'
                continue

            if current_block == 'output' and key == 'name' and val.strip('"\'') == 'normal':
                is_normal_output = True
                
            if key == 'dt_snapshot':
                if current_block == 'output' and is_normal_output:
                    normal_dt_raw = val
            else:
                raw_vars[key] = val

    if normal_dt_raw:
        raw_vars['normal_dt_snapshot'] = normal_dt_raw

    resolved_vars = {
        'c': 299792458.0, 'femto': 1e-15, 'pico': 1e-12, 
        'nano': 1e-9, 'micro': 1e-6, 'micron': 1e-6, 'milli': 1e-3
    }
    
    unresolved = raw_vars.copy()
    
    for _ in range(10):
        progress = False
        for k, expr in list(unresolved.items()):
            current_expr = expr
            for res_k, res_v in sorted(resolved_vars.items(), key=lambda x: len(x[0]), reverse=True):
                current_expr = re.sub(rf'\b{res_k}\b', f"({res_v})", current_expr)
            try:
                val = eval(current_expr, {"__builtins__": None}, {})
                resolved_vars[k] = val
                del unresolved[k]
                progress = True
            except Exception:
                pass 
                
        if not progress and not unresolved: break

    params = {}
    params['move_window'] = resolved_vars.get('move_window', False)

    if 'lambda' in resolved_vars: params['lambda_nm'] = resolved_vars['lambda'] * 1e9
    if 'window_v_x' in resolved_vars: params['window_v_x'] = resolved_vars['window_v_x']
    if 'normal_dt_snapshot' in resolved_vars: params['dt'] = resolved_vars['normal_dt_snapshot']
    
    for k in ['x_min', 'x_max', 'y_min', 'y_max']:
        if k in resolved_vars: params[k] = resolved_vars[k]

    if 'window_start' in resolved_vars:
        params['window_start'] = resolved_vars['window_start']
    elif 'window_start_time' in resolved_vars:
        params['window_start'] = resolved_vars['window_start_time']

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
            
            print(f"Done. Shape: {data.shape}")
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
            print(f"Done. Shape: {data.shape}")
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

# ----------------------------
# Interactive Viewers
# ----------------------------
class DianaInteractiveStatic:
    """Standard viewer for static boxes (no window shifting)"""
    def __init__(self, pools, time_step=0):
        print("  [Viewer] Initializing Standard Static Viewer...")
        self.pools = pools 
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
        
        if self.sel[1]:
            self.imgs[1] = self.ax.imshow(dummy, cmap=self.cmap_fields, origin='lower', aspect='auto')
            self.cbars[1] = plt.colorbar(self.imgs[1], ax=self.ax, fraction=0.046, pad=0.04)
        if self.sel[2]:
            self.imgs[2] = self.ax.imshow(dummy, cmap=self.cmap_density, origin='lower', aspect='auto', alpha=0.6)
            self.cbars[2] = plt.colorbar(self.imgs[2], ax=self.ax, fraction=0.046, pad=0.04)
        if self.sel[3]:
            self.imgs[3] = self.ax.imshow(dummy, cmap=self.cmap_green, origin='lower', aspect='auto', alpha=0.5)
            self.cbars[3] = plt.colorbar(self.imgs[3], ax=self.ax, fraction=0.046, pad=0.04)

        self.ax.set_xlabel("x [pixels]")
        self.ax.set_ylabel("y [pixels]")
        
        self.fig.canvas.mpl_connect('key_press_event', self.onclick)
        self.refresh_plot()
        plt.show()

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


class DianaInteractiveMoving:
    """Viewer calculating and shifting extents dynamically."""
    def __init__(self, pools, grid_params, time_step=0):
        print("  [Viewer] Initializing Moving Window Viewer...")
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
        if p.get('xmin') is None or p.get('xmax') is None: 
            return None
            
        shift = 0.0
        if p.get('v_x', 0) > 0 and p.get('dt') is not None:
            t = time_idx * p['dt']
            if t > p.get('t_start', 0):
                shift = p['v_x'] * (t - p['t_start'])
            
        scale = 1e6 # Convert to micrometers
        y_min_val = p.get('ymin', 0.0)
        y_max_val = p.get('ymax', 0.0)
        
        return [
            (p['xmin'] + shift) * scale, (p['xmax'] + shift) * scale,
            y_min_val * scale, y_max_val * scale
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

        t_safe = min(self.time_step, self.pools[1][self.sel[1]].shape[0]-1) if self.sel[1] else self.time_step
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
    
    parser.add_argument("--lambda", dest="lambda_nm", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--window-v-x", type=float, default=None)
    parser.add_argument("--window-start", type=float, default=None)
    args = parser.parse_args()
    
    deck = parse_input_deck(args.dir)
    move_window = deck.get('move_window', False)
    
    # 1. Core Physics (Lambda)
    lambda_nm = args.lambda_nm if args.lambda_nm is not None else deck.get('lambda_nm')
    if lambda_nm is None:
        val = input("  [Manual Input] Wavelength (lambda) not found. Enter in nm (default 1000): ").strip()
        lambda_nm = float(val) if val else 1000.0
    setup_physics(lambda_nm)
    
    # 2. Extract or Prompt Grid Parameters (Only demanded if move_window is True)
    grid_params = {}
    if move_window:
        print("\n  [Info] Moving window is ACTIVE in input.deck.")
        v_x = args.window_v_x if args.window_v_x is not None else deck.get('window_v_x')
        if v_x is None: 
            v_x = float(input("  [Manual Input] window_v_x not found. Enter in m/s (default 0.0): ").strip() or "0")

        t_start = args.window_start if args.window_start is not None else deck.get('window_start')
        if t_start is None: 
            t_start = float(input("  [Manual Input] window_start not found. Enter in seconds (default 0.0): ").strip() or "0")
        
        dt = args.dt if args.dt is not None else deck.get('dt')
        xmin = args.x_min if args.x_min is not None else deck.get('x_min')
        xmax = args.x_max if args.x_max is not None else deck.get('x_max')
        ymin = args.y_min if args.y_min is not None else deck.get('y_min')
        ymax = args.y_max if args.y_max is not None else deck.get('y_max')

        # Demand bounds & dt ONLY because the window is shifting
        if dt is None or xmin is None or xmax is None:
            print("  [Warning] Grid boundaries/dt incomplete for moving window calculation.")
            if dt is None: dt = float(input("  [Manual Input] Enter dt_snapshot in seconds: ").strip())
            if xmin is None: xmin = float(input("  [Manual Input] Enter x_min in meters: ").strip())
            if xmax is None: xmax = float(input("  [Manual Input] Enter x_max in meters: ").strip())
            if ymin is None: ymin = float(input("  [Manual Input] Enter y_min in meters (or 0): ").strip() or "0")
            if ymax is None: ymax = float(input("  [Manual Input] Enter y_max in meters (or 0): ").strip() or "0")

        grid_params = {'dt': dt, 'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'v_x': v_x, 't_start': t_start}
    else:
        print("\n  [Info] Moving window is INACTIVE (or undefined). Using standard static viewer.")

    # 3. Load Datasets
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
        print("Error: No data loaded. Check directory path and HDF5 files.")
        return

    # 4. Launch Appropriate Viewer
    print("--- Initialization Complete. Starting Viewer ---")
    if move_window:
        DianaInteractiveMoving(pools, grid_params)
    else:
        DianaInteractiveStatic(pools)

if __name__ == "__main__":
    run_cli()