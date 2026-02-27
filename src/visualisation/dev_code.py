#!/usr/bin/env python3
"""
diana_visualisation.py
Auto-Deck Parsing & Dual Viewer (Static / Moving Window)
Task 2 Update: Added species energy rotation (hotkeys 'o'/'p') and axis units.
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

    colors2 = [(0.05 + 0.15*(c+1)/2, 0.4 + 0.6*(c+1)/2, 0.01, (c+1)*0.25) for c in np.linspace(-1, 1, 200)]
    cmap_green = LinearSegmentedColormap.from_list('mycmap2', colors2, N=200)

    cmapB = plt.cm.coolwarm
    cmap_fields = ListedColormap(cmapB(np.arange(cmapB.N)))

    return cmap_density, cmap_green, cmap_fields

# ----------------------------
# Smart Deck Parser & Loaders
# ----------------------------
def parse_input_deck(base_dir):
    parent_dir = os.path.dirname(os.path.abspath(base_dir))
    deck_path = os.path.join(parent_dir, "sdf_files", "input.deck")
    if not os.path.exists(deck_path): return {}
    raw_vars = {}
    current_block = None
    is_normal_output = False
    normal_dt_raw = None
    constant_lambda_raw = None
    laser_lambda_raw = None
    with open(deck_path, 'r') as f:
        for line in f:
            line = line.split('!')[0].strip()
            if not line: continue
            if line.lower().startswith('begin:'):
                current_block = line.split(':')[1].strip().lower()
                continue
            if line.lower().startswith('end:'):
                current_block = None
                continue
            if '=' not in line: continue
            key, val = [x.strip().lower() for x in line.split('=', 1)]
            if key == 'move_window':
                raw_vars['move_window'] = 'True' if val.strip('. "\'').lower() in ['t', 'true'] else 'False'
                continue
            if current_block == 'output' and key == 'name' and val.strip('"\'') == 'normal': is_normal_output = True
            if key == 'dt_snapshot' and current_block == 'output' and is_normal_output: normal_dt_raw = val
            elif key == 'lambda':
                if current_block == 'constant': constant_lambda_raw = val
                elif current_block == 'laser': laser_lambda_raw = val
            else: raw_vars[key] = val
    if normal_dt_raw: raw_vars['normal_dt_snapshot'] = normal_dt_raw
    if constant_lambda_raw: raw_vars['lambda'] = constant_lambda_raw
    elif laser_lambda_raw and laser_lambda_raw != 'lambda': raw_vars['lambda'] = laser_lambda_raw
    resolved_vars = {'c': 299792458.0, 'femto': 1e-15, 'pico': 1e-12, 'nano': 1e-9, 'micro': 1e-6, 'micron': 1e-6, 'microns':1e-6,'milli': 1e-3}
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
            except: pass 
        if not progress: break
    params = {'move_window': resolved_vars.get('move_window', False)}
    if 'lambda' in resolved_vars: params['lambda_nm'] = resolved_vars['lambda'] * 1e9
    if 'window_v_x' in resolved_vars: params['window_v_x'] = resolved_vars['window_v_x']
    if 'normal_dt_snapshot' in resolved_vars: params['dt'] = resolved_vars['normal_dt_snapshot']
    for k in ['x_min', 'x_max', 'y_min', 'y_max']:
        if k in resolved_vars: params[k] = resolved_vars[k]
    if 'window_start' in resolved_vars: params['window_start'] = resolved_vars['window_start']
    return params

def load_and_downsample(filepath, dataset_name, label, norm_key, stride):
    try:
        with h5py.File(filepath, 'r') as f:
            dset = f[dataset_name] if dataset_name in f else f[list(f.keys())[0]]
            data = dset[:, ::stride, ::stride].astype(np.float32)
            factors = CONSTANTS.get("NORM_FACTORS", {})
            if norm_key in factors: data /= factors[norm_key]
            return data
    except: return None

def load_xye_sum(filepath, dataset_name, stride):
    try:
        with h5py.File(filepath, 'r') as f:
            dset = f[dataset_name]
            raw_data = dset[:, ::stride, ::stride, 2:] 
            return np.sum(raw_data, axis=-1).astype(np.float32)
    except: return None

VARIABLES = {
    "Jx": {"file": "Jx.hdf5", "dataset": "Jx", "pool": 1},
    "Jy": {"file": "Jy.hdf5", "dataset": "Jy", "pool": 1},
    "Bz": {"file": "Bz.hdf5", "dataset": "Bz", "pool": 1},
    "Ex": {"file": "Ex.hdf5", "dataset": "Ex", "pool": 1},
    "Ey": {"file": "Ey.hdf5", "dataset": "Ey", "pool": 1},
    "ne": {"file": "n_e.hdf5", "dataset": "ne", "pool": 2}, 
    "n_photon": {"file": "n_photon.hdf5", "dataset": "n_photon", "pool": 2},
    "poynt_x":  {"file": "poynt_x.hdf5",  "dataset": "poynt_x", "pool": 3},
    "xye":      {"file": "x_y_Ekin.hdf5", "dataset": "xy_Ekin", "pool": 3},
}

# ----------------------------
# Interactive Viewers
# ----------------------------
class DianaInteractiveStatic:
    def __init__(self, pools, meta=None, time_step=0):
        self.pools, self.meta, self.time_step = pools, meta, time_step
        self.pool1_meta = deque(k for k in pools[1].keys())
        self.pool2_meta = deque(k for k in pools[2].keys())
        self.pool3_meta = deque(k for k in pools[3].keys())
        self.sel = {1: self.pool1_meta[0] if self.pool1_meta else None,
                    2: self.pool2_meta[0] if self.pool2_meta else None,
                    3: self.pool3_meta[0] if self.pool3_meta else None}

        # Energy species rotation queue
        self.energy_keys = deque(["total_energy_particle"])
        if self.meta:
            for species in ["electron", "ion", "photon", "positron"]:
                if f"total_energy_{species}" in self.meta:
                    self.energy_keys.append(f"total_energy_{species}")

        self.cmap_density, self.cmap_green, self.cmap_fields = setup_colormaps()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.imgs, self.cbars = {1: None, 2: None, 3: None}, {1: None, 2: None, 3: None}

        initial_extent = self.get_extent(0)
        
        for idx, cmap, alpha in [(1, self.cmap_fields, 1.0), (2, self.cmap_density, 0.6), (3, self.cmap_green, 1.0)]:
            if self.sel[idx]:
                self.imgs[idx] = self.ax.imshow(np.zeros((10,10)), cmap=cmap, origin='lower', aspect='auto', alpha=alpha, extent=initial_extent)
                self.cbars[idx] = plt.colorbar(self.imgs[idx], ax=self.ax, fraction=0.046, pad=0.04)

        self.fig.canvas.mpl_connect('key_press_event', self.onclick)
        self.refresh_plot()
        plt.show()

    def get_extent(self, time_idx):
        if self.meta:
            t_meta = min(time_idx, len(self.meta['extents']) - 1)
            e = self.meta['extents'][t_meta]
            return [e[0], e[2], e[1], e[3]]
        return None

    def get_auto_clim(self, data, symmetric=False):
        flat = data.flatten()
        if flat.size > 500000: flat = flat[::5]
        vmin, vmax = np.percentile(flat, [1, 99])
        if symmetric:
            m = max(abs(vmin), abs(vmax)); return -m, m
        if vmin == vmax: vmax += 1e-10 
        return vmin, vmax

    def refresh_plot(self):
        title_parts, scalar_str = [], ""
        current_extent = self.get_extent(self.time_step)
        
        if self.meta:
            t_meta = min(self.time_step, len(self.meta['times']) - 1)
            title_parts.append(f"T={self.meta['times'][t_meta]:.1f} fs (idx={self.time_step})")
            
            # --- Dynamic Species Energy Display ---
            if 'laser_en_total' in self.meta:
                l_en, a_fr = self.meta['laser_en_total'][t_meta], self.meta['abs_frac'][t_meta]
                f_en = self.meta['total_energy_field'][t_meta]
                
                curr_energy_key = self.energy_keys[0]
                p_en = self.meta.get(curr_energy_key, [0])[t_meta]
                
                # Format label (e.g., 'total_energy_electron' -> 'Electron')
                e_lbl = "Part" if curr_energy_key == "total_energy_particle" else curr_energy_key.split('_')[-1].capitalize()
                
                scalar_str = f"Laser: {l_en:.2e} J  |  Abs: {a_fr:.3f}  |  {e_lbl}: {p_en:.2e} J  |  Field: {f_en:.2e} J"
        else: title_parts.append(f"T-idx={self.time_step}")

        for idx in [1, 2, 3]:
            label = self.sel[idx]
            if self.imgs[idx] and label:
                full_data = self.pools[idx][label]
                t = min(self.time_step, full_data.shape[0]-1)
                data_slice = np.transpose(full_data[t])
                self.imgs[idx].set_data(data_slice)
                if current_extent: self.imgs[idx].set_extent(current_extent)
                vmin, vmax = self.get_auto_clim(data_slice, symmetric=(idx == 1))
                self.imgs[idx].set_clim(vmin, vmax)
                unit = "a0" if idx == 1 else ("n/nc" if idx == 2 else "arb")
                self.cbars[idx].set_label(f"{label} ({unit})")
                title_parts.append(label)

        # Update Axes Labels
        if current_extent:
            self.ax.set_xlim(current_extent[0], current_extent[1])
            self.ax.set_ylim(current_extent[2], current_extent[3])
            self.ax.set_xlabel("x [μm]")
            self.ax.set_ylabel("y [μm]")
        else:
            self.ax.set_xlabel("x [pixels]")
            self.ax.set_ylabel("y [pixels]")

        full_title = f"{scalar_str}\n" + " | ".join(title_parts) if scalar_str else " | ".join(title_parts)
        self.ax.set_title(full_title, fontsize=10)
        self.fig.canvas.draw_idle()

    def rotate_pool(self, idx, direction):
        meta = getattr(self, f"pool{idx}_meta")
        if meta: meta.rotate(direction); self.sel[idx] = meta[0]; self.refresh_plot()

    def onclick(self, event):
        if event.key == 'right': self.time_step += 1; self.refresh_plot()
        elif event.key == 'left': self.time_step = max(0, self.time_step - 1); self.refresh_plot()
        elif event.key == 'shift+right': self.time_step += 10; self.refresh_plot()
        elif event.key == 'shift+left': self.time_step = max(0, self.time_step - 10); self.refresh_plot()
        elif event.key in ['n', 'm']: self.rotate_pool(1, 1 if event.key=='n' else -1)
        elif event.key in ['d', 'a']: self.rotate_pool(2, 1 if event.key=='d' else -1)
        elif event.key in ['ctrl+n', 'ctrl+m']: self.rotate_pool(3, 1 if 'n' in event.key else -1)
        # Added keys for rotating energy display
        elif event.key == 'e': self.energy_keys.rotate(-1); self.refresh_plot()
        elif event.key == 'r': self.energy_keys.rotate(1); self.refresh_plot()
        
        for i in [1,2,3]:
            if event.key == str(i) and self.imgs[i]: 
                self.imgs[i].set_visible(not self.imgs[i].get_visible()); self.fig.canvas.draw_idle()

class DianaInteractiveMoving(DianaInteractiveStatic):
    def __init__(self, pools, grid_params, meta=None, time_step=0):
        self.grid_params = grid_params
        super().__init__(pools, meta, time_step)

    def get_extent(self, time_idx):
        if self.meta: return super().get_extent(time_idx)
        p = self.grid_params
        if p.get('xmin') is None: return None
        shift = 0.0
        if p.get('v_x', 0) > 0 and p.get('dt') is not None:
            t = time_idx * p['dt']
            if t > p.get('t_start', 0): shift = p['v_x'] * (t - p['t_start'])
        scale = 1e6
        return [(p['xmin']+shift)*scale, (p['xmax']+shift)*scale, p.get('ymin',0)*scale, p.get('ymax',0)*scale]

# ----------------------------
# CLI Entry
# ----------------------------
def run_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=".", help="Directory with HDF5 files")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    meta = None
    meta_path = os.path.join(args.dir, "metadata.hdf5")
    if os.path.exists(meta_path):
        with h5py.File(meta_path, 'r') as f:
            meta = {"times": f["times"][:] * 1e15, "extents": f["extents"][:] * 1e6}
            
            # Load basic and species energies
            keys_to_load = ["laser_en_total", "abs_frac", "total_energy_particle", "total_energy_field"]
            for sp in ["electron", "ion", "photon", "positron"]: keys_to_load.append(f"total_energy_{sp}")
            
            for k in keys_to_load:
                if k in f: meta[k] = f[k][:]
    
    deck = parse_input_deck(args.dir)
    setup_physics(deck.get('lambda_nm', 1000.0))
    
    pools = {1: {}, 2: {}, 3: {}}
    for key, mv in VARIABLES.items():
        if key == "xye": continue
        path = os.path.join(args.dir, mv["file"])
        if os.path.isfile(path):
            data = load_and_downsample(path, mv["dataset"], key, key, args.stride)
            if data is not None: pools[mv["pool"]][key] = data

    path = os.path.join(args.dir, VARIABLES["xye"]["file"])
    if os.path.isfile(path):
        data = load_xye_sum(path, VARIABLES["xye"]["dataset"], args.stride)
        if data is not None: pools[3]["High_E_Sum"] = data

    if not any(pools[p] for p in pools): return
    if deck.get('move_window') or meta: DianaInteractiveMoving(pools, deck, meta=meta)
    else: DianaInteractiveStatic(pools, meta=meta)

if __name__ == "__main__":
    run_cli()