#!/usr/bin/env python3
import os
import argparse
import sys  # noqa: F401
import re
from collections import deque
import numpy as np
import h5py
import matplotlib

matplotlib.use("TkAgg")
# matplotlib.use("QtAgg") # This may work on Windows
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# ----------------------------
# Global Physics Setup
# ----------------------------
CONSTANTS = {}


def setup_physics(lambda_nm):
    """
    Set up the physical constants and normalisation factors based on the input wavelength (in nm).
    This is crucial for ensuring that the visualisation is in meaningful units.
    The normalisation factors are used to convert raw simulation data into physical units, making it easier to interpret the results.
    """
    lambda_m = lambda_nm * 1.0e-9
    c = 299792458.0
    me = 9.10938356e-31
    e = 1.60217663e-19
    eps0 = 8.85418781e-12
    omega = 2 * np.pi * c / lambda_m
    nc = eps0 * me * omega**2 / e**2
    joule_to_mev = 1.60218e-13

    # About energy normalisation: the normalisation factor for energy density is U_norm = n_c m_e c^2.
    # About normalisation factor: the thing after each variable is one by which you want to divide the variable. e.g. n_normlised = n / nc
    CONSTANTS["NORM_FACTORS"] = {
        "ne": nc,
        "n_photon": nc,
        "Ex": (me * omega * c) / e,
        "Ey": (me * omega * c) / e,
        "Ez": (me * omega * c) / e,
        "Bz": (me * omega) / e,
        "By": (me * omega) / e,
        "Bx": (me * omega) / e,
        "Jx": e * nc * c,
        "Jy": e * nc * c,
        "Jz": e * nc * c,
        "poynt_x": me * c**3 * nc,
        "poynt_y": me * c**3 * nc,
        "poynt_z": me * c**3 * nc,
        "ekbar": joule_to_mev,
        "ekbar_electron": joule_to_mev,
        "ekbar_photon": joule_to_mev,
        "ekbar_ion": joule_to_mev,
        "ekbar_positron": joule_to_mev,  # This assumes the energy is in Joules and converts to MeV.
    }


def setup_colormaps():
    """
    Set up custom colormaps for different visualisation purposes. Only God and Gemini know why these specific colormaps and alpha values are chosen, but they seem to work well for visualising the respective data types.
    Returns:
        tuple: A tuple containing the colormaps for density, green, fields, and energy distribution.
    """
    # Pool 2: Density (Yarg)
    cmap1 = plt.cm.gist_yarg  # type: ignore
    my_cmap = cmap1(np.arange(cmap1.N))
    my_cmap[:, -1] = 0.8
    cmap_density = ListedColormap(my_cmap)

    # Pool 3: Green
    colors2 = [
        (0.05 + 0.15 * (c + 1) / 2, 0.4 + 0.6 * (c + 1) / 2, 0.01, (c + 1) * 0.25)
        for c in np.linspace(-1, 1, 200)
    ]
    cmap_green = LinearSegmentedColormap.from_list("mycmap2", colors2, N=200)

    # Pool 1: Fields (Coolwarm)
    cmap_fields = plt.cm.coolwarm  # type: ignore

    # Pool 4: Energy Distribution (Inferno with transparent zeros)
    cmap_dist_base = plt.cm.inferno(np.arange(plt.cm.inferno.N))  # type: ignore
    cmap_dist_base[0, -1] = 0  # Make zero-value pixels fully transparent
    cmap_dist = ListedColormap(cmap_dist_base)

    return cmap_density, cmap_green, cmap_fields, cmap_dist


# ----------------------------
# Smart Deck Parser & Loaders
# ----------------------------
def parse_input_deck(base_dir):
    """
    This can parse the input.deck file to extract relevant parameters for visualisation, such as whether the simulation uses a moving window, the laser wavelength, and the spatial extents of the simulation.
    Note that the current code doesn't really need most of these parameters. (Legacy issue). But I chose to parse them anyway, just in case I want to use them for something in the future.
    If the input.deck file is not found or if certain parameters are missing, the function will return an empty dictionary or a dictionary with only the successfully parsed parameters.
    In this case, it allows the visualisation code to run with default settings in the absence of specific parameters.
    """

    parent_dir = os.path.dirname(os.path.abspath(base_dir))
    deck_path = os.path.join(parent_dir, "sdf_files", "input.deck")
    if not os.path.exists(deck_path):
        deck_path = os.path.join(parent_dir, "input.deck")
    if not os.path.exists(deck_path):
        return {}  # Check two locations for input.deck: parent_dir/sdf_files/input.deck and parent_dir/input.deck. If not found, return empty dict.

    raw_vars = {}
    current_block = None
    is_normal_output = False  # This one is special. In EPOCH we sometimes have two (or more?) output blocks
    # One for normal output. We want the dt_snapshot from the normal block. So we have to parse dt_snapshot in that block only.
    normal_dt_raw = None
    constant_lambda_raw = None
    laser_lambda_raw = None
    with open(deck_path, "r") as f:
        for line in f:
            line = line.split("!")[0].strip()
            if not line:
                continue
            if line.lower().startswith("begin:"):
                current_block = (
                    line.split(":")[1].strip().lower()
                )  # Check the beginning of a block. E.g. "Begin: Output" -> current_block = "output". This is useful for parsing parameters that are only relevant within certain blocks.
                continue
            if line.lower().startswith("end:"):
                current_block = None
                continue
            if "=" not in line:
                continue
            key, val = [x.strip().lower() for x in line.split("=", 1)]
            if key == "move_window":
                raw_vars["move_window"] = (
                    "True" if val.strip(". \"'").lower() in ["t", "true"] else "False"
                )
                continue
            if (
                current_block == "output"
                and key == "name"
                and val.strip("\"'") == "normal"
            ):
                is_normal_output = True
            if key == "dt_snapshot" and current_block == "output" and is_normal_output:
                normal_dt_raw = val
            elif key == "lambda":
                if current_block == "constant":
                    constant_lambda_raw = val
                elif current_block == "laser":
                    laser_lambda_raw = val
            else:
                raw_vars[key] = val
    if normal_dt_raw:
        raw_vars["normal_dt_snapshot"] = normal_dt_raw
    if constant_lambda_raw:
        raw_vars["lambda"] = constant_lambda_raw
    elif laser_lambda_raw and laser_lambda_raw != "lambda":
        raw_vars["lambda"] = laser_lambda_raw
    resolved_vars = {
        "c": 299792458.0,
        "femto": 1e-15,
        "pico": 1e-12,
        "nano": 1e-9,
        "micro": 1e-6,
        "micron": 1e-6,
        "microns": 1e-6,
        "milli": 1e-3,
    }
    unresolved = raw_vars.copy()
    for _ in range(10):
        progress = False
        for k, expr in list(unresolved.items()):
            current_expr = expr
            for res_k, res_v in sorted(
                resolved_vars.items(), key=lambda x: len(x[0]), reverse=True
            ):
                current_expr = re.sub(rf"\b{res_k}\b", f"({res_v})", current_expr)
            try:
                val = eval(current_expr, {"__builtins__": None}, {})
                resolved_vars[k] = val
                del unresolved[k]
                progress = True
            except:  # noqa: E722
                pass
        if not progress:
            break
    params = {"move_window": resolved_vars.get("move_window", False)}
    if "lambda" in resolved_vars:
        params["lambda_nm"] = resolved_vars["lambda"] * 1e9
    if "window_v_x" in resolved_vars:
        params["window_v_x"] = resolved_vars["window_v_x"]
    if "normal_dt_snapshot" in resolved_vars:
        params["dt"] = resolved_vars["normal_dt_snapshot"]
    for k in ["x_min", "x_max", "y_min", "y_max"]:
        if k in resolved_vars:
            params[k] = resolved_vars[k]
    if "window_start" in resolved_vars:
        params["window_start"] = resolved_vars["window_start"]
    return params


def load_and_downsample(filepath, dataset_name, label, norm_key, stride):
    """
    Loads a dataset from an HDF5 file and downsamples it by the given stride. If stride > 1, it takes every nth point along the spatial dimensions to reduce the resolution for faster visualization.
    It also applies the appropriate normalisation factor based on the label of the variable.
    """
    try:
        with h5py.File(filepath, "r") as f:
            dset = f[dataset_name] if dataset_name in f else f[list(f.keys())[0]]
            if dset.ndim == 4:  # type: ignore
                data = dset[:, ::stride, ::stride, :].astype(np.float32)  # type: ignore
            else:
                data = dset[:, ::stride, ::stride].astype(np.float32)  # type: ignore
            factors = CONSTANTS.get("NORM_FACTORS", {})
            if norm_key in factors:
                data /= factors[norm_key]
            return data
    except:  # noqa: E722
        return None  # noqa: E701, E722


VARIABLES = {
    "Jx": {"file": "Jx.hdf5", "dataset": "Jx", "pool": 1},
    "Jy": {"file": "Jy.hdf5", "dataset": "Jy", "pool": 1},
    "Jz": {"file": "Jz.hdf5", "dataset": "Jz", "pool": 1},
    "Bx": {"file": "Bx.hdf5", "dataset": "Bx", "pool": 1},
    "By": {"file": "By.hdf5", "dataset": "By", "pool": 1},
    "Bz": {"file": "Bz.hdf5", "dataset": "Bz", "pool": 1},
    "Ex": {"file": "Ex.hdf5", "dataset": "Ex", "pool": 1},
    "Ey": {"file": "Ey.hdf5", "dataset": "Ey", "pool": 1},
    "Ez": {"file": "Ez.hdf5", "dataset": "Ez", "pool": 1},
    "ne": {
        "file": "n_e.hdf5",
        "dataset": "ne",
        "pool": 2,
    },  # Be very careful. There is an underscore in n_e.hdf5 but not in the key "ne". Legacy issue (sigh).
    "n_photon": {"file": "n_photon.hdf5", "dataset": "n_photon", "pool": 2},
    "poynt_x": {"file": "poynt_x.hdf5", "dataset": "poynt_x", "pool": 3},
    "poynt_y": {"file": "poynt_y.hdf5", "dataset": "poynt_y", "pool": 3},
    "poynt_z": {"file": "poynt_z.hdf5", "dataset": "poynt_z", "pool": 3},
    "ekbar": {"file": "ekbar.hdf5", "dataset": "ekbar", "pool": 3},
    "ekbar_electron": {
        "file": "ekbar_electron.hdf5",
        "dataset": "ekbar_electron",
        "pool": 3,
    },
    "ekbar_photon": {"file": "ekbar_photon.hdf5", "dataset": "ekbar_photon", "pool": 3},
    "ekbar_ion": {"file": "ekbar_ion.hdf5", "dataset": "ekbar_ion", "pool": 3},
    "ekbar_positron": {
        "file": "ekbar_positron.hdf5",
        "dataset": "ekbar_positron",
        "pool": 3,
    },
    "dist_electron": {
        "file": "dist_electron.hdf5",
        "dataset": "dist_electron",
        "pool": 4,
    },  # This is the spatial_energy dist fn. Don't call it otherwise, less the code will not work.
    "dist_photon": {"file": "dist_photon.hdf5", "dataset": "dist_photon", "pool": 4},
    "dist_ion": {"file": "dist_ion.hdf5", "dataset": "dist_ion", "pool": 4},
    "dist_positron": {
        "file": "dist_positron.hdf5",
        "dataset": "dist_positron",
        "pool": 4,
    },
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
        self.pool4_meta = deque(k for k in pools[4].keys())

        self.sel = {
            1: self.pool1_meta[0] if self.pool1_meta else None,
            2: self.pool2_meta[0] if self.pool2_meta else None,
            3: self.pool3_meta[0] if self.pool3_meta else None,
            4: self.pool4_meta[0] if self.pool4_meta else None,
        }

        self.energy_bin = 0
        self.vmax_adj = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
        self.energy_keys = deque(["total_energy_particle"])
        if self.meta:
            for sp in ["electron", "ion", "photon", "positron"]:
                if f"total_energy_{sp}" in self.meta:
                    self.energy_keys.append(f"total_energy_{sp}")

        self.cmaps = setup_colormaps()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.imgs, self.cbars = {}, {}

        extent = self.get_extent(0)
        layer_setup = [
            (1, self.cmaps[2], 1.0),
            (2, self.cmaps[0], 0.6),
            (3, self.cmaps[1], 1.0),
            (4, self.cmaps[3], 0.7),
        ]

        for idx, cmap, alpha in layer_setup:
            if self.sel[idx]:
                self.imgs[idx] = self.ax.imshow(
                    np.zeros((2, 2)),
                    cmap=cmap,
                    origin="lower",
                    aspect="auto",
                    alpha=alpha,
                    extent=extent,  # type: ignore
                    zorder=idx,
                )  # type: ignore
                self.cbars[idx] = plt.colorbar(
                    self.imgs[idx], ax=self.ax, fraction=0.046, pad=0.04
                )
                if idx == 4:
                    self.imgs[idx].set_visible(False)

        self.fig.canvas.mpl_connect("key_press_event", self.onclick)
        self.refresh_plot()
        plt.show()

    def get_extent(self, time_idx):
        if self.meta and "extents" in self.meta:
            t_meta = min(time_idx, len(self.meta["extents"]) - 1)
            e = self.meta["extents"][t_meta]
            return [e[0], e[2], e[1], e[3]]
        return [0, 1, 0, 1]

    def get_energy_cutoff_text(self, b_idx, num_bins):
        if self.meta and "dist_extents" in self.meta:
            # dist_extents assumed in Joules: [E_min, E_max, ...]
            e_min = self.meta["dist_extents"][0] / 1.60218e-13
            e_max = self.meta["dist_extents"][1] / 1.60218e-13
            cutoff = e_min + (e_max - e_min) * (b_idx / num_bins)
            return f">{cutoff:.1f} MeV | < {e_max:.1f} MeV"
        return f">Bin {b_idx}"

    def get_auto_clim(self, data, pool_idx):
        flat = data.flatten()
        if flat.size > 500000:
            flat = flat[::5]
        vmin, vmax = np.percentile(flat, [1, 99.9])
        vmax *= self.vmax_adj[pool_idx]
        if vmax <= 0:
            vmax = 1e-10
        if pool_idx == 1:
            m = max(abs(vmin), abs(vmax))
            return -m, m
        return 0, vmax

    def refresh_plot(self):
        title_parts, scalar_str = [], ""
        extent = self.get_extent(self.time_step)

        if self.meta:
            t_meta = min(self.time_step, len(self.meta["times"]) - 1)
            title_parts.append(f"T={self.meta['times'][t_meta]:.1f} fs")
            if "laser_en_total" in self.meta:
                l_en, a_fr, en_field = (
                    self.meta["laser_en_total"][t_meta],
                    self.meta["abs_frac"][t_meta],
                    self.meta["total_energy_field"][t_meta],
                )
                curr_energy_key = self.energy_keys[0]
                p_en = self.meta.get(curr_energy_key, [0])[t_meta]
                e_lbl = curr_energy_key.split("_")[-1].capitalize()
                scalar_str = f"Laser: {l_en:.2e} J | Abs: {a_fr:.3f} | Field: {en_field:.2e} J | {e_lbl}: {p_en:.2e} J"

        for idx in [1, 2, 3, 4]:
            label = self.sel[idx]
            if self.imgs.get(idx) and label:
                is_visible = self.imgs[idx].get_visible()
                self.cbars[idx].ax.set_visible(is_visible)

                if is_visible:
                    full_data = self.pools[idx][label]
                    t = min(self.time_step, full_data.shape[0] - 1)

                    if idx == 4 and full_data.ndim == 4:
                        num_bins = full_data.shape[3]
                        b = min(self.energy_bin, num_bins - 1)
                        # STACKED (CUMULATIVE) DISTRIBUTION LOGIC
                        data_slice = np.sum(full_data[t, :, :, b:], axis=2).T
                        threshold_text = self.get_energy_cutoff_text(b, num_bins)
                        title_label = f"{label} ({threshold_text})"
                    else:
                        data_slice = np.transpose(full_data[t])
                        title_label = label

                    self.imgs[idx].set_data(data_slice)
                    self.imgs[idx].set_extent(extent)
                    vmin, vmax = self.get_auto_clim(data_slice, idx)
                    self.imgs[idx].set_clim(vmin, vmax)

                    unit = (
                        "a0"
                        if idx == 1
                        else (
                            "n/nc"
                            if idx == 2
                            else ("MeV" if "ekbar" in label else "arb")
                        )
                    )
                    self.cbars[idx].set_label(f"{title_label} ({unit})")
                    title_parts.append(title_label)

        self.ax.set_xlabel("x [μm]")
        self.ax.set_ylabel("y [μm]")
        self.ax.set_title(f"{scalar_str}\n" + " | ".join(title_parts), fontsize=10)
        self.fig.canvas.draw_idle()

    def onclick(self, event):
        if event.key == "right":
            self.time_step += 1
        elif event.key == "left":
            self.time_step = max(0, self.time_step - 1)
        elif event.key == "shift+right":
            self.time_step += 10
        elif event.key == "shift+left":
            self.time_step = max(0, self.time_step - 10)

        elif event.key in ["n", "m"]:
            self.pool1_meta.rotate(1 if event.key == "n" else -1)
            self.sel[1] = self.pool1_meta[0]
        elif event.key in ["d", "a"]:
            self.pool2_meta.rotate(1 if event.key == "d" else -1)
            self.sel[2] = self.pool2_meta[0]
        elif event.key in ["ctrl+n", "ctrl+m"]:
            self.pool3_meta.rotate(1 if "n" in event.key else -1)
            self.sel[3] = self.pool3_meta[0]
        elif event.key in ["v", "b"]:
            self.pool4_meta.rotate(1 if event.key == "v" else -1)
            self.sel[4] = self.pool4_meta[0]

        elif event.key == "]":
            self.energy_bin += 1
        elif event.key == "[":
            self.energy_bin = max(0, self.energy_bin - 1)
        elif event.key == "=":
            for i in [1, 2, 3, 4]:
                if self.imgs.get(i) and self.imgs[i].get_visible():
                    self.vmax_adj[i] *= 0.7
        elif event.key == "-":
            for i in [1, 2, 3, 4]:
                if self.imgs.get(i) and self.imgs[i].get_visible():
                    self.vmax_adj[i] *= 1.4

        elif event.key == "e":
            self.energy_keys.rotate(-1)
        elif event.key == "r":
            self.energy_keys.rotate(1)
        elif event.key in ["1", "2", "3", "4"]:
            idx = int(event.key)
            if idx in self.imgs:
                self.imgs[idx].set_visible(not self.imgs[idx].get_visible())

        self.refresh_plot()


class DianaInteractiveMoving(DianaInteractiveStatic):
    def __init__(self, pools, grid_params, meta=None, time_step=0):
        self.grid_params = grid_params
        super().__init__(pools, meta, time_step)

    def get_extent(self, time_idx):
        if self.meta:
            return super().get_extent(time_idx)
        p = self.grid_params
        shift = 0.0
        if p.get("window_v_x", 0) > 0 and p.get("dt") is not None:
            t = time_idx * p["dt"]
            if t > p.get("window_start", 0):
                shift = p["window_v_x"] * (t - p["window_start"])
        scale = 1e6
        return [
            (p["x_min"] + shift) * scale,
            (p["x_max"] + shift) * scale,
            p["y_min"] * scale,
            p["y_max"] * scale,
        ]


def run_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, default=".", help="Directory with HDF5 files"
    )
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    meta = None
    meta_path = os.path.join(args.dir, "metadata.hdf5")
    if os.path.exists(meta_path):
        with h5py.File(meta_path, "r") as f:
            meta = {"times": f["times"][:] * 1e15, "extents": f["extents"][:] * 1e6}  # type: ignore
            if "dist_extents" in f:
                meta["dist_extents"] = f["dist_extents"][:]  # type: ignore
            for k in [
                "laser_en_total",
                "abs_frac",
                "total_energy_particle",
                "total_energy_field",
                "total_energy_electron",
                "total_energy_photon",
                "total_energy_ion",
                "total_energy_positron",
            ]:
                if k in f:
                    meta[k] = f[k][:]  # type: ignore

    deck = parse_input_deck(args.dir)
    setup_physics(deck.get("lambda_nm", 1000.0))
    pools = {1: {}, 2: {}, 3: {}, 4: {}}
    for key, mv in VARIABLES.items():
        path = os.path.join(args.dir, mv["file"])
        if os.path.isfile(path):
            data = load_and_downsample(path, mv["dataset"], key, key, args.stride)
            if data is not None:
                pools[mv["pool"]][key] = data

    if deck.get("move_window") or meta:
        DianaInteractiveMoving(pools, deck, meta=meta)
    else:
        DianaInteractiveStatic(pools, meta=meta)


if __name__ == "__main__":
    run_cli()
