import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider


def main():
    parser = argparse.ArgumentParser(
        description="Interactive viewer for EPOCH spectra subdirectories."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Parent directory containing the spectra subfolders",
    )
    args = parser.parse_args()

    parent_dir = os.path.abspath(args.dir)

    # Scan parent directory and group PNGs by their containing subdirectory
    groups = {}
    for root, dirs, files in os.walk(parent_dir):
        png_files = [f for f in files if f.endswith(".png")]
        if png_files:
            folder_name = os.path.basename(root)
            full_paths = sorted([os.path.join(root, f) for f in png_files])
            groups[folder_name] = full_paths

    if not groups:
        print(f"No PNG files found in any subdirectories of {parent_dir}")
        return

    group_names = sorted(list(groups.keys()))

    current_group_idx = 0
    current_frame_idx = 0

    # Setup the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.15)

    # Load the first image directly from disk
    first_group = group_names[current_group_idx]
    img_display = ax.imshow(mpimg.imread(groups[first_group][current_frame_idx]))
    ax.axis("off")

    # Setup the Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # type: ignore
    max_frames = len(groups[first_group]) - 1
    slider = Slider(
        ax_slider, "Time Frame", 0, max_frames, valinit=0, valstep=1, valfmt="%d"
    )

    def update_view(group_switched=False):
        nonlocal img_display
        group = group_names[current_group_idx]

        # Safely cap the index in case subdirectories have different file counts
        safe_idx = min(current_frame_idx, len(groups[group]) - 1)
        img_path = groups[group][safe_idx]

        # Read from disk on the fly (your operating system will naturally cache recent files)
        new_img = mpimg.imread(img_path)

        if group_switched:
            # Clear the axis to completely reset the aspect ratio and bounds
            ax.clear()
            img_display = ax.imshow(new_img)
            ax.axis("off")
        else:
            # Fast update for sliding through time within the same group
            img_display.set_data(new_img)

        ax.set_title(f"Folder: {group} | Frame: {safe_idx}", fontsize=14, pad=10)
        fig.canvas.draw_idle()

    def on_slider_update(val):
        nonlocal current_frame_idx
        current_frame_idx = int(slider.val)
        update_view(group_switched=False)

    slider.on_changed(on_slider_update)

    def on_key(event):
        nonlocal current_group_idx, current_frame_idx

        # Up/Down arrows to cycle through folders (plot types)
        if event.key in ["up", "down"]:
            if event.key == "up":
                current_group_idx = (current_group_idx + 1) % len(group_names)
            else:
                current_group_idx = (current_group_idx - 1) % len(group_names)

            # Adjust the slider max value for the new folder's length
            group = group_names[current_group_idx]
            new_max = len(groups[group]) - 1

            slider.valmax = new_max
            slider.ax.set_xlim(slider.valmin, slider.valmax)

            # Keep current frame index if possible, otherwise cap it
            current_frame_idx = min(current_frame_idx, new_max)
            slider.set_val(current_frame_idx)
            update_view(group_switched=True)

        # Left/Right arrows to step through time frames
        elif event.key == "right":
            slider.set_val(min(slider.val + 1, slider.valmax))
        elif event.key == "left":
            slider.set_val(max(slider.val - 1, slider.valmin))

    # Connect the keyboard events
    fig.canvas.mpl_connect("key_press_event", on_key)

    ax.set_title(
        f"Folder: {group_names[current_group_idx]} | Frame: {current_frame_idx}",
        fontsize=14,
        pad=10,
    )

    print("\n=== Viewer Controls ===")
    print("[Left/Right Arrow] : Step forward/backward in time")
    print("[Up/Down Arrow]    : Switch between folders (spectra types)")
    print("[Mouse Slider]     : Jump quickly across time")
    print("=======================\n")

    plt.show()


if __name__ == "__main__":
    main()
