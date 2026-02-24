# Project_EPOCH
My little "finding around" with EPOCH

## Functionalities
1. Making gif from simulation sdf: the script is ./src/visualisation/save_2d_gif.py and the function is called save_2d_animation_to_gif(). I also set a CLI entry point for this called save_gif_2d. To use it, call save_gif_2d --dir "input_directory_where_sdf_files_are". The function collects all sdf files in the directory, asks you for the variable to be animated, and then makes a gif for you. The gif is saved in the current directory.
2. Visualising simulation with manual control and multi-variable displaying: the script is ./src/visualisation/diana_visulisation_test.py and the function is run_cli. The entry point is called "vis_test". (Unfortunately, currently the test version is the best version). To use, type vis_test --dir "input_directory_hdf5_files". You can also decide the stride it uses (default 1) by typing --stride 2 (or 4). The more the strides, the lower the resolution but the faster the script. This is helpful when the data files are very large. WARNING: the script assumes your hdf5 files are in a directory like sth/hdf5_output (and you should put sth/hdf5_output as the input directory) AND there is an input.deck for your simulation under sth/sdf_files. The script needs to scan the input.deck for some info (and asks you for input if it doesn't find them).
3. Converting sdf files to hdf5 files: the script is ./src/post_processing/converter.py and the entry point is called sdf_converter. To use, type sdf_converter --dir "input_directory_containing_sdf_files". In default, it saves the hdf5 files in sth/hdf5_output (your sdf files are in sth/sdf_files).

### Notes
Some features are quite hard-wired, especially the structure of your directory. This isn't great. I will have to fix this in the future.
This "software" does not track .sdf, .hdf5, and .gif files. So, when cloning the repo, one needs to run the epoch simulation and convert them to hdf5 files.

