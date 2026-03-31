# This script demonstrates how to create an animation of the "Derived_Number_Density_Electron" variable from the SDF files using Matplotlib's FuncAnimation. The animation will show how the electron number density evolves over time (epoch).

import sdf_xarray as sdfxr
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from IPython.display import HTML

ds= sdfxr.open_mfdataset("/home/pnd531/Desktop/Project_EPOCH/dev_test/test_ang_spectrum/sdf_files/*.sdf")
da = ds["Electric_Field_Ex"]
fig, ax = plt.subplots()
anim = da.epoch.animate(ax=ax)
anim.save("./animation.gif")
plt.show()