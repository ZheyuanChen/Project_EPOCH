import sdf_xarray as sdfxr
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

#sdfxr.open_dataset("/home/pnd531/Desktop/Project_EPOCH/dev_test/test_ang_spectrum/sdf_files/0001.sdf")
ds= sdfxr.open_mfdataset("/home/pnd531/Desktop/Project_EPOCH/dev_test/test_ang_spectrum/sdf_files/*.sdf")


da = ds["Derived_Number_Density_Electron"]
