import xarray as xr
from sdf_xarray import SDFPreprocess
import sdf_xarray as sdfxr
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
ds = xr.open_mfdataset("test_2d/*.sdf", preprocess=SDFPreprocess())
#ds = xr.open_dataset("test_2d/0010.sdf")
#print(ds)






#print(ds["Electric_Field_Ex"])



plt.ion()
#ds["Derived_Number_Density_Electron"].plot(x="X_Grid_mid", y="Y_Grid_mid")






ani = ds["Derived_Number_Density_Electron"].epoch.animate()

ani.save("Derived_Number_Density_Electron.gif", fps=5)
plt.show(block=True)