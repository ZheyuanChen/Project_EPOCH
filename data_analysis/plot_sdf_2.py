import xarray as xr
import sdf
import sdf_helper as sh
import numpy as np
import matplotlib.pyplot as plt


data = sh.getdata(79,'Data')
#>>Reading file Data/0007.sdf

# List all available variables
#sh.list_variables(data)

Electric_Field_Ex = data.Electric_Field_Ex
raw = Electric_Field_Ex.data
#print(type(raw))
#print(np.mean(raw))

print(data.__dict__)

#
#plt.ion()
sh.plot_auto(data.Electric_Field_Ey)
plt.show(block=True)
