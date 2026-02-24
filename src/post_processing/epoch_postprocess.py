# This is the legacy script provided by Diana for sdf-to-hdf5 conversion. I am not sure if I changed it or not, but I will keep it here for reference. The new script is epoch_postprocess.py, which is more streamlined and only extracts the necessary variables for our analysis.

import h5py
import numpy as np
import os
import sdf_helper as sh
import scipy
print(scipy.__version__)

root=[os.path.abspath(os.path.dirname(__file__))+'/']

root=root[0]
print ('**************\n''ROOT FOLDER \n'+root+'\n**************')
      
f=h5py.File(root+'fields.hdf5',"w", driver="core")
for j in range(1000):
    if os.path.isfile(root+str(j).zfill(4)+'.sdf'):
        print(root+str(j).zfill(4)+'.sdf')
        data=sh.getdata(j)

        #sh.list_variables(data)

        Bz_data = data.Magnetic_Field_Bz
        #Ex_data = data.Electric_Field_Ex
        #Ey_data = data.Electric_Field_Ey
        #Ez_data = data.Electric_Field_Ez

        Bz=Bz_data.data
        #Ex=Ex_data.data
        #Ey=Ey_data.data
        #Ez=Ez_data.data

        Bz = np.expand_dims(Bz, axis=0)
        #Ex = np.expand_dims(Ex, axis=0)
        #Ey = np.expand_dims(Ey, axis=0)
        #Ez = np.expand_dims(Ez, axis=0)

        if j==0:
            f.create_dataset('Bz', data=Bz, chunks=True, maxshape=(None,None,None))
            #f.create_dataset('Ex', data=Ex, chunks=True, maxshape=(None,None,None))
            #f.create_dataset('Ey', data=Ey, chunks=True, maxshape=(None,None,None))
            #f.create_dataset('Ez', data=Ez, chunks=True, maxshape=(None,None,None))

        else:# Append new data 
            f['Bz'].resize((f['Bz'].shape[0] + 1), axis=0)
            f['Bz'][-1] = Bz
            #f['Ex'].resize((f['Ex'].shape[0] + 1), axis=0)
            #f['Ex'][-1] = Ex
            #['Ey'].resize((f['Ey'].shape[0] + 1), axis=0)
            #f['Ey'][-1] = Ey
            #f['Ez'].resize((f['Ez'].shape[0] + 1), axis=0)
            #f['Ez'][-1] = Ez

        print(j)        
print(' File created!')  
f.close()

f=h5py.File(root+'particles.hdf5',"w", driver="core")
for j in range(1000):
    if os.path.isfile(root+str(j).zfill(4)+'.sdf'):
        print(root+str(j).zfill(4)+'.sdf')
        data=sh.getdata(j)

        electron_density_data = data.Derived_Number_Density_electron
        electron_density=electron_density_data.data
        electron_density = np.expand_dims(electron_density, axis=0)

        if j==0:
            f.create_dataset('electron_density', data=electron_density, chunks=True, maxshape=(None,None,None))
        else:# Append new data 
            f['electron_density'].resize((f['electron_density'].shape[0] + 1), axis=0)
            f['electron_density'][-1] = electron_density
        print(j)
        
print(' File created!') 
f.close()