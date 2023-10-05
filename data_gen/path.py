import h5py  as h5
import numpy as np
import os
from   tools import *
import json
def radius():
    '''
    find global minimum radius.
    '''
    ### old magritte models ###
    model_files = model_find()
    global_radius = float('inf')
    for file in model_files:
        with h5.File(file,'r') as f:
            position = np.array(f['geometry/points/position'])

        x_min = position[:,0].min()
        x_max = position[:,0].max()
        y_min = position[:,1].min()
        y_max = position[:,1].max()
        z_min = position[:,2].min()
        z_max = position[:,2].max()
        # define the smallest difference as local radius
        radius = np.min([(x_max   - x_min)/2, (y_max - y_min)/2,(z_max - z_min)/2])
        if radius <= global_radius:
            global_radius = radius
    return global_radius

Raidus = radius()
print(Raidus)
# find old magritte models
model_files = model_find()
# get absolute path of datasets
origine,origin_data = path_rotations(model_files,dir='physical_forward',type_='Original')
_,r1_data = path_rotations(model_files,dir='physical_forward',type_='R1')
_, = path_rotations(model_files,dir='physical_forward',type_='R2')

data = {'radius':Radius,'datasets':origin_data+r1_data+r2_data}
with open('datasets.json','w') as file:
    json.dump(data,file)