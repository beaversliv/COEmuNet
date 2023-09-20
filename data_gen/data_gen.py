# generate input: temp, v_turb,vz, density, freq; output: I
import torch
import json
import gc
import pandas             as pd
import numpy              as np
import h5py               as h5
import math
import time
from p3droslo.model       import TensorModel
from p3droslo.lines       import Line
from p3droslo.haar        import Haar
from p3droslo.forward     import image_along_last_axis as forward_image_along_last_axis
from mpi4py               import MPI
from tools                import *
from torch.profiler       import profile, record_function, ProfilerActivity
import magritte.core      as magritte

line = Line(
        species_name = "co",
        transition   = 0,
        datafile     = "co.txt",
        molar_mass   = 28.0
    )
def data_gen(model_file):
    model         = magritte.Model(model_file)
    model.compute_spectral_discretisation ()
    model.compute_inverse_line_widths     ()
    model.compute_LTE_level_populations   ()
    fcen = model.lines.lineProducingSpecies[0].linedata.frequency[0]
    vpix = 300   # velocity pixel size [m/s] 
    dd   = vpix * (model.parameters.nfreqs()-1)/2 / magritte.CC
    fmin = fcen - fcen*dd
    fmax = fcen + fcen*dd

    # start computing Intensity
    model.compute_spectral_discretisation (fmin, fmax)
    # along the first ray
    model.compute_image (0)
    image_nr = -1
    frequencies = np.array(model.images[image_nr].freqs)
    frequencies = torch.from_numpy(frequencies)

    position    = np.array(model.geometry.points.position)        # shape (427008,3)
    velocity    = np.array(model.geometry.points.velocity)        # shape (427008,3)
    temperature = np.array(model.thermodynamics.temperature.gas)  # shape (427008,)
    abundance   = np.array(model.chemistry.species.abundance)
    CO          = abundance[:,1]                                  # shape (427008,)
    vturb2      = np.array(model.thermodynamics.turbulence.vturb2) # shape (419840,)
    vturb = np.sqrt(vturb2)
    v_x = velocity[:,0]
    v_y = velocity[:,1]
    v_z = velocity[:,2]
    # input cubes
    haar = Haar(position, q=9)
    nCO_dat   = haar.map_data(CO, interpolate=True)[-1][   64:192,64:192,64:192]
    tmp_dat   = haar.map_data(temperature, interpolate=True)[-1][   64:192,64:192,64:192]
    vturb_dat = haar.map_data(vturb, interpolate=True)[-1][   64:192,64:192,64:192]
    v_x_dat   = haar.map_data(v_x, interpolate=True)[-1][   64:192,64:192,64:192]
    v_y_dat   = haar.map_data(v_y, interpolate=True)[-1][   64:192,64:192,64:192]
    v_z_dat   = haar.map_data(v_z, interpolate=True)[-1][   64:192,64:192,64:192]
    # creare model
    p3droslo_model = TensorModel(shape=nCO_dat.shape, sizes=haar.xyz_L)
    p3droslo_model['CO'         ]  = nCO_dat
    p3droslo_model['temperature']  = tmp_dat
    p3droslo_model['v_turbulence'] = vturb_dat
    p3droslo_model['velocity_x']       =        v_x_dat
    p3droslo_model['velocity_y']       =        v_y_dat
    p3droslo_model['velocity_z']       =        v_z_dat

    img = line.LTE_image_along_last_axis(
    density      = p3droslo_model['CO'         ],
    temperature  = p3droslo_model['temperature'],
    v_turbulence = p3droslo_model['v_turbulence'],
    velocity_los = p3droslo_model['velocity_z'],
    frequencies  = frequencies,
    dx           = p3droslo_model.dx(3-1)
    )
    # Avoid negative values (should probably avoid these earlier...)
    img = torch.abs(img)
    return nCO_dat,tmp_dat,vturb_dat,v_z_dat,frequencies,img

def main():
    name_lists = 'lists.json'
    with open(name_lists,'r') as file:
        lists = json.load(file)

    list1 = lists['r_models']
    # list2 = lists['datasets']
    list1 = list1[10903*3:10903*3+20]
    datasets = [f'/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/pism_forward/Demo/dataset_{i}.hdf5' for i in range(60,80)]

    for idx in range(20):
        nCO_dat,tmp_dat,vturb_dat,v_z_dat,frequencies,img = data_gen(list1[idx])
    #     # start_time = time.time()
        with h5.File(datasets[idx], "w") as file:
                file.create_dataset('frequencies',  data=frequencies)
                file.create_dataset("CO",           data=nCO_dat)
                file.create_dataset("temperature",  data=tmp_dat)
                file.create_dataset("v_turbulence", data=vturb_dat)
                file.create_dataset("velocity_z",   data=v_z_dat)
                file.create_dataset('I',            data=img)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print("Writing dataset : {:.3f} mins".format(elapsed_time/60))
if __name__ == '__main__':
    main()