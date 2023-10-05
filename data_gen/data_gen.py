# generate input: temp, v_turb,vz, density, freq; output: I
# individual dataset
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
from astropy              import constants
from torch.profiler       import profile, record_function, ProfilerActivity
import magritte.core      as magritte

line = Line(
        species_name = "co",
        transition   = 0,
        datafile     = "/home/dc-su2/physical_informed/data_gen/co.txt",
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
    velocity    = np.array(model.geometry.points.velocity)* constants.c.si.value        # shape (427008,3)
    temperature = np.array(model.thermodynamics.temperature.gas)  # shape (427008,)
    abundance   = np.array(model.chemistry.species.abundance)
    CO          = abundance[:,1]                                  # shape (427008,)
    vturb2      = np.array(model.thermodynamics.turbulence.vturb2) # shape (419840,)
    vturb = np.sqrt(vturb2)
    v_x = velocity[:,0]
    v_y = velocity[:,1]
    v_z = velocity[:,2]
    # input cubes
    haar = Haar(position, q=7)
    # resolution (64,64,64), change q
    # proflie it for 64 case
    nCO_dat   = haar.map_data(CO, interpolate=True)[-1]
    tmp_dat   = haar.map_data(temperature, interpolate=True)[-1]
    vturb_dat = haar.map_data(vturb, interpolate=True)[-1]
    v_x_dat   = haar.map_data(v_x, interpolate=True)[-1]
    v_y_dat   = haar.map_data(v_y, interpolate=True)[-1]
    v_z_dat   = haar.map_data(v_z, interpolate=True)[-1]
    # creare model
    p3droslo_model = TensorModel(shape=nCO_dat.shape, sizes=haar.xyz_L)
    p3droslo_model['CO'         ]  = nCO_dat
    p3droslo_model['temperature']  = tmp_dat
    p3droslo_model['v_turbulence'] = vturb_dat
    p3droslo_model['velocity_x']       =        v_x_dat
    p3droslo_model['velocity_y']       =        v_y_dat
    p3droslo_model['velocity_z']       =        v_z_dat
    # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    img = line.LTE_image_along_last_axis(
    density      = p3droslo_model['CO'         ],
    temperature  = p3droslo_model['temperature'],
    v_turbulence = p3droslo_model['v_turbulence'],
    velocity_los = p3droslo_model['velocity_z'],
    frequencies  = frequencies,
    dx           = p3droslo_model.dx(3-1)
    )
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    # Avoid negative values (should probably avoid these earlier...)
    img = torch.abs(img)
    return nCO_dat,tmp_dat,vturb_dat,v_z_dat,frequencies,img

def main():
    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    nproc = comm.Get_size()

    name_lists = '/home/dc-su2/physical_informed/data_gen/rests.json'
    with open(name_lists,'r') as file:
        lists = json.load(file)
    list1      = lists['r_models']
    list2      = lists['datasets']
    n_tasks    = len(list1)
    # batch_size = 200
    tasks_per_rank = int(n_tasks / nproc)
    print(f'Rank {rank}, Number of Ranks: {nproc}, Tasks per rank: {tasks_per_rank}')
    
    # all_x = []
    # all_img = []
    # all_freqs = []
    start_index = rank*tasks_per_rank
    end_index = min(rank*tasks_per_rank + tasks_per_rank, n_tasks)

    for idx in range(start_index,end_index):
    # for i in range(rank, n_tasks, nproc):
        nCO_dat,tmp_dat,vturb_dat,v_z_dat,frequencies,img = data_gen(list1[idx])
    #     all_x.append([nCO_dat, tmp_dat, vturb_dat, v_z_dat])
    #     all_img.append(img)
    #     all_freqs.append(frequencies)

    #     if len(all_x) == batch_size:
    #         # Gather results from all processes
    #         all_x = comm.gather(all_x, root=0)
    #         all_img = comm.gather(all_img, root=0)
    #         all_freqs = comm.gather(all_freqs, root=0)

    #         if rank == 0:
    #             # Flatten the gathered results into single arrays
    #             x = np.concatenate(all_x, axis=0)
    #             img = np.concatenate(all_img, axis=0)
    #             freqs = np.concatenate(all_freqs, axis=0)
                
    #             # Save the results to an HDF5 file
    #             with h5.File(f'rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/Rotation_Dataset/Batches/batch_{i // batch_size}.hdf5', 'w') as f:
    #                 f.create_dataset('x', data=x)
    #                 f.create_dataset('img', data=img)
    #                 f.create_dataset('freqs', data=freqs)
        
    #     # Clear the arrays for the next batch
    #     all_x = []
    #     all_img = []
    #     all_freqs = []
    # close process
    # comm.barrier()

        with h5.File(list2[idx], "w") as file:
                file.create_dataset('frequencies',  data=frequencies)
                file.create_dataset("CO",           data=nCO_dat)
                file.create_dataset("temperature",  data=tmp_dat)
                file.create_dataset("v_turbulence", data=vturb_dat)
                file.create_dataset("velocity_z",   data=v_z_dat)
                file.create_dataset('I',            data=img)

if __name__ == '__main__':
    main()