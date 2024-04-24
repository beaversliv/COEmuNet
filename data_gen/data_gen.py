import torch
import json
import gc
import pandas             as pd
import numpy              as np
import h5py               as h5
import math
import time
import logging
import sys
from p3droslo.model       import TensorModel
from p3droslo.lines       import Line
from p3droslo.haar        import Haar
from p3droslo.utils       import planck  # CMB, big bang background

from mpi4py               import MPI
from astropy              import constants
from torch.profiler       import profile, record_function, ProfilerActivity
import magritte.core      as magritte
from astropy             import constants
from scipy.spatial.transform import Rotation
from tools               import model_find
def data_gen(model_file,line,radius,type_='or',model_grid=64):
    with h5.File(model_file,'r') as f:
        position = np.array(f['geometry/points/position'])
        velocity = np.array(f['geometry/points/velocity'])* constants.c.si.value
        temperature = np.array(f['thermodynamics/temperature/gas'])
        abundance   = np.array(f['chemistry/species/abundance'])
        vturb2      = np.array(f['thermodynamics/turbulence/vturb2'])
    
    fcen = line.frequency
    vpix = 300   # velocity pixel size [m/s] 
    nqua = 31
    dd   = vpix * (nqua-1)/2 / constants.c.si.value
    fmin = fcen - fcen*dd
    fmax = fcen + fcen*dd
    frequencies = torch.linspace(fmin,fmax,nqua,dtype=torch.float64)
    # # redefine frequency range, only focused on intersted centred region
    start_freq,end_freq = frequencies[11],frequencies[19]
    frequencies = torch.linspace(start_freq,end_freq,31)

    x_min = position[:,0].min()
    x_max = position[:,0].max()
    y_min = position[:,1].min()
    y_max = position[:,1].max()
    z_min = position[:,2].min()
    z_max = position[:,2].max()
    center = np.array([(x_max + x_min)/2, (y_max + y_min)/2,(z_max + z_min)/2])
 
    distances = np.linalg.norm(position - center, axis=1)
    indices_within_ball = np.where(distances <= radius)[0]
    position       = position[indices_within_ball]
    velocity       = velocity[indices_within_ball]
    temperature    = temperature[indices_within_ball]
    abundance      = abundance[indices_within_ball]
    vturb2         = vturb2[indices_within_ball]

    if type_ == 'or':
        # unrotated
        position = position
        velocity = velocity
    else:
        # Create a random rotation in radians around a random axis
        random_axis = np.random.rand(3)
        random_axis /= np.linalg.norm(random_axis)
        random_angles = np.random.uniform(0, 2 * np.pi)
        # random_axis = np.array([0.0,0.0,1.0])
        # random_angles = np.pi/2
        # Create a rotation object and get the rotation matrix
        r = Rotation.from_rotvec(random_angles * random_axis)
        position = r.apply(position)

        rotation_matrix = r.as_matrix()
        velocity = np.matmul(rotation_matrix,velocity.T)
        velocity = velocity.T

    v_x = velocity[:,0]
    v_y = velocity[:,1]
    v_z = velocity[:,2]
    co = abundance[:,1]
    vturb = np.sqrt(vturb2)
    if model_grid == 32:
        # get grid 32,don't touch!!!
        haar = Haar(position, q=7)
        nCO_dat   = haar.map_data(co, interpolate=True)[-1][16:16+32,16:16+32,16:16+32]
        tmp_dat   = haar.map_data(temperature, interpolate=True)[-1][16:16+32,16:16+32,16:16+32]
        vturb_dat = haar.map_data(vturb, interpolate=True)[-1][16:16+32,16:16+32,16:16+32]
        v_x_dat   = haar.map_data(v_x, interpolate=True)[-1][16:16+32,16:16+32,16:16+32]
        v_y_dat   = haar.map_data(v_y, interpolate=True)[-1][16:16+32,16:16+32,16:16+32]
        v_z_dat   = haar.map_data(v_z, interpolate=True)[-1][16:16+32,16:16+32,16:16+32]   
    elif model_grid == 64:
         # input cubes for grid 64, don't touch!!!
        haar = Haar(position, q=8)#q=8, gens (128,128,128)
        nCO_dat   = haar.map_data(co, interpolate=True)[-1][32:32+64,32:32+64,32:32+64]
        tmp_dat   = haar.map_data(temperature, interpolate=True)[-1][32:32+64,32:32+64,32:32+64]
        vturb_dat = haar.map_data(vturb, interpolate=True)[-1][32:32+64,32:32+64,32:32+64]
        v_x_dat   = haar.map_data(v_x, interpolate=True)[-1][32:32+64,32:32+64,32:32+64]
        v_y_dat   = haar.map_data(v_y, interpolate=True)[-1][32:32+64,32:32+64,32:32+64]
        v_z_dat   = haar.map_data(v_z, interpolate=True)[-1][32:32+64,32:32+64,32:32+64]
    elif model_grid == 128:
        # get grid 128,don't touch!!!
        haar = Haar(position, q=9)
        nCO_dat   = haar.map_data(co, interpolate=True)[-1][64:64+128,64:64+128,64:64+128]
        tmp_dat   = haar.map_data(temperature, interpolate=True)[-1][64:64+128,64:64+128,64:64+128]
        vturb_dat = haar.map_data(vturb, interpolate=True)[-1][64:64+128,64:64+128,64:64+128]
        v_x_dat   = haar.map_data(v_x, interpolate=True)[-1][64:64+128,64:64+128,64:64+128]
        v_y_dat   = haar.map_data(v_y, interpolate=True)[-1][64:64+128,64:64+128,64:64+128]
        v_z_dat   = haar.map_data(v_z, interpolate=True)[-1][64:64+128,64:64+128,64:64+128]

    # creare model
    p3droslo_model = TensorModel(shape=nCO_dat.shape, sizes=haar.xyz_L)
    p3droslo_model['CO'         ]  = nCO_dat
    p3droslo_model['temperature']  = tmp_dat
    p3droslo_model['v_turbulence'] = vturb_dat
    p3droslo_model['velocity_x']       =        v_x_dat
    p3droslo_model['velocity_y']       =        v_y_dat
    p3droslo_model['velocity_z']       =        v_z_dat
    # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    # intensity along z-axis
    # time measurement
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

    # the physics CMB, physical threshold
    T_CMB = 2.725
    CMB = planck(T_CMB,frequencies) 
    A = torch.ones_like(img)
    CMB_matrix = CMB * A
    img[img<CMB_matrix] = CMB_matrix[img < CMB_matrix]

    return nCO_dat,tmp_dat,v_z_dat,frequencies,img


def main(type_):
    # setup logging
    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    nproc = comm.Get_size()
     
    name_lists = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/files/mul_freq_gird64.json'
    with open(name_lists,'r') as file:
        lists = json.load(file)
    datasets   = lists['datasets']
    radius     = float(lists['radius'])

    line = Line(
        species_name = "co",
        transition   = 1,
        datafile     = "/home/dc-su2/physical_informed/data_gen/co.txt",
        molar_mass   = 28.0
    )

    model_files = model_find()

    if type_ == 'or':
        datasets = datasets[:10903]
        
    elif type_ == 'r1':
        datasets = datasets[:10903]
        # logging.basicConfig(filename=f'/home/dc-su2/physical_informed/data_gen/files/faceon_runtime128_{rank}.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    elif type_ == 'r2':
        datasets = datasets[10903:]
        
    n_tasks    = len(datasets)
    # tasks_per_rank = int(n_tasks / nproc)
    tasks_per_rank = math.ceil(n_tasks / nproc)
    start_index = rank*tasks_per_rank
    end_index = min(rank*tasks_per_rank + tasks_per_rank, n_tasks)
    # print(f'Rank {rank}, Total Number of Tasks: {n_tasks}, Number of Ranks: {nproc}, Tasks per rank: {tasks_per_rank}')
    for idx in range(start_index,end_index):
        start = time.time()
        print(f'reading {model_files[idx]}')
        nCO_dat,tmp_dat,v_z_dat,frequencies,img = data_gen(model_files[idx],line,radius,type_,64)
        end = time.time()
        running_time = end - start
        
        # logging.info(f'Running time: {running_time:.5} seconds')
        path = datasets[idx]
        print(f'writing {path}\n')
        with h5.File(path, "w") as file:
            file.create_dataset('frequencies',  data=frequencies)
            file.create_dataset("CO",           data=nCO_dat)
            file.create_dataset("temperature",  data=tmp_dat)
            file.create_dataset("velocity_z",   data=v_z_dat)
            file.create_dataset('I',            data=img)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python <your_script.py> <type>')
        sys.exit(1)
    type_ = sys.argv[1]
    
    main(type_)