import torch
import json
import gc
import pandas             as pd
import numpy              as np
import h5py               as h5
import math
import time
import sys
from p3droslo.model       import TensorModel
from p3droslo.lines       import Line
from p3droslo.haar        import Haar

from mpi4py               import MPI
from astropy              import constants
from torch.profiler       import profile, record_function, ProfilerActivity
import magritte.core      as magritte
from astropy             import constants
from scipy.spatial.transform import Rotation
from tools               import model_find
def data_gen(model_file,radius,type_='or'):
    with h5.File(model_file,'r') as f:
        position = np.array(f['geometry/points/position'])
        velocity = np.array(f['geometry/points/velocity'])* constants.c.si.value
        temperature = np.array(f['thermodynamics/temperature/gas'])
        abundance   = np.array(f['chemistry/species/abundance'])
        vturb2      = np.array(f['thermodynamics/turbulence/vturb2'])
        fcen        = f['lines/lineProducingSpecies_0/linedata/frequency'][0]
    vpix = 300   # velocity pixel size [m/s] 
    nqua = 31
    dd   = vpix * (nqua-1)/2 / constants.c.si.value
    fmin = fcen - fcen*dd
    fmax = fcen + fcen*dd
    frequencies = torch.linspace(fmin,fmax,nqua)

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
    else:
        # Create a random rotation in radians around a random axis
        random_axis = np.random.rand(3)
        random_angles = np.random.uniform(0, 2 * np.pi)
        # random_axis = np.array([0.0,0.0,1.0])
        # random_angles = np.pi/2
        # Create a rotation object and get the rotation matrix
        r = Rotation.from_rotvec(random_angles * random_axis)
        position = r.apply(position)
        rotation_matrix = r.as_matrix()
        # velocity = np.matmul(rotation_matrix.T,velocity.T)
        # velocity = velocity.T
        # simplify by (AB).T = B.T @ A.T  and (A.T).T = A
        velocity = np.matmul(velocity,rotation_matrix)

    v_x = velocity[:,0]
    v_y = velocity[:,1]
    v_z = velocity[:,2]
    co = abundance[:,1]
    vturb = np.sqrt(vturb2)

    # input cubes
    haar = Haar(position, q=8)
    nCO_dat   = haar.map_data(co, interpolate=True)[-1][32:32+64,32:32+64,32:32+64]
    tmp_dat   = haar.map_data(temperature, interpolate=True)[-1][32:32+64,32:32+64,32:32+64]
    vturb_dat = haar.map_data(vturb, interpolate=True)[-1][32:32+64,32:32+64,32:32+64]
    v_x_dat   = haar.map_data(v_x, interpolate=True)[-1][32:32+64,32:32+64,32:32+64]
    v_y_dat   = haar.map_data(v_y, interpolate=True)[-1][32:32+64,32:32+64,32:32+64]
    v_z_dat   = haar.map_data(v_z, interpolate=True)[-1][32:32+64,32:32+64,32:32+64]
    
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
line = Line(
        species_name = "co",
        transition   = 0,
        datafile     = "/home/dc-su2/physical_informed/data_gen/co.txt",
        molar_mass   = 28.0
    )

def main(type_):
    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    nproc = comm.Get_size()

    name_lists = '/home/dc-su2/physical_informed/data_gen/datasets.json'
    with open(name_lists,'r') as file:
        lists = json.load(file)
    datasets   = lists['datasets']
    radius     = float(lists['radius'])
    model_files = model_find()

    if type_ == 'or':
        datasets = datasets[:10903]
        
    elif type_ == 'r1':
        datasets = datasets[10903:10903*2]
        
    else:
        datasets = datasets[10903*2:]
        
    n_tasks    = len(datasets)
    # tasks_per_rank = int(n_tasks / nproc)
    tasks_per_rank = math.ceil(n_tasks / nproc)
    start_index = rank*tasks_per_rank
    end_index = min(rank*tasks_per_rank + tasks_per_rank, n_tasks)
    print(f'Rank {rank}, Total Number of Tasks: {n_tasks}, Number of Ranks: {nproc}, Tasks per rank: {tasks_per_rank}')
    for idx in range(start_index,end_index):
        nCO_dat,tmp_dat,vturb_dat,v_z_dat,frequencies,img = data_gen(model_files[idx],radius,type_)
        path = datasets[idx]
        print('writing', path)
        with h5.File(path, "w") as file:
            file.create_dataset('frequencies',  data=frequencies)
            file.create_dataset("CO",           data=nCO_dat)
            file.create_dataset("temperature",  data=tmp_dat)
            file.create_dataset("v_turbulence", data=vturb_dat)
            file.create_dataset("velocity_z",   data=v_z_dat)
            file.create_dataset('I',            data=img)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python <your_script.py> <type>')
        sys.exit(1)
    type_ = sys.argv[1]
    
    main(type_)