import torch
import json
import gc
import pandas             as pd
import numpy              as np
import h5py               as h5


from p3droslo.model       import TensorModel
from p3droslo.lines       import Line
from p3droslo.haar        import Haar
from p3droslo.utils       import planck  # CMB, big bang background

from mpi4py               import MPI
from torch.profiler       import profile, record_function, ProfilerActivity

from astropy              import constants
from scipy.spatial.transform import Rotation
from tools                import model_find
import argparse
import time
import logging
import sys
import math 
def generate_random_rotation_matrix():
    """
    Generate a random rotation matrix for 3D object rotation

    Returns:
    numpy.ndarray: A 3x3 rotation matrix.
    """
    # Generate a random unit vector uniformly distributed on the unit sphere
    u = np.random.uniform(0, 1, 3)
    theta = 2 * np.pi * u[0]
    phi = np.arccos(1 - 2 * u[1])
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    axis = np.array([x, y, z])
    
    # Generate a random angle for rotation about this vector
    angle = 2 * np.pi * np.random.uniform(0, 1)
    
    # Construct the rotation matrix using the axis-angle representation
    r = Rotation.from_rotvec(angle * axis)
    # Using Rodrigues' rotation formula
    # K = np.array([[0, -axis[2], axis[1]],
    #               [axis[2], 0, -axis[0]],
    #               [-axis[1], axis[0], 0]])
    
    # R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    R = r.as_matrix() 
    return R

def data_gen(model_file,line,radius,type_='faceon',mulfreq=True,model_grid=64):
    with h5.File(model_file,'r') as f:
        position = np.array(f['geometry/points/position'])
        velocity = np.array(f['geometry/points/velocity'])* constants.c.si.value
        temperature = np.array(f['thermodynamics/temperature/gas'])
        abundance   = np.array(f['chemistry/species/abundance'])
        vturb2      = np.array(f['thermodynamics/turbulence/vturb2'])
    
    fcen = line.frequency
    if mulfreq:
        print('multiple frequency')
        vpix = 300   # velocity pixel size [m/s] 
        nqua = 31
        dd   = vpix * (nqua-1)/2 / constants.c.si.value
        fmin = fcen - fcen*dd
        fmax = fcen + fcen*dd
        frequencies = torch.linspace(fmin,fmax,nqua,dtype=torch.float64)
    else:
        print('single frequency')
        frequencies = torch.tensor([fcen],dtype=torch.float64)

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

    if type_ == 'faceon':
        # unrotated
        position = position
        velocity = velocity
    else:
        rotation_matrix = generate_random_rotation_matrix()
        position = np.matmul(rotation_matrix,position.T)
        position = position.T
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

    # create model
    p3droslo_model = TensorModel(shape=nCO_dat.shape, sizes=haar.xyz_L)
    p3droslo_model['CO'         ]  = nCO_dat
    p3droslo_model['temperature']  = tmp_dat
    p3droslo_model['v_turbulence'] = vturb_dat
    p3droslo_model['velocity_x']       =        v_x_dat
    p3droslo_model['velocity_y']       =        v_y_dat
    p3droslo_model['velocity_z']       =        v_z_dat
    # intensity along z-axis
    # time measurement
    start = time.time()
    img = line.LTE_image_along_last_axis(
    density      = p3droslo_model['CO'         ],
    temperature  = p3droslo_model['temperature'],
    v_turbulence = p3droslo_model['v_turbulence'],
    velocity_los = p3droslo_model['velocity_z'],
    frequencies  = frequencies,
    dx           = p3droslo_model.dx(3-1)
    )
    end = time.time()
    running_time = end - start
    # logging.info(f'Running time: {running_time:.6} seconds')
    # Avoid negative values (should probably avoid these earlier...)
    img = torch.abs(img)
    
    # the physics CMB, physical threshold
    # T_CMB = 2.725
    # CMB = planck(T_CMB,frequencies) 
    # A = torch.ones_like(img)
    # CMB_matrix = CMB * A
    # img[img<CMB_matrix] = CMB_matrix[img < CMB_matrix]

    return nCO_dat,tmp_dat,v_z_dat,frequencies,img
def data_path_files(model_file,rotation_idx,gen_path):

    listb = model_file.split('/')
    # if mul case, just 'physical_forward/mul_freq/grid64'
    listb.insert(6,gen_path)
    listb[-1] = f'{listb[-2]}_{rotation_idx}.hdf5'
    rotation_file = ('/').join(listb)
    return rotation_file
def parse_args():
    parser = argparse.ArgumentParser(description="A script with command-line arguments")
    parser.add_argument('--type', type = str, default = "Specify the type (faceon,rotation)")
    parser.add_argument('--model_grid', type = int, default = 64)
    parser.add_argument('--num_rotations', type = int, default = 50)
    parser.add_argument('--mulfreq', action='store_true', default = False,help="Flag to indicate if multiple frequencies should be used (default: False)")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    nproc = comm.Get_size()
    if args.mulfreq:
        transition = 1
    else:
        transition = 0
    line = Line(
        species_name = "co",
        transition   = transition,
        datafile     = "/home/dc-su2/physical_informed/data/data_gen/co.txt",
        molar_mass   = 28.0
    )

    model_files = model_find()
    radius = 28984584112701.441406 # pre-calculated in radius.py
    # distirbute tasks to each processor
    n_tasks    = len(model_files) * args.num_rotations
    tasks_per_rank = math.ceil(n_tasks / nproc)
    start_index = rank*tasks_per_rank
    end_index = min(rank*tasks_per_rank + tasks_per_rank, n_tasks)
    
    for idx in range(start_index,end_index):
        file_idx = idx // args.num_rotations
        rotation_idx = idx % args.num_rotations
        model_file = model_files[file_idx]
        print(f'Rank {rank} processing file {model_file} with rotation {rotation_idx}')
        
        if args.mulfreq:
            gen_file = data_path_files(model_file,rotation_idx,gen_path='physical_forward/mul_freq/grid64')
        else:
            gen_file = data_path_files(model_file,rotation_idx,gen_path='physical_forward/sgl_freq/grid64')
        nCO_dat,tmp_dat,v_z_dat,frequencies,img = data_gen(model_file,line=line,radius=radius,type_=args.type,mulfreq=args.mulfreq,model_grid=args.model_grid)
        print(f'Rank {rank} writing {gen_file}')
        with h5.File(gen_file, "w") as file:
            file.create_dataset('frequencies',  data=frequencies)
            file.create_dataset("CO",           data=nCO_dat)
            file.create_dataset("temperature",  data=tmp_dat)
            file.create_dataset("velocity_z",   data=v_z_dat)
            file.create_dataset('I',            data=img)
    comm.Barrier()
if __name__ == '__main__':
    main()
