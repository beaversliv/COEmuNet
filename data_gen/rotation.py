# script to rotate one magritte model and save dataset (cubes,intensity)
# fix vpix as 300 [m/s]
from scipy.spatial.transform import Rotation
import numpy as np
import h5py as h5
import magritte.core     as magritte
import magritte.setup    as setup      # Model setup
from scipy.interpolate import griddata # map to 2d cartesian data point.
import matplotlib.pyplot as plt
from astropy              import constants, units   # Unit conversions

import magritte.plot     as plot       # Plotting
import os
import pandas            as pd
from tools       import *
import math
from mpi4py      import MPI

lambda_file         = '/home/dc-su2/physical_informed/data_gen/co.txt'
nquads              = 31
vpix = 300   # velocity pixel size [m/s], velocity step, difference between 2 channel maps, samller->close to central line
                  # play with it, to see the channel maps

def rotation_gen(model_file,rotated_model_file,r):
    # re-write some parameters for compating with new Magritte environment
    try:
        magritte.Model(model_file)
    except RuntimeError:
    # re-write some parameters for compating with new Magritte environment
        with h5.File(model_file, 'a') as file:
            # Setting parameters.
            file.attrs['use_scattering'] = 'false'
            file.attrs['hnrays'] = 6
            file.attrs['nlines'] = 1
            file.attrs['nfreqs'] = 51
        print(f'finish re-wirte: {model_file}')
    # read info from old magritte model
    model = magritte.Model(model_file)
    position  = np.array(model.geometry.points.position)
    velocity  = np.array(model.geometry.points.velocity)
    nbs       = np.array(model.geometry.points.neighbors)
    n_nbs     = np.array(model.geometry.points.n_neighbors)
    abundance = np.array(model.chemistry.species.abundance)
    temp      = np.array(model.thermodynamics.temperature.gas)
    trb       = np.array(model.thermodynamics.turbulence.vturb2)
    boundary  = np.array(model.geometry.boundary.boundary2point)
    ncells = model.parameters.npoints()
    n_boundary = model.parameters.nboundary()
    
    position_rotated = r.apply(position)

    r_model = magritte.Model ()                              # Create model object
    r_model.parameters.set_model_name         (rotated_model_file)   # Magritte model file
    r_model.parameters.set_dimension          (3)            # This is a 3D model
    r_model.parameters.set_npoints            (ncells)       # Number of points
    r_model.parameters.set_nrays              (12)           # Number of rays  
    r_model.parameters.set_nspecs             (5)            # Number of species (min. 5)
    r_model.parameters.set_nlspecs            (1)            # Number of line species
    r_model.parameters.set_nquads             (nquads)       # Number of quadrature points

    r_model.geometry.points.position.set(position_rotated)
    # r_model.geometry.points.position.set(position)
    r_model.geometry.points.velocity.set(velocity)

    r_model.geometry.points.  neighbors.set(  nbs)
    r_model.geometry.points.n_neighbors.set(n_nbs)

    r_model.chemistry.species.abundance = abundance
    r_model.chemistry.species.symbol    = ['dummy0', 'CO', 'H2', 'e-', 'dummy1']

    r_model.thermodynamics.temperature.gas  .set(temp)
    r_model.thermodynamics.turbulence.vturb2.set(trb)

    r_model.parameters.set_nboundary(boundary.shape[0])
    r_model.geometry.boundary.boundary2point.set(boundary)

    r_model = setup.set_boundary_condition_zero  (r_model)
    # observe from the direction of the first ray [0,0,1]
    r_model = setup.set_uniform_rays            (r_model,first_ray=np.array([0.0,0.0,1.0]))
    r_model = setup.set_linedata_from_LAMDA_file(r_model, lambda_file, {'considered transitions': [0]})
    # model = setup.set_linedata_from_LAMDA_file(model, lamda_file)   # Consider all transitions
    r_model = setup.set_quadrature              (r_model)
    r_model.write()
    return rotated_model_file

def main():
    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    nproc = comm.Get_size()
    procname = MPI.Get_processor_name()

    model_files   = model_find()
    r_model_files,dataset_files = path_rotations(model_files)

    n_tasks = len(model_files)
    # Calculate how many tasks to allocate to each rank. Round up to ensure it still works with odd numbers of tasks.
    tasks_per_rank = math.ceil(n_tasks / nproc)
    print(f'Rank {rank}, Total Number of Tasks: {n_tasks}, Number of Ranks: {nproc}, Tasks per rank: {tasks_per_rank}')
    start_index = rank*tasks_per_rank
    end_index = min(rank*tasks_per_rank + tasks_per_rank, n_tasks)

    for idx in range(start_index,end_index):
        model_file = model_files[idx]
        r_model_file = r_model_files[idx]

        # Create a random rotation in radians around a random axis
        # random_axis = np.random.rand(3)
        # random_angles = np.random.uniform(0, 2 * np.pi)
        # # Create a rotation object and get the rotation matrix
        # r = Rotation.from_rotvec(random_angles * random_axis)

        r_model_file = rotation_gen(model_file,r_model_file)

if __name__ == '__main__':
    main()