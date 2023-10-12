import h5py  as h5
import numpy as np
import os
from   tools import *
from mpi4py import MPI
import json
def radius(file):
    '''
    Find radius of shperical model.
    '''
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
    return radius
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    # find old magritte models
    model_files = model_find()
    n_task = len(model_files)
    task_per_rank = n_task//nproc
    start_idx = rank*task_per_rank
    end_idx   = min(rank*task_per_rank+task_per_rank,n_task)
    print(f'total number of tasks:{n_task},rank {rank} process files {start_idx} - {end_idx}')

    # Read and compute local minimum radius in each process
    local_min_radius = float('inf')
    for idx in range(start_idx,end_idx):
        file = model_files[idx]
        r = radius(file)
        if r <= local_min_radius:
            local_min_radius = r
    # Reduce to find global minimum radius
    global_min_radius = comm.reduce(local_min_radius, op=MPI.MIN, root=0)

    if rank == 0:
        print("Global Minimum Radius{:.6f}:".format(global_min_radius))

if __name__ == '__main__':
    main()
    