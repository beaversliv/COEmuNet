from mpi4py import MPI
import numpy             as np
import h5py              as h5
import pandas            as pd
import os
import math

# comm  = MPI.COMM_WORLD
# rank  = comm.Get_rank()
# nproc = comm.Get_size()
# procname = MPI.Get_processor_name()

# n_tasks = len(inputFiles[10800:])

# # Calculate how many tasks to allocate to each rank. Round up to ensure it still works with odd numbers of tasks.
# tasks_per_rank = int(n_tasks / nproc)
# print(f'Rank {rank}, Total Number of Tasks: {n_tasks}, Number of Ranks: {nproc}, Tasks per rank: {tasks_per_rank}')

# # Use the rank to choose which block of inputs to process on this rank
# if rank == 10:
#     start_index = rank*tasks_per_rank
#     end_index = max(rank*tasks_per_rank + tasks_per_rank, n_tasks)
# else:
#     start_index = rank*tasks_per_rank
#     end_index = min(rank*tasks_per_rank + tasks_per_rank, n_tasks)

# print(f'Rank {rank} does tasks {start_index} to {end_index}')
datasets = [f'/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/pism_forward/Demo/dataset_{i}.hdf5' for i in range(80)]
tasks_per_rank = len(datasets)
X  = np.zeros((tasks_per_rank,4,128,128,128))
Y = np.zeros((tasks_per_rank,128,128,31))
FREQS = np.zeros((tasks_per_rank,31))
# j = 0
start_index = 0
end_index = len(datasets)
for i in range(start_index,end_index):
    sample_path = datasets[i]
    # print(f'proc {rank} does number {i} :{sample_path}\n')
    with h5.File(sample_path,'r') as f:
        co     = np.array(f['CO'])
        tmp    = np.array(f['temperature'])
        v_turv = np.array(f['v_turbulence'])
        v_z    = np.array(f['velocity_z'])
        freq   = np.array(f['frequencies'])
        Img    = np.array(f['I'])
    tur = (v_z,v_turv,tmp,co)
    for m in range(4):
        X[i,m,:,:,:] = tur[m]
    Y[i] = Img
    FREQS[i] = freq
    # j+=1
with h5.File('/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/pism_forward/Demo/data.hdf5', 'w') as file:
    file['input'] = X
    file['output']= Y
    file['nfreqs'] = FREQS
print(f'<<<finished generating dataset\n')
    
