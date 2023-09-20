from mpi4py import MPI
import numpy             as np
import h5py              as h5
import pandas            as pd
import os
import math

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
nproc = comm.Get_size()
procname = MPI.Get_processor_name()

print('start reading dataset path')
Files = pd.read_csv('data_name_files.csv',index_col=0)
inputFiles = list(Files.loc['input'])

n_tasks = len(inputFiles[10800:])

# Calculate how many tasks to allocate to each rank. Round up to ensure it still works with odd numbers of tasks.
tasks_per_rank = int(n_tasks / nproc)
print(f'Rank {rank}, Total Number of Tasks: {n_tasks}, Number of Ranks: {nproc}, Tasks per rank: {tasks_per_rank}')

# Use the rank to choose which block of inputs to process on this rank
if rank == 10:
    start_index = rank*tasks_per_rank
    end_index = max(rank*tasks_per_rank + tasks_per_rank, n_tasks)
else:
    start_index = rank*tasks_per_rank
    end_index = min(rank*tasks_per_rank + tasks_per_rank, n_tasks)

print(f'Rank {rank} does tasks {start_index} to {end_index}')
X  = np.zeros((tasks_per_rank,5,100,100,100))
ZS = np.zeros((tasks_per_rank,51,256,256))
XS = np.zeros((tasks_per_rank,256))
YS = np.zeros((tasks_per_rank,256))
NFREQ = np.zeros((tasks_per_rank,51))
j = 0
for i in range(start_index,end_index):
    model_file   = inputFiles[i]
    sample_path = f'{os.path.split(model_file)[0]}/dataset_{os.path.split(model_file)[1]}'

    # print(f'proc {rank} does number {i} :{sample_path}\n')
    sample = h5.File(sample_path,'r')
    x  = np.array(sample['input']) # shape (5,100,100,100)
    zs = np.array(sample['output'])# shape (51,256,256)
    xs = np.array(sample['xs'])
    ys = np.array(sample['ys'])
    nfreq = np.array(sample['nfreqs'])
    sample.close()
    
    X[j,:,:,:,:] = x
    ZS[j] = zs
    XS[j] = xs
    YS[j] = ys
    NFREQ[j] = nfreq
    j+=1
with h5.File(f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/Batches/batch_108.hdf5', 'w') as file:
    file['input'] = X
    file['output']= ZS
    file['xs'] = XS
    file['ys']= YS
    file['nfreqs'] = NFREQ
print(f'<<<finished generating batch_108\n')
    
