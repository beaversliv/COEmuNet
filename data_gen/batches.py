from mpi4py import MPI
import numpy             as np
import h5py              as h5
import pandas            as pd
import os
import math
import json
import random
import sys

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
nproc = comm.Get_size()
procname = MPI.Get_processor_name()

name_lists = '/home/dc-su2/physical_informed/data_gen/datasets.json'
# name_lists = '/home/dc-su2/physical_informed/data_gen/org_data.json'
with open(name_lists,'r') as file:
    lists = json.load(file)
samples = lists['datasets']
# randomly select 12000 samples from dataset
datasets = random.sample(samples,12000)

segment_size = 2400

start_index = rank * segment_size
end_index = (rank + 1) * segment_size if rank < nproc - 1 else len(datasets)

process_file_list = datasets[start_index:end_index]

X  = np.zeros((segment_size,3,64,64,64))
Y = np.zeros((segment_size,64,64,31))
FREQS = np.zeros((segment_size,31))
for idx,file in enumerate(process_file_list):
    with h5.File(file,'r') as f:
        co     = np.array(f['CO'])
        tmp    = np.array(f['temperature'])
        v_z    = np.array(f['velocity_z'])
        freq   = np.array(f['frequencies'])
        img    = np.array(f['I'])
    x_i = np.stack((v_z,tmp,co),axis=0)
    # if x_i.shape != (4,64,64,64):
    #     print(file)
    X[idx] = x_i
    Y[idx] = img
    FREQS[idx] = freq
with h5.File(f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/Batches/rotate12000_{rank}.hdf5', 'w') as file:
    file['input'] = X
    file['output']= Y
    file['nfreqs'] = FREQS
    
    
