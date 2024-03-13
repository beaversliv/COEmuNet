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

def load_dataset_names(json_file_path):
    '''
    locate paths for individual samples
    return: list of paths
    '''
    with open(json_file_path, 'r') as file:
        lists = json.load(file)
    # face-on dataset    
    return lists['datasets']

def distribute_datasets(datasets, rank, nproc, segment_size=10903):
    '''
    split list of paths (datasets) into batches by
    marking start and end idx
    '''
    start_index = rank * segment_size
    end_index = (rank + 1) * segment_size if rank < nproc - 1 else len(datasets)
    return datasets[start_index:end_index]

def process_datasets(file_list):
    '''
    pack individual samples into baches based on the splited file paths
    '''
    grid, freq, num_input = 32, 31, 3
    X = np.zeros((len(file_list), num_input, grid, grid, grid))
    Y = np.zeros((len(file_list), grid, grid, freq))
    FREQS = np.zeros((len(file_list), freq))

    for idx, file in enumerate(file_list):
        with h5.File(file, 'r') as f:
            co = np.array(f['CO'])
            tmp = np.array(f['temperature'])
            v_z = np.array(f['velocity_z'])
            freq = np.array(f['frequencies'])
            img = np.array(f['I'])
        
        X[idx] = np.stack((v_z, tmp, co), axis=0)
        Y[idx] = img
        FREQS[idx] = freq
    
    return X, Y, FREQS
def save_processed_data(output_path, X, Y, FREQS):
    '''
    save batches into hdf5
    '''
    with h5.File(output_path, 'w') as file:
        file['input'] = X
        file['output'] = Y
        file['nfreqs'] = FREQS

# Main MPI execution flow
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    # name_lists = '/home/dc-su2/physical_informed/data_gen/datasets.json'
    name_lists = '/home/dc-su2/physical_informed/data_gen/grid32_data.json'
    datasets = load_dataset_names(name_lists)
    process_file_list = distribute_datasets(datasets, rank, nproc)

    X, Y, FREQS = process_datasets(process_file_list)
    output_file = f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/grid32/rotation/batches.hdf5'
    save_processed_data(output_file, X, Y, FREQS)
    
    
