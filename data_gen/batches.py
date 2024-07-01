from mpi4py import MPI
import numpy             as np
import h5py              as h5
import pandas            as pd
import os
import math
import json
import random
import sys

def load_dataset_names(json_file_path):
    '''
    locate paths for individual samples
    return: list of paths
    '''
    with open(json_file_path, 'r') as file:
        lists = json.load(file)
    datasets = lists['datasets']   
    # random.shuffle(datasets)
    return datasets

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
    grid, freq, num_input = 64, 1, 3
    X = np.zeros((len(file_list), num_input, grid, grid, grid))
    Y = np.zeros((len(file_list), grid, grid, freq))
    for idx, file in enumerate(file_list):
        with h5.File(file, 'r') as f:
            co = np.array(f['CO'])
            tmp = np.array(f['temperature'])
            v_z = np.array(f['velocity_z'])
            img = np.array(f['I'])
        
        X[idx] = np.stack((v_z, tmp, co), axis=0)
        Y[idx] = img
    
    return X, Y
def save_processed_data(output_path, X, Y):
    '''
    save batches into hdf5
    '''
    with h5.File(output_path, 'w') as file:
        file['input'] = X
        file['output'] = Y
# Main MPI execution flow
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    # name_lists = '/home/dc-su2/physical_informed/data_gen/datasets.json'
    name_lists = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/files/grid64_data.json'
    datasets = load_dataset_names(name_lists)
    # process_file_list = distribute_datasets(datasets[10903:10903*2], rank, nproc,len(datasets)//nproc)
    process_file_list = distribute_datasets(datasets[10903:], rank, nproc,10903*2)
    X, Y = process_datasets(process_file_list)
    print(Y.shape)
    output_file = f'/home/dc-su2/rds/rds-dirac-dr004/Magritte/random_grid64_data.hdf5'
    save_processed_data(output_file, X, Y)
    
