import h5py as h5
import numpy as np
import random
import os
import time
from mpi4py import MPI
import sys
import json
import logging
import argparse
from collections import OrderedDict
from utils import check_file

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'Function {func.__name__} took {end_time - start_time:.4f} seconds')
        return result
    return wrapper

def batch_files(file_list, batch_size):
    for i in range(0, len(file_list), batch_size):
        yield file_list[i:i + batch_size]

def save_batch_to_hdf5(rank, batch, filename):
    logger.info(f'rank {rank} starts to save {filename}')
    with h5.File(filename,'w') as h5f:
        X = []
        Y = []
        for idx,file in enumerate(batch):
            with h5.File(file, 'r') as f:
                x = np.array(f['input'],dtype=np.float32)
                y = np.array(f['output'],dtype=np.float32)
            X.append(x)
            Y.append(y)
        chunk_x = np.stack(X,axis=0)
        chunk_y = np.stack(Y,axis=0)
        logger.info(f'rank {rank}: {filename} saved {chunk_x.shape},{chunk_y.shape}')
            # os.remove(file)
        h5f['input'] = chunk_x
        h5f['output'] = chunk_y
def parse_args():
    parser = argparse.ArgumentParser(description="Select dataset and override config options")
    parser.add_argument('--dataset', choices=['random', 'dummy','mulfreq', 'faceon'], required=True,
                        help="Specify the dataset to use: 'random', 'mulfreq', or 'faceon'")
    return parser.parse_args()

@timing_decorator
def train_test_split_main(clean_file_path,logger):
    args = parse_args()
    dataset_name = args.dataset
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        try:
            with open(clean_file_path, 'r') as f:
                    file_paths = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to open file {clean_file_path}. Please run outlier.py\n {e}")
            comm.Abort()  # This will terminate all processes

        # Shuffle the file paths
        random.shuffle(file_paths)
        
        # Split into train and test sets (80-20 split)
        split_index = int(0.8 * len(file_paths))
        train_files = file_paths[:split_index]
        test_files = file_paths[split_index:]

        batch_size = 5000  # Adjust as needed based on memory constraints
        # Batch train and test files
        train_batches = list(batch_files(train_files, batch_size))
        test_batches = list(batch_files(test_files, batch_size))
    else:
        train_batches = None
        test_batches = None
    train_batches = comm.bcast(train_batches, root=0)
    test_batches = comm.bcast(test_batches, root=0)

    all_batches = train_batches + test_batches
    total_batches = len(all_batches)
    logger.info(f'total_batches {total_batches}')
    batches_per_rank = total_batches // size
    extra_batches = total_batches % size

    if rank < extra_batches:
        start_index = rank * (batches_per_rank + 1)
        end_index = start_index + batches_per_rank + 1
    else:
        start_index = rank * batches_per_rank + extra_batches
        end_index = start_index + batches_per_rank
    logger.info(f'rank {rank} does job {start_index} - {end_index}')
    for i in range(start_index, end_index):
        batch = all_batches[i]
        if i < len(train_batches):
            if dataset_name == 'mulfreq':
                folder = 'mul_freq'
            else:
                folder = 'sgl_freq'
            filename = f"/home/dc-su2/rds/rds-dirac-dp012/dc-su2/physical_forward/{folder}/grid64/rotation3/train_{i}.hdf5"
        else:
            if dataset_name == 'mulfreq':
                folder = 'mul_freq'
            else:
                folder = 'sgl_freq'
            test_idx = i - len(train_batches)
            filename = f"/home/dc-su2/rds/rds-dirac-dp012/dc-su2/physical_forward/{folder}/grid64/rotation3/test_{test_idx}.hdf5"
        logger.info(f'saved {filename}')
        # save_batch_to_hdf5(rank, batch, filename)

