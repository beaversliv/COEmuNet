import h5py as h5
import numpy as np
import random
import os
import time
from mpi4py import MPI
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'Function {func.__name__} took {end_time - start_time:.4f} seconds')
        return result
    return wrapper
def model_find():
    '''
    Absolute path of all original magritte models
    xxxx.hdf5
    '''
    model_files = []
    path = "/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/AMRVAC_3D/"
    for model_dir in os.listdir(path):
        # model_dir is modelxxx
        model_path = os.path.join(path,model_dir)
        for file_dir in os.listdir(model_path):
            #file_dir is 0789
            current_path = os.path.join(model_path,file_dir)
            model_file = os.path.join(current_path,f'{file_dir}.hdf5')

            model_files.append(model_file) 
    return model_files
def convert_to_rotation_files(model_file,rotation_idx):

    listb = model_file.split('/')
    listb[7] = 'physical_forward/sgl_freq/grid64/R1'
    listb[-1] = f'{listb[-2]}_{rotation_idx}.hdf5'
    rotation_file = ('/').join(listb)
    return rotation_file
def convert_to_faceon_files(model_file):
    listb = model_file.split('/')
    listb[7] = 'physical_forward/sgl_freq/grid64/Original'
    listb[-1] = f'{listb[-2]}.hdf5'
    faceon_file = ('/').join(listb)
    return faceon_file
def file_paths_gen(num_rotations=50):
    model_files = model_find()
    rotation_files = []
    for idx in range(10903 * num_rotations):
        file_idx = idx // num_rotations
        rotation_idx = idx % num_rotations
        model_file = model_files[file_idx]
        rotation_file = convert_to_rotation_files(model_file,rotation_idx)
        rotation_files.append(rotation_file)
    return rotation_files

def batch_files(file_list, batch_size):
    for i in range(0, len(file_list), batch_size):
        yield file_list[i:i + batch_size]

def save_batch_to_hdf5(batch, filename):
    grid, freq, num_input = 64, 1, 3
    with h5.File(filename,'w') as h5f:
        num_samples = len(batch)
        X = np.zeros((num_samples, num_input, grid, grid, grid))
        Y = np.zeros((num_samples, grid, grid, freq))
        for idx,file in enumerate(batch):
            with h5.File(file, 'r') as f:
                co = np.array(f['CO'])
                tmp = np.array(f['temperature'])
                v_z = np.array(f['velocity_z'])
                img = np.array(f['I'])
            X[idx] = np.stack((v_z, tmp, co), axis=0)
            Y[idx] = img
            
            os.remove(file)
        h5f['input'] = X
        h5f['output'] = Y

@timing_decorator
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        file_paths = file_paths_gen(num_rotations=50)
        # Shuffle the file paths
        random.shuffle(file_paths)
        
        # Split into train and test sets (80-20 split)
        split_index = int(0.8 * len(file_paths))
        train_files = file_paths[:split_index]
        test_files = file_paths[split_index:]

        batch_size = 10903  # Adjust as needed based on memory constraints
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

    batches_per_rank = total_batches // size
    extra_batches = total_batches % size

    if rank < extra_batches:
        start_index = rank * (batches_per_rank + 1)
        end_index = start_index + batches_per_rank + 1
    else:
        start_index = rank * batches_per_rank + extra_batches
        end_index = start_index + batches_per_rank

    for i in range(start_index, end_index):
        batch = all_batches[i]
        if i < len(train_batches):
            filename = f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/sgl_freq/grid64/Rotation/train_{i}.hdf5'
        else:
            test_idx = i - len(train_batches)
            filename = f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/sgl_freq/grid64/Rotation/test_{test_idx}.hdf5'
        save_batch_to_hdf5(batch, filename)
        print(f'Rank {rank} saved {filename}')
if __name__ =='__main__':
    main()