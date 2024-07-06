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

# def generate_cumulative_offsets(files):
#     """Generate cumulative offsets for each file to map global indices."""
#     offsets = []
#     total = 0
#     for file in files:
#         with h5py.File(file, 'r') as h5f:
#             num_samples = h5f['input'].shape[0]
#             offsets.append(total)
#             total += num_samples
#     return offsets

# def map_indices_to_files(indices, offsets):
#     """Map global indices to file-specific indices and file number."""
#     file_mapped_indices = {}
#     for global_idx in indices:
#         # Find the file to which the index belongs using the offsets
#         file_idx = next(i for i, offset in enumerate(offsets) if global_idx < offset + h5py.File(files[i], 'r')['input'].shape[0])
#         local_idx = global_idx - offsets[file_idx]
#         if file_idx not in file_mapped_indices:
#             file_mapped_indices[file_idx] = []
#         file_mapped_indices[file_idx].append(local_idx)
#     return file_mapped_indices

# def split_and_save(files, indices, train_prefix, test_output, max_train_size=15 * 1024 * 1024 * 1024): 
#     """Split data into training and testing sets, saving results to multiple training files."""
#     train_data = []
#     test_data = []

#     train_indices_mapped = map_indices_to_files(indices['train'], generate_cumulative_offsets(files))
#     test_indices_mapped = map_indices_to_files(indices['test'], generate_cumulative_offsets(files))

#     # Retrieve training data
#     current_train_file = 0
#     current_train_size = 0
#     for file_idx, file in enumerate(files):
#         with h5py.File(file, 'r') as h5f:
#             input_data = np.array(h5f['input'], np.float64)
#             output_data = np.array(h5f['output'], np.float64)

#             if file_idx in train_indices_mapped:
#                 for idx in train_indices_mapped[file_idx]:
#                     train_data.append((input_data[idx], output_data[idx]))
#                     current_train_size += input_data[idx].nbytes + output_data[idx].nbytes

#                     if current_train_size >= max_train_size:
#                         train_inputs, train_outputs = zip(*train_data)
#                         with h5py.File(f"{train_prefix}_{current_train_file}.h5", 'w') as h5f_out:
#                             h5f_out.create_dataset('input', data=np.array(train_inputs))
#                             h5f_out.create_dataset('output', data=np.array(train_outputs))

#                         current_train_file += 1
#                         current_train_size = 0
#                         train_data = []

#             if file_idx in test_indices_mapped:
#                 for idx in test_indices_mapped[file_idx]:
#                     test_data.append((input_data[idx], output_data[idx]))

#     # Save any remaining training data
#     if train_data:
#         train_inputs, train_outputs = zip(*train_data)
#         with h5py.File(f"{train_prefix}_{current_train_file}.h5", 'w') as h5f_out:
#             h5f_out.create_dataset('input', data=np.array(train_inputs))
#             h5f_out.create_dataset('output', data=np.array(train_outputs))

#     # Save testing data to a single HDF5 file
#     test_inputs, test_outputs = zip(*test_data)
#     with h5py.File(test_output, 'w') as h5f_out:
#         h5f_out.create_dataset('input', data=np.array(test_inputs))
#         h5f_out.create_dataset('output', data=np.array(test_outputs))

# def generate_indices(files, test_size=0.2, random_state=42):
#     """Generate train-test split indices across all files."""
#     total_samples = sum(h5py.File(file, 'r')['input'].shape[0] for file in files)
#     all_indices = np.arange(total_samples)
#     train_idx, test_idx = train_test_split(all_indices, test_size=test_size, random_state=random_state)

#     return {'train': train_idx, 'test': test_idx}

if __name__ =='__main__':
    main()
    # files = ['/home/dc-su2/rds/rds-dirac-dr004/Magritte/dummy.hdf5', '/home/dc-su2/rds/rds-dirac-dr004/Magritte/dummy1.hdf5','/home/dc-su2/rds/rds-dirac-dr004/Magritte/dummy2.hdf5']  # Replace with your actual file paths
    # indices = generate_indices(files)
    # split_and_save(files, indices, 'train', 'test.h5')
