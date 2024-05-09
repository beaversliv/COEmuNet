import h5py
import numpy as np
from sklearn.model_selection import train_test_split

def generate_cumulative_offsets(files):
    """Generate cumulative offsets for each file to map global indices."""
    offsets = []
    total = 0
    for file in files:
        with h5py.File(file, 'r') as h5f:
            num_samples = h5f['input'].shape[0]
            offsets.append(total)
            total += num_samples
    return offsets

def map_indices_to_files(indices, offsets):
    """Map global indices to file-specific indices and file number."""
    file_mapped_indices = {}
    for global_idx in indices:
        # Find the file to which the index belongs using the offsets
        file_idx = next(i for i, offset in enumerate(offsets) if global_idx < offset + h5py.File(files[i], 'r')['input'].shape[0])
        local_idx = global_idx - offsets[file_idx]
        if file_idx not in file_mapped_indices:
            file_mapped_indices[file_idx] = []
        file_mapped_indices[file_idx].append(local_idx)
    return file_mapped_indices

def split_and_save(files, indices, train_prefix, test_output, max_train_size=15 * 1024 * 1024 * 1024):
    """Split data into training and testing sets, saving results to multiple training files."""
    train_data = []
    test_data = []

    train_indices_mapped = map_indices_to_files(indices['train'], generate_cumulative_offsets(files))
    test_indices_mapped = map_indices_to_files(indices['test'], generate_cumulative_offsets(files))

    # Retrieve training data
    current_train_file = 0
    current_train_size = 0
    for file_idx, file in enumerate(files):
        with h5py.File(file, 'r') as h5f:
            input_data = np.array(h5f['input'], np.float64)
            output_data = np.array(h5f['output'], np.float64)

            if file_idx in train_indices_mapped:
                for idx in train_indices_mapped[file_idx]:
                    train_data.append((input_data[idx], output_data[idx]))
                    current_train_size += input_data[idx].nbytes + output_data[idx].nbytes

                    if current_train_size >= max_train_size:
                        train_inputs, train_outputs = zip(*train_data)
                        with h5py.File(f"{train_prefix}_{current_train_file}.h5", 'w') as h5f_out:
                            h5f_out.create_dataset('input', data=np.array(train_inputs))
                            h5f_out.create_dataset('output', data=np.array(train_outputs))

                        current_train_file += 1
                        current_train_size = 0
                        train_data = []

            if file_idx in test_indices_mapped:
                for idx in test_indices_mapped[file_idx]:
                    test_data.append((input_data[idx], output_data[idx]))

    # Save any remaining training data
    if train_data:
        train_inputs, train_outputs = zip(*train_data)
        with h5py.File(f"{train_prefix}_{current_train_file}.h5", 'w') as h5f_out:
            h5f_out.create_dataset('input', data=np.array(train_inputs))
            h5f_out.create_dataset('output', data=np.array(train_outputs))

    # Save testing data to a single HDF5 file
    test_inputs, test_outputs = zip(*test_data)
    with h5py.File(test_output, 'w') as h5f_out:
        h5f_out.create_dataset('input', data=np.array(test_inputs))
        h5f_out.create_dataset('output', data=np.array(test_outputs))

def generate_indices(files, test_size=0.2, random_state=42):
    """Generate train-test split indices across all files."""
    total_samples = sum(h5py.File(file, 'r')['input'].shape[0] for file in files)
    all_indices = np.arange(total_samples)
    train_idx, test_idx = train_test_split(all_indices, test_size=test_size, random_state=random_state)

    return {'train': train_idx, 'test': test_idx}

if __name__ =='__main__':

    files = ['file1.h5', 'file2.h5']  # Replace with your actual file paths
    indices = generate_indices(files)
    split_and_save(files, indices, 'train', 'test.h5')
