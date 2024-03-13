import torch
import numpy as np
import h5py as h5
from torch.utils.data import TensorDataset
# Assume CustomTransform and IntensityDataset are defined elsewhere and are imported correctly

def load_face_on_view_grid64():
    file_statistics = '/home/dc-su2/physical_informed/cnn/original/clean_statistics.pkl'
    train_file_path = [f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/faceon/new_train_{i}.hdf5' for i in range(4)]
    test_file_path = ['/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/faceon/new_vali.hdf5',
                      '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/faceon/new_test.hdf5']

    custom_transform = CustomTransform(file_statistics)
    train_dataset = IntensityDataset(train_file_path, transform=custom_transform)
    test_dataset = IntensityDataset(test_file_path, transform=custom_transform)
    return train_dataset, test_dataset

def load_random_view_grid64_12000():
    file_statistics = '/home/dc-su2/physical_informed/cnn/rotate/12000_statistics.pkl'
    file_paths = [f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/data_augment/clean_rotate12000_{i}.hdf5' for i in range(5)]
    train_file_path = file_paths[:4]
    test_file_path = file_paths[4:]

    custom_transform = CustomTransform(file_statistics)
    train_dataset = IntensityDataset(train_file_path, transform=custom_transform)
    test_dataset = IntensityDataset(test_file_path, transform=custom_transform)
    return train_dataset, test_dataset

def load_face_on_view_grid32():
    path = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/grid32/faceon/clean_batches.hdf5'
    with h5.File(path, 'r') as file:
        x = np.array(file['input'], np.float32)
        y = np.array(file['output'], np.float32)
    xtr, xte = x[:9000], x[9000:]
    ytr, yte = y[:9000], y[9000:]
    train_dataset = TensorDataset(torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.float32))
    return train_dataset, test_dataset

def load_random_view_grid32():
    path = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/grid32/rotation/clean_batches.hdf5'
    with h5.File(path, 'r') as file:
        x = np.array(file['input'], np.float32)
        y = np.array(file['output'], np.float32)
    xtr, xte = x[:26102], x[26102:]
    ytr, yte = y[:26102], y[26102:]
    train_dataset = TensorDataset(torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.float32))
    return train_dataset, test_dataset






