import torch
import numpy as np
import h5py as h5
from torch.utils.data import TensorDataset
from utils.dataloader     import CustomTransform,IntensityDataset
# Assume CustomTransform and IntensityDataset are defined elsewhere and are imported correctly

def load_face_on_view_grid64():
    file_statistics = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/clean_statistics.pkl'
    train_file_path = [f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/sgl_freq/grid64/faceon/clean_train_{i}.hdf5' for i in range(4)]
    test_file_path = ['/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/sgl_freq/grid64/faceon/clean_vali.hdf5',
                      '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/sgl_freq/grid64/faceon/clean_test.hdf5']

    custom_transform = CustomTransform(file_statistics)
    train_dataset = IntensityDataset(train_file_path, transform=custom_transform)
    test_dataset = IntensityDataset(test_file_path, transform=custom_transform)
    return train_dataset, test_dataset

def load_random_view_grid64():
    path = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/sgl_freq/grid64/random/clean_batches.hdf5'
    with h5.File(path, 'r') as file:
        x = np.array(file['input'], np.float32)
        y = np.array(file['output'], np.float32)
    xtr, xte = x[:17500],x[17500:]
    ytr,yte  = y[:17500],y[17500:]
    train_dataset = TensorDataset(torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.float32))
    return train_dataset, test_dataset

def load_face_on_view_grid32():
    path = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/sgl_freq/grid32/faceon/clean_batches.hdf5'
    with h5.File(path, 'r') as file:
        x = np.array(file['input'], np.float32)
        y = np.array(file['output'], np.float32)
    xtr, xte = x[:9000], x[9000:]
    ytr, yte = y[:9000], y[9000:]
    train_dataset = TensorDataset(torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.float32))
    return train_dataset, test_dataset

def load_random_view_grid32():
    path = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/sgl_freq/grid32/rotation/clean_batches.hdf5'
    with h5.File(path, 'r') as file:
        x = np.array(file['input'], np.float32)
        y = np.array(file['output'], np.float32)
    xtr, xte = x[:26102], x[26102:]
    ytr, yte = y[:26102], y[26102:]
    train_dataset = TensorDataset(torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.float32))
    return train_dataset, test_dataset

def load_face_on_view_grid128():
    path = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/sgl_freq/grid128/faceon/clean_batches.hdf5'
    with h5.File(path, 'r') as file:
        x = np.array(file['input'], np.float32)
        y = np.array(file['output'], np.float32)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    xtr, xte = x_shuffled[:640], x_shuffled[640:]
    ytr, yte = y_shuffled[:640], y_shuffled[640:]
    train_dataset = TensorDataset(torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.float32))
    return train_dataset, test_dataset

def load_random_view_grid128():
    path = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/sgl_freq/grid128/rotation/clean_batches.hdf5'
    with h5.File(path, 'r') as file:
        x = np.array(file['input'], np.float32)
        y = np.array(file['output'], np.float32)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    xtr, xte = x_shuffled[:1920], x_shuffled[1920:]
    ytr, yte = y_shuffled[:1920], y[1920:]
    train_dataset = TensorDataset(torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.float32))
    return train_dataset, test_dataset






