import torch
import numpy as np
import h5py as h5
from torch.utils.data import TensorDataset
from utils.dataloader     import CustomTransform,IntensityDataset
from utils.preprocessing  import preProcessing
# Assume CustomTransform and IntensityDataset are defined elsewhere and are imported correctly
from sklearn.model_selection      import train_test_split

def dataset_gen(args):
    if config['model_grid'] == 32:
        if config['dataset'] == 'faceon':
            path = '/home/dc-su2/rds/rds-dirac-dr004/Magritte/faceon_grid32_data0.hdf5'
        elif config['dataset'] == 'random':
            path = '/home/dc-su2/rds/rds-dirac-dr004/Magritte/random_grid32_data0.hdf5'
    if config['model_grid'] == 64:
        if config['dataset'] == 'faceon':
            path = '/home/dc-su2/rds/rds-dirac-dr004/Magritte/faceon_grid64_data0.hdf5'
        elif config['dataset'] == 'random':
            path = '/home/dc-su2/rds/rds-dirac-dr004/Magritte/random_grid64_data0.hdf5'
    if config['model_grid'] == 128:
        if config['dataset'] == 'faceon':
            path = '/home/dc-su2/rds/rds-dirac-dr004/Magritte/faceon_grid128_data0.hdf5'
        elif config['dataset'] == 'random':
            path = '/home/dc-su2/rds/rds-dirac-dr004/Magritte/random_grid128_data0.hdf5'
    data_gen = preProcessing(path)
    x,y = data_gen()
    xtr, xte, ytr,yte = train_test_split(x,y,test_size=0.2,random_state=42)
    xtr = torch.tensor(xtr,dtype=torch.float32)
    ytr = torch.tensor(ytr,dtype=torch.float32)
    xte = torch.tensor(xte,dtype=torch.float32)
    yte = torch.tensor(yte,dtype=torch.float32)

    train_dataset = TensorDataset(xtr, ytr)
    test_dataset = TensorDataset(xte, yte)
    return train_dataset,test_dataset

def postProcessing(y):
    if config['model_grid'] == 32:
        if args.dataset == 'faceon':
            min_ = -47.387955
            median = 8.168968
            y = y*median + min_
            y = np.exp(y)
    return y



