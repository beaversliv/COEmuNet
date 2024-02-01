import torch
import torch.nn as nn
import torchvision
from thop import profile
import sys
from utils.so3_model      import ClsSO3Net
import h5py as h5
import numpy as np
def get_data(path):
    sample = h5.File(path,'r')
    x  = np.array(sample['input'],np.float32)   # shape(100,3,100,100,100)
    y = np.array(sample['output'][:,:,:,15:16], np.float32)# shape(100,1,256,256)
    
    meta = {}

    x_t = np.transpose(x, (1, 0, 2, 3, 4))
    for idx in [0]:
        meta[idx] = {}
        meta[idx]['mean'] = x_t[idx].mean()
        meta[idx]['std'] = x_t[idx].std()
        x_t[idx] = (x_t[idx] - x_t[idx].mean())/x_t[idx].std()
    
    for idx in [1, 2]:
        meta[idx] = {}
        meta[idx]['min'] = np.min(x_t[idx])
        meta[idx]['median'] = np.median(x_t[idx])
        x_t[idx] = np.log(x_t[idx])
        
        x_t[idx] = x_t[idx] - np.min(x_t[idx])
        x_t[idx] = x_t[idx]/np.median(x_t[idx])
    
    y_v = y.reshape(-1)
    y = np.where(y == 0, np.min(y_v[y_v != 0]), y)
    y = np.log(y)
    y = y-np.min(y)
    y = y/np.median(y)
    
    return np.transpose(x_t, (1, 0, 2, 3, 4)), np.transpose(y,(0,3,1,2))


 

model = ClsSO3Net()
 

path2 = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/data_augment/rotate1200.hdf5'
x, y = get_data(path2)
x0 = torch.tensor(x[0:1],dtype=torch.float32)
# Use thop to profile the model and get the number of trainable parameters
flops, params = profile(model, inputs=(x0,))
print(f"{model}, {params}, {flops}")
  