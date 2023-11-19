import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from Model          import Net

import h5py     as h5
import numpy    as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
    
def get_data(path):
    with h5.File(path,'r') as sample:
        x  = np.array(sample['input'])   # shape(1000,3,64,64,64)
        y =  np.array(sample['output'][:,:,:,15:16]) # shape(1000,64,64,1)
    
    meta = {}
    
    x_t = np.transpose(x, (1, 0, 2, 3, 4))
    for idx in [0]:
        meta[idx] = {}
        meta[idx]['mean'] = x_t[idx].mean()
        meta[idx]['std'] = x_t[idx].std()
        x_t[idx] = (x_t[idx]-x_t[idx].mean())/x_t[idx].std()
    # idx = 1
    # x_t[idx] = np.exp(x_t[idx])
    
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
    
    return np.transpose(x_t, (1, 0, 2, 3, 4)), y.transpose(0,3,1,2)
    
### data pre-processing ###
path = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/Batches/test.hdf5'
x, y = get_data(path)
x = torch.Tensor(x)
y = torch.Tensor(y)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### load ResNet ###
model = torch.load('/home/dc-su2/physical_informed/cnn/results/model.pth')
model.to(device) 
model.eval()
# Pass your data through the encoder
with torch.no_grad():
    data, target = Variable(x).to(device), Variable(y).to(device)
    # stacked (batch,32*4*4*4,3)
    # latent (batch, 16*16*16)
    latent_space,output= model(data)

latent_space = latent_space.cpu()
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=0)
tsne_result = tsne.fit_transform(latent_space)

plt.figure(figsize=(8, 8))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True)

plt.title("t-SNE Visualization")
plt.savefig('/home/dc-su2/physical_informed/cnn/results/latent space.png')
plt.show()
