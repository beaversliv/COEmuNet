import torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from dataloader     import CustomTransform,IntensityDataset
from Model          import Net
from focal_frequency_loss import FocalFrequencyLoss

import h5py as h5
import numpy as np
import os
import sys
import time
import pickle
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt

np.random.seed(1234)
torch.manual_seed(1234)  

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--model_name', type = str, default = '3dResNet')
    parser.add_argument('--dataset', type = str, default = 'magritte')
    parser.add_argument('--epochs', type = int, default = 200)
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--lr_decay', type = float, default = 0.95)


    args = parser.parse_args()
    
    config = OrderedDict([
            ('path_dir', args.path_dir),
            ('model_name', args.model_name),
            
            ('dataset', args.dataset),
            ('epochs', args.epochs),
            ('batch_size', args.batch_size),
            ('lr', args.lr),
            ('lr_decay', args.lr_decay)
            ])
    
    return config

config = parse_args()


def get_data(path):
    sample = h5.File(path,'r')
    x  = sample['input']   # shape(100,3,100,100,100)
    y = sample['output'][:,:,:,15:16] # shape(100,1,256,256)
    
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
# path = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/Batches/rotate.hdf5'
path2 = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/Batches/unrotate.hdf5'

x, y = get_data(path2)
xtr,xte = x[:800],x[800:]
ytr,yte = y[:800],y[800:]

xtr = torch.Tensor(xtr)
ytr = torch.Tensor(ytr)
xte = torch.Tensor(xte)
yte = torch.Tensor(yte)

train_dataset = TensorDataset(xtr, ytr)
test_dataset = TensorDataset(xte, yte)

### torch data loader ###
train_dataloader = DataLoader(train_dataset, batch_size= config['batch_size'], shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size= 32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### set a model ###
model = Net()
model.to(device)  

def loss_object(target,pred):
    # discrete fourier transform
    target_freq = torch.fft.fft2(target)
    pred_freq = torch.fft.fft2(pred)
    # MAE in frequency domain
    loss_freq = torch.mean(torch.abs(target_freq-pred_freq))
    # MSE in spatial domain
    loss_mse  = nn.functional.mse_loss(pred,target)

    freq_loss_scale = 1.0
    mse_loss_scale = 1.0
    loss = mse_loss_scale*loss_mse +freq_loss_scale * loss_freq
    # loss = mse_loss_scale*nn.functional.mse_loss(pred,target) + mge_loss_scale*mge_loss(pred, target)
    #     loss_mse = nn.MSELoss()
#     loss_ff = FocalFrequencyLoss(loss_weight=2.0,alpha=2.0)
#     return 0.8 * loss_mse(target,pred) + 0.2 * loss_ff(target,pred)
    return loss

# loss_object = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], betas=(0.9, 0.999))

### train step ###
def train(epoch):
    epoch_loss = 0.
    model.train() 
    for bidx, samples in enumerate(train_dataloader):
        data, target = Variable(samples[0]).to(device), Variable(samples[1]).to(device)

        optimizer.zero_grad()
        latent,output = model(data)
        loss = loss_object(target,output)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().cpu().numpy()
    print('Train Epoch: {}/{} Loss: {:.4f}'.format(
            epoch, config['epochs'], epoch_loss))
    return epoch_loss

### test/val step ###
def test(epoch):
    model.eval()
    P = []
    T = []
    L = []
    for bidx, samples in enumerate(test_dataloader):
        data, target = Variable(samples[0]).to(device), Variable(samples[1]).to(device)
        latent,pred = model(data)
        loss = loss_object(target,pred)
        
        P.append(pred.detach().cpu().numpy())
        T.append(target.detach().cpu().numpy())
        L.append(loss.detach().cpu().numpy())
    
    print('Test Epoch: {}/{} Loss: {:.4f}\n'.format(
            epoch, config['epochs'], np.mean(L)))
    P = np.vstack(P)
    T = np.vstack(T)
    return P, T, L
### run ###
def run():
    losses = []
    vl     = []
    for epoch in range(config['epochs']):
        epoch_loss = train(epoch)
        _,_,val_loss   = test(epoch)
        losses.append(epoch_loss)
        vl.append(np.mean(val_loss))
    return losses,vl

def main():
    start = time.time()
    losses,vl = run()
    end = time.time()
    print(f'running time:{(end-start)/60} mins')
    pred, target, _ = test(config['epochs'])

    data = (losses, vl, pred, target)
    # Save to a pickle file
    with open("/home/dc-su2/physical_informed/cnn/results/history.pkl", "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    torch.save(model,'/home/dc-su2/physical_informed/cnn/results/model.pth')

    for i in range(0,200,10):
        fig, axs = plt.subplots(1, 2,figsize=(12, 5))
        im1 = axs[0].imshow(target[i][0],vmin=np.min(target[i][0]),vmax = np.max(target[i][0]))
        axs[0].set_title('target')
        fig.colorbar(im1,ax=axs[0])

        im2 = axs[1].imshow(pred[i][0],vmin=np.min(target[i][0]),vmax = np.max(target[i][0]))
        axs[1].set_title('prediction')
        fig.colorbar(im2,ax=axs[1])
        plt.savefig('/home/dc-su2/physical_informed/cnn/results/img/ex{}.png'.format(i))
        plt.close()

if __name__ == '__main__':
    main()