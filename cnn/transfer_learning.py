import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
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
    parser.add_argument('--batch_size', type = int, default = 32)
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
path = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/Batches/rotate.hdf5'
x, y = get_data(path)

# xtr = torch.Tensor(x[:20]+x[200:220]+x[400:420])
# ytr = torch.Tensor(y[:20]+y[200:220]+y[400:420])
# xte = torch.Tensor(x[20:200]+x[220:400]+x[420:])
# yte = torch.Tensor(y[20:200]+y[220:400]+y[420:])

xtr,ytr = torch.Tensor(x),torch.Tensor(y)
train_dataset = TensorDataset(xtr, ytr)
train_dataloader = DataLoader(train_dataset, batch_size= config['batch_size'], shuffle=True)

# test_dataset = TensorDataset(xte, yte)
test_dataloader = DataLoader(train_dataset, batch_size= 16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### load ResNet ###
model = torch.load('/home/dc-su2/physical_informed/cnn/results/unrotate_Allmodel.pth')
model.to(device) 

optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], betas=(0.9, 0.999))

# def fourier_loss(target,reconstructed):
#     # Fourier Transform
#     fft_reconstructed = torch.fft.fftn(reconstructed)
#     fft_target = torch.fft.fftn(target)

#     # High-pass filter can be applied here if needed

#     # Calculate loss
#     real_loss = F.mse_loss(fft_reconstructed.real, fft_target.real)
#     imag_loss = F.mse_loss(fft_reconstructed.imag, fft_target.imag)
#     return real_loss + imag_loss
# def loss_object(target,reconstructed):
#     '''
#     Loss function for ResNet : MSE + high frequency loss

#     input: predict image, target image
#     return: total loss 
#     '''
#     alpha = 1.0
#     beta  = 1.5
#     mse = F.mse_loss(target,reconstructed)
#     freq_loss = fourier_loss(target,reconstructed)
#     total_loss = alpha * mse + beta * freq_loss
#     return total_loss

loss_object = FocalFrequencyLoss(loss_weight=2.0,alpha=0.5)

### train step ###
def train(epoch):
    epoch_loss = 0.
    model.train() 
    for bidx, samples in enumerate(train_dataloader):
        data, target = Variable(samples[0]).to(device), Variable(samples[1]).to(device)

        optimizer.zero_grad()
        latent,output = model(data)
        loss = loss_object(target, output)
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
        loss = loss_object(target, pred)
        
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
    with open("/home/dc-su2/physical_informed/cnn/results/rotate600.pkl", "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    torch.save(model,'/home/dc-su2/physical_informed/cnn/results/rotate600.pth')

    for i in range(0,60,5):
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