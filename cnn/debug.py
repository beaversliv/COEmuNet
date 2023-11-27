import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from utils.dataloader     import CustomTransform,IntensityDataset
from utils.model          import Net,VGGFeatures
from utils.loss           import mean_absolute_percentage_error, calculate_ssim_batch
from utils.focal_frequency_loss import FocalFrequencyLoss

import h5py as h5
import numpy as np
from tqdm import tqdm
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
path2 = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/Batches/rotate_1200.hdf5'

x, y = get_data(path2)
xtr,xte = x[:1000],x[1000:]
ytr,yte = y[:1000],y[1000:]

xtr = torch.Tensor(xtr)
ytr = torch.Tensor(ytr)
xte = torch.Tensor(xte)
yte = torch.Tensor(yte)

train_dataset = TensorDataset(xtr, ytr)
test_dataset = TensorDataset(xte, yte)

### torch data loader ###
train_dataloader = DataLoader(train_dataset, batch_size= config['batch_size'], shuffle=True)
val_dataloader = DataLoader(test_dataset, batch_size= config['batch_size'], shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### set a model ###
model = Net()
# model = nn.DataParallel(model,device_ids=[0,1])
model.to(device)

### Pre-trained VGG16 ###
# vgg = VGGFeatures()
# vgg.to(device)
# vgg.eval()  # Important to set in evaluation mode!

def loss_object(target,pred):
    mse_loss = nn.functional.mse_loss(pred, target)
    freq_loss_scale = 1.0
    # target_freq = torch.fft.fft2(target)
    # pred_freq = torch.fft.fft2(pred)
    # freq_loss = torch.mean(torch.abs(target_freq - pred_freq)**2)
    ffl = FocalFrequencyLoss(loss_weight=1.0,alpha=1.0)
    freq_loss = ffl(target,pred)
    loss = mse_loss + freq_loss_scale * freq_loss

    return loss
optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], betas=(0.9, 0.999))
#setp_size_up = [2-10] * (10903/16)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1,step_size_up=1500,mode='triangular',cycle_momentum=False)
### train step ###
def train(epoch):
    total_loss = 0.
    model.train() 
    for bidx, samples in enumerate(train_dataloader):
        data, target = Variable(samples[0]).to(device), Variable(samples[1]).to(device)

        optimizer.zero_grad()
        latent,output = model(data)
        # repeat the grayscale channel to create a 3-channel input
        # output = output.repeat(1,3,1,1)
        # target = target.repeat(1,3,1,1)

        # generated_features = vgg(output)
        # target_features    = vgg(target)
        # loss = perceptual_loss(generated_features, target_features)
        loss = loss_object(target,output)

        loss.backward()
        optimizer.step()
        # scheduler.step()
        total_loss += loss.detach().cpu().numpy()

    epoch_loss = total_loss/len(train_dataloader) # divide number of batches
    print('Train Epoch: {}/{} Loss: {:.4f}'.format(
            epoch, config['epochs'], epoch_loss))
    return epoch_loss

### vali step ###
def validation(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for bidx,samples in enumerate(val_dataloader):
            data, target = Variable(samples[0]).to(device), Variable(samples[1]).to(device)
            latent,output = model(data)
            loss = loss_object(output,target)
            val_loss += loss.detach().cpu().numpy()
    # Calculate average loss over all batches
    avg_val_loss = val_loss / len(val_dataloader)
    print('Val Epoch: {}/{} Loss: {:.4f}\n'.format(
            epoch, config['epochs'], avg_val_loss))
    return avg_val_loss

### test step ###
def test(epoch):
    model.eval()
    P = []
    T = []
    L = []
    for bidx, samples in enumerate(val_dataloader):
        data, target = Variable(samples[0]).to(device), Variable(samples[1]).to(device)
        latent, pred = model(data)
        loss = loss_object(target, pred)
        
        P.append(pred.detach().cpu().numpy())
        T.append(target.detach().cpu().numpy())
        L.append(loss.detach().cpu().numpy())
    
    print('Test Epoch: {}/{} Loss: {:.4f}'.format(
            epoch, config["epochs"], np.mean(L)))
    P = np.vstack(P)
    T = np.vstack(T)
    return P, T, L

### run ###
def run():
    tr_losses = []
    vl_losses = []
    for epoch in tqdm(range(config['epochs'])):
        epoch_loss = train(epoch)
        val_loss   = validation(epoch)
        tr_losses.append(epoch_loss)
        vl_losses.append(val_loss)
    return tr_losses,vl_losses

def main():
    start = time.time()
    tr_losses,vl_losses = run()
    end = time.time()
    print(f'running time:{(end-start)/60} mins')
    pred, target, _ = test(config['epochs'])
    mean_error, median_error = mean_absolute_percentage_error(target,pred)
    print('mean relative error: {:.4f}\n, median relative error: {:.4f}'.format(mean_error,median_error))
    avg_ssim = calculate_ssim_batch(target,pred)
    print('SSIM: {:.4f}'.format(avg_ssim))

    data = (tr_losses,vl_losses, pred, target)
    # Save to a pickle file
    with open("/home/dc-su2/physical_informed/cnn/rotate/results/history.pkl", "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    torch.save(model,'/home/dc-su2/physical_informed/cnn/rotate/results/model.pth')

    for i in range(0,200,10):
        fig, axs = plt.subplots(1, 2,figsize=(12, 5))
        im1 = axs[0].imshow(target[i][0],vmin=np.min(target[i][0]),vmax = np.max(target[i][0]))
        axs[0].set_title('target')
        fig.colorbar(im1,ax=axs[0])

        im2 = axs[1].imshow(pred[i][0],vmin=np.min(target[i][0]),vmax = np.max(target[i][0]))
        axs[1].set_title('prediction')
        fig.colorbar(im2,ax=axs[1])
        plt.savefig('/home/dc-su2/physical_informed/cnn/rotate/results/img/ex{}.png'.format(i))
        plt.close()

if __name__ == '__main__':
    main()