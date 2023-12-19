import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from utils.dataloader     import CustomTransform,IntensityDataset
from utils.model          import Net
from utils.loss           import loss_object,mean_absolute_percentage_error, calculate_ssim_batch

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
print('original face on view')
# Check if a GPU is available
if torch.cuda.is_available():
    # Print the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    # Get the name of the current GPU (if there is more than one)
    current_gpu = torch.cuda.current_device()
    print(f"Name of the current GPU: {torch.cuda.get_device_name(current_gpu)}")
else:
    print("No GPU available, using CPU.")


# Global Constants
np.random.seed(1234)
torch.manual_seed(1234) 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--model_name', type = str, default = '3dResNet')
    parser.add_argument('--dataset', type = str, default = 'p3droslo')
    parser.add_argument('--epochs', type = int, default = 200)
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--lr', type = float, default = 1e-4)
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
# file paths for train, vali and test
file_statistics = '/home/dc-su2/physical_informed/cnn/original/unrotate_statistics.pkl'
train_file_path = [f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/Batches/train_{i}.hdf5' for i in range(4)]
vali_file_path  = ['/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/Batches/vali.hdf5']
test_file_path  = ['/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/Batches/test.hdf5']

custom_transform = CustomTransform(file_statistics)
train_dataset= IntensityDataset(train_file_path,transform=custom_transform)
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

vali_dataset= IntensityDataset(vali_file_path,transform=custom_transform)
vali_dataloader = DataLoader(vali_dataset, batch_size=config['batch_size'], shuffle=True)

test_dataset= IntensityDataset(test_file_path,transform=custom_transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### set a model ###
model = Net()
# model = nn.DataParallel(model,device_ids=[0,1,2])
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], weight_decay=1e-2, betas=(0.9, 0.999))

### train step ###
def train(epoch):
    total_loss = 0.
    model.train() 
    for bidx, samples in enumerate(train_dataloader):
        data, target = Variable(samples[0]).to(device), Variable(samples[1]).to(device)

        optimizer.zero_grad()
        latent,output = model(data)
        loss = loss_object(target,output,use_freq_loss=True, use_perceptual_loss=False)

        loss.backward()
        optimizer.step()
        total_loss += loss.detach().cpu().numpy()

    epoch_loss = total_loss/len(train_dataloader) # divide number of batches
    print('Train Epoch: {}/{} Loss: {:.4f}'.format(
            epoch, config['epochs'], epoch_loss))
    return epoch_loss

### test/val step ###
def test(epoch,dataloader):
    model.eval()
    P = []
    T = []
    L = []
    for bidx, samples in enumerate(dataloader):
        data, target = Variable(samples[0]).to(device), Variable(samples[1]).to(device)
        latent,pred = model(data)
        loss = loss_object(target, pred,use_freq_loss=True, use_perceptual_loss=False)
        
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
    for epoch in tqdm(range(config['epochs'])):
        epoch_loss = train(epoch)
        _,_,val_loss   = test(epoch,vali_dataloader)
        losses.append(epoch_loss)
        vl.append(np.mean(val_loss))
    return losses,vl

def main():
    start = time.time()
    losses,vl = run()
    end = time.time()
    print(f'running time:{(end-start)/60} mins\n')
    pred, target, _ = test(config['epochs'],test_dataloader)

    ### other quantitive measure ###
    mean_error, median_error = mean_absolute_percentage_error(target,pred)
    print('mean relative error: {:.4f}\n, median relative error: {:.4f}'.format(mean_error,median_error))
    avg_ssim = calculate_ssim_batch(target,pred)
    print('SSIM: {:.4f}'.format(avg_ssim))

    data = (losses, vl, pred, target)
    # Save to a pickle file
    with open("/home/dc-su2/physical_informed/cnn/original/results/history.pkl", "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    torch.save(model,'/home/dc-su2/physical_informed/cnn/original/results/model.pth')

    for i in range(0,600,50):
        fig, axs = plt.subplots(1, 2,figsize=(12, 5))
        im1 = axs[0].imshow(target[i][0],vmin=np.min(target[i][0]),vmax = np.max(target[i][0]))
        axs[0].set_title('target')
        fig.colorbar(im1,ax=axs[0])

        im2 = axs[1].imshow(pred[i][0],vmin=np.min(target[i][0]),vmax = np.max(target[i][0]))
        axs[1].set_title('prediction')
        fig.colorbar(im2,ax=axs[1])
        plt.savefig('/home/dc-su2/physical_informed/cnn/original/results/img/ex{}.png'.format(i))
        plt.close()

if __name__ == '__main__':
    main()
