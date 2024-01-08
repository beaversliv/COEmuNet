import torch
from torch.utils.data import DataLoader, TensorDataset,DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable
from torch.profiler import profile, record_function, ProfilerActivity

from utils.so3_model      import ClsSO3Net
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
if torch.cuda.is_available():
    # Print the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    # Get the name of the current GPU (if there is more than one)
    current_gpu = torch.cuda.current_device()
    print(f"Name of the current GPU: {torch.cuda.get_device_name(current_gpu)}")
else:
    print("No GPU available, using CPU.")
np.random.seed(1234)
torch.manual_seed(1234)  

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--model_name', type = str, default = '3dResNet')
    parser.add_argument('--dataset', type = str, default = 'magritte')
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 32)
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
path2 = '/data/astro1/ss1421/physical_forward/cnn/Batches/rotate2400.hdf5'

x, y = get_data(path2)
xtr,xte = x[:2000],x[2000:]
ytr,yte = y[:2000],y[2000:]

xtr = torch.Tensor(xtr)
ytr = torch.Tensor(ytr)
xte = torch.Tensor(xte)
yte = torch.Tensor(yte)

train_dataset = TensorDataset(xtr, ytr)
test_dataset = TensorDataset(xte, yte)

### torch data loader ###
train_dataloader = DataLoader(train_dataset, batch_size= config['batch_size'], shuffle=True)
val_dataloader = DataLoader(test_dataset, batch_size= 4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### set a model ###
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
# def model_memory_size(model):
#     param_size = 0
#     for param in model.parameters():
#         param_size += param.nelement() * param.element_size()
#     buffer_size = 0
#     for buffer in model.buffers():
#         buffer_size += buffer.nelement() * buffer.element_size()
#     total_size = param_size + buffer_size
#     return total_size

# total_size = model_memory_size(model)
# print(f"Model size: {total_size / 1024 / 1024:.2f} MB")
# Model size: 636.90 MB

# num_params = count_parameters(model)
# print(f"Number of parameters: {num_params}")
# Number of parameters: 142767718

model = ClsSO3Net()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], betas=(0.9, 0.999))

### train step ###
def train(epoch):
    total_loss = 0.
    model.train()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof: 
        for bidx, samples in enumerate(train_dataloader):
            data, target = Variable(samples[0]).to(device), Variable(samples[1]).to(device)
            with record_function("model_training"):
                optimizer.zero_grad()
                latent,output = model(data)
            
                loss = loss_object(target, output, use_freq_loss=True, use_perceptual_loss=False)

                loss.backward()
                optimizer.step()
                # scheduler.step()
                total_loss += loss.detach().cpu().numpy()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
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
            loss = loss_object(target, output, use_freq_loss=True, use_perceptual_loss=False)
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
        loss = loss_object(target, pred, use_freq_loss=True, use_perceptual_loss=False)
        
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
        torch.cuda.empty_cache() # Clear cache after training
        
        val_loss   = validation(epoch)
        torch.cuda.empty_cache()  # Clear cache after evaluation
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
    with open("/home/s/ss1421/Documents/physical_informed_surrogate_model/cnn/steerable/history.pkl", "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    # torch.save(model,'/home/s/ss1421/Documents/physical_informed_surrogate_model/cnn/steerable/model.pth')

    for i in range(0,200,10):
        fig, axs = plt.subplots(1, 2,figsize=(12, 5))
        im1 = axs[0].imshow(target[i][0],vmin=np.min(target[i][0]),vmax = np.max(target[i][0]))
        axs[0].set_title('target')
        fig.colorbar(im1,ax=axs[0])

        im2 = axs[1].imshow(pred[i][0],vmin=np.min(target[i][0]),vmax = np.max(target[i][0]))
        axs[1].set_title('prediction')
        fig.colorbar(im2,ax=axs[1])
        plt.savefig('/home/s/ss1421/Documents/physical_informed_surrogate_model/cnn/steerable/img/ex{}.png'.format(i))
        plt.close()

if __name__ == '__main__':
    main()