import torch
from torch.utils.data import DataLoader, TensorDataset,DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable
from torch.profiler import profile, record_function, ProfilerActivity

from utils.so3_model      import ClsSO3Net
from utils.loss           import loss_object,mean_absolute_percentage_error, calculate_ssim_batch
from utils.plot           import img_plt

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

### train step ###
def train(model,train_dataloader,optimizer,config,device,epoch):
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

def test(model,test_dataloader,device,config,epoch):
    model.eval()
    P = []
    T = []
    L = []
    for bidx, samples in enumerate(test_dataloader):
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
    return P, T, np.mean(L)

### run ###
def run(model,train_dataloader,test_dataloader,optimizer,device,config):
    tr_losses = []
    vl_losses = []
    for epoch in tqdm(range(config['epochs'])):
        epoch_loss = train(model,train_dataloader,optimizer,config,device,epoch)
        torch.cuda.empty_cache() # Clear cache after training
        
        _,_,val_loss   = test(model,test_dataloader,device,config,epoch)
        torch.cuda.empty_cache()  # Clear cache after evaluation
        tr_losses.append(epoch_loss)
        vl_losses.append(val_loss)
    return tr_losses,vl_losses

def ddp_run(rank, world_size, model,device,config,train_dataloader, val_dataloader,optimizer):
    # Setup for DDP
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    ddp_model = DDP(model, device_ids=[rank])

    tr_losses, vl_losses = run(ddp_model, train_dataloader, val_dataloader, loss_object, optimizer, config, device)
    if rank == 0:
        gathered_tr_losses = [torch.zeros_like(tr_losses) for _ in range(world_size)]
        gathered_vl_losses = [torch.zeros_like(vl_losses) for _ in range(world_size)]
        dist.gather(tensor=tr_losses, gather_list=gathered_tr_losses, dst=0)
        dist.gather(tensor=vl_losses, gather_list=gathered_vl_losses, dst=0)
         # Combine data from all processes
        combined_tr_losses = torch.cat(gathered_tr_losses).tolist()
        combined_vl_losses = torch.cat(gathered_vl_losses).tolist()
        data = (combined_tr_losses,combined_vl_losses)

        with open("/home/s/ss1421/Documents/physical_informed_surrogate_model/cnn/steerable/history.pkl", "wb") as pickle_file:
            pickle.dump(data, pickle_file)
    else:
        # Other processes send their data
        dist.gather(tensor=tr_losses, gather_list=None, dst=0)
        dist.gather(tensor=vl_losses, gather_list=None, dst=0)
    # Cleanup
    dist.destroy_process_group()
    if rank == 0:
        return combined_tr_losses, combined_vl_losses

def main():
    config = parse_args()
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
    test_dataloader = DataLoader(test_dataset, batch_size= 4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    model = ClsSO3Net()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], betas=(0.9, 0.999))

    ### start training ###
    start = time.time()
    world_size = torch.cuda.device_count()
    mp.spawn(ddp_run, args=(world_size, model, device, config, train_dataloader, test_dataloader, optimizer), nprocs=world_size, join=True)
    end = time.time()

    print(f'running time:{(end-start)/60} mins')
    
    
    ### validation ###
    pred, target, _ = test(config['epochs'])
    pred_targ = (pred, target)

    with open("/home/s/ss1421/Documents/physical_informed_surrogate_model/cnn/steerable/history.pkl", "rb") as pickle_file:
        losses = pickle.load(pickle_file)
    
    losses.update(pred_targ)
    with open("/home/s/ss1421/Documents/physical_informed_surrogate_model/cnn/steerable/history.pkl", "wb") as pickle_file:
        pickle.dump(losses, pickle_file)

    mean_error, median_error = mean_absolute_percentage_error(target,pred)
    print('mean relative error: {:.4f}\n, median relative error: {:.4f}'.format(mean_error,median_error))
    avg_ssim = calculate_ssim_batch(target,pred)
    print('SSIM: {:.4f}'.format(avg_ssim))

if __name__ == '__main__':
    main()