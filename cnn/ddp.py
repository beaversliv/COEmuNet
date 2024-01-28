import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset,DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


from torch.profiler import profile, record_function, ProfilerActivity

from utils.so3_model      import ClsSO3Net
from utils.loss           import Lossfunction,ResNetFeatures,mean_absolute_percentage_error, calculate_ssim_batch
from utils.plot           import img_plt

import h5py as h5
import numpy as np
from tqdm import tqdm
import os
import sys
import json
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

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def setup(rank, world_size):
    "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--model_name', type = str, default = '3dResNet')
    parser.add_argument('--dataset', type = str, default = 'magritte')
    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--batch_size', type = int, default = 16) # each GPU handle 16 samples
    parser.add_argument('--lr', type = float, default = 2e-3)
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
    x = np.array(sample['input'],np.float32)   # shape(100,3,100,100,100)
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

### train step ###
class Trainer:
    def __init__(self,ddp_model,train_dataloader,test_dataloader,config,rank,world_size):
        self.ddp_model = ddp_model
        self.train_dataloader = train_dataloader
        self.test_dataloader  = test_dataloader

        self.config = config
        self.rank   = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
        # Define the optimizer for the DDP model
        self.optimizer = torch.optim.Adam(ddp_model.parameters(), lr=self.config['lr'], betas=(0.9, 0.999))

        # Initialize ResNetFeatures and move it to the correct device
        self.resnet18 = ResNetFeatures().to(rank).eval()  # Set to evaluation mode
        # Initialize custom loss function with ResNet18 feature extractor
        self.loss_object = Lossfunction(self.resnet18,use_freq_loss=False,use_perceptual_loss=False,
                                        mse_loss_scale = 1.0,freq_loss_scale=1.0, perceptual_loss_scale=1.0)
    def train(self):
        total_loss = 0.
        self.ddp_model.train()
        
        for bidx, samples in enumerate(self.train_dataloader):
            data, target = Variable(samples[0]).to(self.rank), Variable(samples[1]).to(self.rank)
            self.optimizer.zero_grad()
            latent, output = self.ddp_model(data)
            loss = self.loss_object(target, output)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach().cpu().numpy()
        epoch_loss = total_loss / len(self.train_dataloader)  # divide number of batches
        return epoch_loss

    
    def test(self):
        self.ddp_model.eval()
        P = []
        T = []
        L = []
        for bidx, samples in enumerate(self.test_dataloader):
            data, target = Variable(samples[0]).to(self.rank), Variable(samples[1]).to(self.rank)
            latent, pred = self.ddp_model(data)
            loss = self.loss_object(target, pred)
            
            P.append(pred.detach().cpu().numpy())
            T.append(target.detach().cpu().numpy())
            L.append(loss.detach().cpu().numpy())
        P = np.vstack(P)
        T = np.vstack(T)
        return P,T,np.mean(L)
    def run(self):
        history = {'train_loss': [], 'val_loss': []} 
        for epoch in tqdm(range(self.config['epochs'])):
            epoch_loss = self.train()
            torch.cuda.empty_cache()  # Clear cache after training
            
            _, _, val_loss = self.test()
            torch.cuda.empty_cache()  # Clear cache after evaluation
            history['train_loss'].append(epoch_loss)  # Update history
            history['val_loss'].append(val_loss)      # Update history
            print('Train Epoch: {}/{} Loss: {:.4f}'.format(
                    epoch, self.config['epochs'], epoch_loss))
            print('Test Epoch: {}/{} Loss: {:.4f}\n'.format(
                epoch, self.config["epochs"], val_loss))

        return history
    
    def save(self, model_path, history_path,history,path, world_size):
        pred, target, test_loss = self.test()
        # convert to tensor
        pred_tensor = torch.tensor(pred).to(self.device)
        target_tensor = torch.tensor(target).to(self.device)
        
        # Gather predictions and targets from all GPUs
        gathered_preds = [torch.zeros_like(pred_tensor) for _ in range(world_size)]
        gathered_targets = [torch.zeros_like(target_tensor) for _ in range(world_size)]

        torch.distributed.all_gather(gathered_preds, pred_tensor)
        torch.distributed.all_gather(gathered_targets, target_tensor)

        # Aggregate test loss
        aggregated_loss = torch.tensor(test_loss).to(self.device)
        torch.distributed.all_reduce(aggregated_loss, op=torch.distributed.ReduceOp.SUM)
        aggregated_loss = aggregated_loss.item() / world_size
        

        if self.rank == 0:
            # Save only the model parameters
            torch.save(self.ddp_model.module.state_dict(), model_path)
            print('saved model!\n')
            # Concatenate the gathered results
            all_preds = torch.cat(gathered_preds, dim=0).cpu().numpy()
            all_targets = torch.cat(gathered_targets, dim=0).cpu().numpy()
            # Save the training history
            with open(history_path, "wb") as pickle_file:
                pickle.dump({
                    "history": history,
                    "predictions": all_preds,
                    "targets": all_targets
                }, pickle_file)
            print('saved history!\n')
            print('Test Epoch: {} Loss: {:.4f}\n'.format(self.config["epochs"], aggregated_loss))

            mean_error, median_error = mean_absolute_percentage_error(all_targets, all_preds)
            print('Mean Relative Error: {:.4f}, Median Relative Error: {:.4f}'.format(mean_error, median_error))

            avg_ssim = calculate_ssim_batch(all_targets, all_preds)
            print('SSIM: {:.4f}'.format(avg_ssim))

            # Plot and save images
            img_plt(all_targets, all_preds, path=path)

    
  
def prepare(dataset,rank, world_size, batch_size, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader
def ddp_train(rank, world_size, train_dataset, test_dataset, config):
    # Initialize any necessary components for DDP
   
    setup(rank, world_size)
    model = ClsSO3Net()
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
   
    train_dataloader = prepare(train_dataset, rank, world_size, config['batch_size'], pin_memory=False, num_workers=2)
    test_dataloader = prepare(test_dataset,rank, world_size, 4, pin_memory=False, num_workers=2)
    # Create the Trainer instance
    trainer = Trainer(ddp_model, train_dataloader, test_dataloader, config, rank, world_size)
    
    # Run the training and testing
    history = trainer.run()
    trainer.save(model_path = '/home/dc-su2/physical_informed/cnn/steerable/model.pth', 
                 history_path = '/home/dc-su2/physical_informed/cnn/steerable/history.json',
                 history = history,
                 path = '/home/dc-su2/physical_informed/cnn/steerable/img/', 
                 world_size = world_size)
   
    # Clean up
    cleanup()
def main():
    
    world_size = torch.cuda.device_count()

    config = parse_args()
    path2 = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/data_augment/rotate1200.hdf5'
    x, y = get_data(path2)
    xtr,xte = x[:1000],x[1000:]
    ytr,yte = y[:1000],y[1000:]

    xtr = torch.Tensor(xtr)
    ytr = torch.Tensor(ytr)
    xte = torch.Tensor(xte)
    yte = torch.Tensor(yte)

    train_dataset = TensorDataset(xtr, ytr)
    test_dataset = TensorDataset(xte, yte)

    ### start training ###
    start = time.time()
    mp.spawn(ddp_train,
             args=(world_size, train_dataset, test_dataset, config),
             nprocs=world_size,
             join=True)
    end = time.time()
    print(f'running time:{(end-start)/60} mins')

if __name__ == '__main__':
   main()