import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset,DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity

from utils.preprocessing  import preProcessing
from utils.ResNet3DModel  import Net
from utils.loss           import FreqMSE,mean_absolute_percentage_error, calculate_ssim_batch
# from utils.plot           import img_plt

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
from sklearn.model_selection      import train_test_split 
from socket                       import gethostname
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def setup(rank, world_size):
    "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--model_name', type = str, default = '3dResNet')
    parser.add_argument('--dataset', type = str, default = 'magritte')
    parser.add_argument('--epochs', type = int, default = 10)
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
### train step ###
class Trainer:
    def __init__(self,ddp_model,train_dataloader,test_dataloader,config,rank,world_size):
        self.ddp_model = ddp_model
        self.train_dataloader = train_dataloader
        self.test_dataloader  = test_dataloader

        self.config = config
        self.rank   = rank # global rank, world_size -1, distinguish each process
        self.world_size = world_size # num GPUs
        # Define the optimizer for the DDP model
        self.optimizer = torch.optim.Adam(ddp_model.parameters(), lr=self.config['lr'], betas=(0.9, 0.999))
        self.loss_object = FreqMSE(alpha=0.8,beta=0.2)
                                       
    def train(self):
        total_loss = 0.
        self.ddp_model.train()
        
        for bidx, samples in enumerate(self.train_dataloader):
            data, target = Variable(samples[0]).to(self.rank), Variable(samples[1]).to(self.rank)
            self.optimizer.zero_grad()
            output = self.ddp_model(data)
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
            pred = self.ddp_model(data)
            loss = self.loss_object(target, pred)
            
            P.append(pred.detach().cpu().numpy())
            T.append(target.detach().cpu().numpy())
            L.append(loss.detach().cpu().numpy())
        P = np.vstack(P)
        T = np.vstack(T)
        return P,T,np.mean(L)

    def run(self):
        history = {'train_loss': [], 'val_loss': []} 
        for epoch in tqdm(range(self.config['epochs']), disable=self.rank != 0):  # Disable tqdm progress bar except for rank 0
            epoch_loss = self.train()
            torch.cuda.empty_cache()  # Clear cache after training
            
            _, _, val_loss = self.test()
            torch.cuda.empty_cache()  # Clear cache after evaluation

            # Aggregate losses
            aggregated_epoch_loss = torch.tensor(epoch_loss).to(self.rank)
            torch.distributed.all_reduce(aggregated_epoch_loss, op=torch.distributed.ReduceOp.SUM)
            aggregated_epoch_loss /= self.world_size

            aggregated_val_loss = torch.tensor(val_loss).to(self.rank)
            torch.distributed.all_reduce(aggregated_val_loss, op=torch.distributed.ReduceOp.SUM)
            aggregated_val_loss /= self.world_size

            # Update history on master process
            if self.rank == 0:
                history['train_loss'].append(aggregated_epoch_loss.item())
                history['val_loss'].append(aggregated_val_loss.item())
                print(f'Train Epoch: {epoch}/{self.config["epochs"]} Loss: {aggregated_epoch_loss.item():.4f}')
                print(f'Test Epoch: {epoch}/{self.config["epochs"]} Loss: {aggregated_val_loss.item():.4f}\n')

        return history
    def postProcessing(self,y):
    
        min_ = -50.24472
        median = 11.025192
        y = y*median + min_
        y = np.exp(y)
        return y
    def relativeLoss(self,original_target,original_pred):
        return np.mean( np.abs(original_target-original_pred) / np.max(original_target, axis=1,keepdims=True))
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

            original_target = self.postProcessing(all_targets)
            original_pred = self.postProcessing(all_preds)
            print(f'relative loss {self.relativeLoss(original_target,original_pred)}')

            avg_ssim = calculate_ssim_batch(target,pred)
            for freq in range(len(avg_ssim)):
                print(f'frequency {freq + 1} has ssim {avg_ssim[freq]:.4f}')

            # # Plot and save images
            # img_plt(all_targets, all_preds, path=path)

    
  
def prepare(dataset,rank, world_size, batch_size, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader
def ddp_train(rank, world_size, train_dataset, test_dataset, config,gpus_per_node):
    # Initialize any necessary components for DDP
   
    setup(rank, world_size)
    print(f"Process {rank} out of {world_size} says: {torch.cuda.device_count()} GPUs available\n")
    
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    model = Net()
    model = model.to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank],find_unused_parameters=True)
   
    train_dataloader = prepare(train_dataset, rank, world_size, config['batch_size'], pin_memory=True, num_workers=3)
    test_dataloader = prepare(test_dataset,rank, world_size, config['batch_size'], pin_memory=True, num_workers=3)
    # Create the Trainer instance
    trainer = Trainer(ddp_model, train_dataloader, test_dataloader, config, rank, world_size)
    
    # Run the training and testing
    history = trainer.run()
    trainer.save(model_path = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/rotate/results/sql/test_model.pth', 
                 history_path = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/rotate/results/sql/test_history.pkl',
                 history = history,
                 path = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/rotate/results/sql/', 
                 world_size = world_size)
   
    # Clean up
    cleanup()
def main():
    # Initialize any necessary components for DDP
    world_size    = int(os.environ.get("SLURM_NTASKS"))
    rank          = int(os.environ.get("SLURM_PROCID"))
    gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE"))
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)
   
    config = parse_args()
    data_gen = preProcessing('/home/dc-su2/rds/rds-dirac-dr004/Magritte/random_grid64_data0.hdf5')
    x,y = data_gen.get_data()
    
    # train test split
    # xtr, xte, ytr,yte = train_test_split(x,y,test_size=0.2,random_state=42)
    xtr,xte = x[:800],x[800:1000]
    ytr,yte = y[:800],y[800:1000]
    xtr = torch.tensor(xtr,dtype=torch.float32)
    ytr = torch.tensor(ytr,dtype=torch.float32)
    xte = torch.tensor(xte,dtype=torch.float32)
    yte = torch.tensor(yte,dtype=torch.float32)

    train_dataset = TensorDataset(xtr, ytr)
    test_dataset = TensorDataset(xte, yte)


    ### start training ###
    start = time.time()
    mp.spawn(ddp_train,
             args=(rank, world_size, train_dataset, test_dataset, config,gpus_per_node),
             nprocs=world_size,
             join=True)
    end = time.time()
    print(f'running time:{(end-start)/60} mins')

if __name__ == '__main__':
   main()