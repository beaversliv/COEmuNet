import torch
import torch.nn as nn
import torch.multiprocessing  as mp
import torch.distributed      as dist
from torch.autograd           import Variable
from torch.utils.data         import DataLoader, TensorDataset,DistributedSampler
from torch.nn.parallel        import DistributedDataParallel as DDP
from torch.profiler           import profile, record_function, ProfilerActivity
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR,CyclicLR
from utils.preprocessing  import preProcessing,get_data

from utils.loss           import SobelMse,FreqMae,SobelMae,mean_absolute_percentage_error, calculate_ssim_batch
from utils.trainclass     import ddpTrainer
from utils.config         import parse_args,load_config,merge_config
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
import socket    
### Resnet ###
class Conv_BN_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv_BN_Relu,self).__init__()
        self.layers =  nn.ModuleList()
        self.layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride = stride, padding = padding))
        self.layers.append(nn.BatchNorm3d(out_channels))
        self.layers.append(nn.ReLU())
    
    def forward(self,x):   
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return  x

class residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, skip=False):
        super(residual_Block,self).__init__()
        self.skip = skip
        
        self.CBR0 =  Conv_BN_Relu(in_channels, out_channels, kernel_size, stride, padding)
        self.CBR1 =  Conv_BN_Relu(out_channels, out_channels, kernel_size, stride = 1, padding = 'same')
        self.short_CBR0 = Conv_BN_Relu(out_channels, out_channels, kernel_size = (1, 1, 1), stride = 1, padding = 0)
        
    def forward(self, x):  
        residual = x
        x = self.CBR0(x)
        x = self.CBR1(x)
        
        if self.skip :
            # convolutional residual block
            shortcut = self.short_CBR0(x)
            y = x + shortcut
            return y
        else:
          # identical residual block
            y = x + residual
            return y    
        
class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder,self).__init__()
        self.layers =  nn.ModuleList()
        self.layers.append(Conv_BN_Relu(in_channels, 4, kernel_size=3, stride=1, padding = 0)) # [batch, 4, 62, 62, 62]
        self.layers.append(nn.MaxPool3d(2))                                                    # [batch, 4, 31, 31, 31]
        
        self.layers.append(residual_Block(4, 8, kernel_size=3, stride=2, padding = 1, skip=True)) # [batch, 8, 16, 16, 16]
        self.layers.append(residual_Block(8, 8, kernel_size=3, stride=1, padding = 'same', skip=False)) # [batch, 8, 16, 16, 16]
        self.layers.append(residual_Block(8, 8, kernel_size=3, stride=1, padding = 'same', skip=False)) # [batch, 8, 16, 16, 16]
        # self.layers.append(residual_Block(8, 8, kernel_size=3, stride=1, padding = 'same', skip=False))
        
        self.layers.append(residual_Block(8, 16, kernel_size=3, stride=2, padding = 1, skip=True))        # [batch, 16, 8,8,8]
        self.layers.append(residual_Block(16, 16, kernel_size=3, stride=1, padding = 'same', skip=False)) # [batch, 16, 8,8,8]
        self.layers.append(residual_Block(16, 16, kernel_size=3, stride=1, padding = 'same', skip=False)) # [batch, 16, 8,8,8]
        # self.layers.append(residual_Block(16, 16, kernel_size=3, stride=1, padding = 'same', skip=False))        

        self.layers.append(residual_Block(16, 32, kernel_size=3, stride=2, padding = 1, skip=True)) # [batch, 32, 4,4,4]
        self.layers.append(residual_Block(32, 32, kernel_size=3, stride=1, padding = 'same', skip=False)) # [batch, 32, 4,4,4]
        self.layers.append(residual_Block(32, 32, kernel_size=3, stride=1, padding = 'same', skip=False)) # [batch, 32, 4,4,4]
    def forward(self, x):   
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return  x
class Decoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()

        # Example: Halving the channels and doubling the spatial dimension with each step
        for i in range(3):
            self.layers.append(nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=3, stride=1,padding=1))
            self.layers.append(nn.BatchNorm2d(int(in_channels / 2)))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))  # Upsampling
            in_channels = int(in_channels/2)

        # Final convolution to get the desired number of output channels (1 in this case)
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Latent(nn.Module):
    def __init__(self,input_dim,model_grid=64):
        super(Latent,self).__init__()
        self.layers =  nn.ModuleList()
        if model_grid == 32:
            out_dim = 64 * 4 * 4
        elif model_grid == 64:
            out_dim = 64 * 8 * 8
        elif model_grid == 128:
            out_dim = 64 * 16 * 16

        self.layers.append(nn.Linear(input_dim, 16**3))
        self.layers.append(nn.Linear(16**3, out_dim))
        self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class FinetuneNet(nn.Module):
    def __init__(self,model_grid=64):
        super(FinetuneNet, self).__init__()
        self.model_grid = model_grid
        self.encoder0 = Encoder(1)
        self.encoder1 = Encoder(1)
        self.encoder2 = Encoder(1)
        # grid 64
        if model_grid == 32:
            input_dim = 32*2*2*2*3
        elif model_grid == 64:
            input_dim = 32*4*4*4*3
        elif model_grid == 128:
            input_dim = 32*8*8*8*3
        self.latent = Latent(input_dim=input_dim)

        self.decoder= Decoder(in_channels=64, out_channels=1)
    def forward(self,x):
        x0 = self.encoder0(x[:, 0:1, :, :, :])
        x1 = self.encoder1(x[:, 1:2, :, :, :])
        x2 = self.encoder2(x[:, 2:3, :, :, :])
      
        # x0 shape (batch size, 32*4*4*4)
        x0 = torch.flatten(x0, start_dim=1)   
        x1 = torch.flatten(x1, start_dim=1)   
        x2 = torch.flatten(x2, start_dim=1) 
        # x shape (batch size, 32*4*4*4*3)
        features = torch.cat([x0, x1, x2], dim = -1)
        latent_output = self.latent(features)

        if self.model_grid == 32:
            x = latent_output.view(-1, 64, 4, 4)
        elif self.model_grid == 64:
            x = latent_output.view(-1, 64, 8, 8)
        elif self.model_grid == 128:
            x = latent_output.view(-1, 64, 16, 16)
        
	    # shape (batch_size,64,8,8)
        output = self.decoder(x)
        return output
def setup(rank, world_size):
    "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"

    def get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    # Ensure MASTER_ADDR is set
    if "MASTER_ADDR" not in os.environ:
        raise ValueError("MASTER_ADDR environment variable is not set")
    
    # Set MASTER_PORT to a free port
    if "MASTER_PORT" not in os.environ:
        free_port = get_free_port()
        os.environ["MASTER_PORT"] = str(free_port)
    
    # Ensure MASTER_PORT is set
    if "MASTER_PORT" not in os.environ or os.environ["MASTER_PORT"] is None:
        raise ValueError("MASTER_PORT environment variable is not set")

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()
def main(): 
    # Initialize any necessary components for DDP
    world_size    = int(os.environ.get("SLURM_NTASKS"))
    rank          = int(os.environ.get("SLURM_PROCID"))
    gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE"))
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"Hello from rank {rank} of {world_size} on {socket.gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)
    setup(rank, world_size)

    args = parse_args()
    config = load_config(args.config)
    config = merge_config(args, config)
    
    np.random.seed(config['model']['seed'])
    torch.manual_seed(config['model']['seed'])
    torch.cuda.manual_seed_all(config['model']['seed'])

    x,y = get_data('/home/dc-su2/rds/rds-dirac-dr004/Magritte/minmax_random_grid64_data0.hdf5')
    # x,y = np.random.rand(32,3,64,64,64), np.random.rand(32,1,64,64)
    # train test split
    xtr, xte, ytr,yte = train_test_split(x,y,test_size=0.2,random_state=42)
    xtr = torch.tensor(xtr,dtype=torch.float32)
    ytr = torch.tensor(ytr,dtype=torch.float32)
    xte = torch.tensor(xte,dtype=torch.float32)
    yte = torch.tensor(yte,dtype=torch.float32)

    train_dataset = TensorDataset(xtr, ytr)
    test_dataset = TensorDataset(xte, yte)

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    train_dataloader = DataLoader(train_dataset, config['dataset']['batch_size'],sampler=sampler, pin_memory=True, num_workers=num_workers, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], num_workers=num_workers, shuffle=False)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)

    model = FinetuneNet(config['dataset']['grid'])
    # model_dic = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/results/best/best_model.pth'
    # checkpoint = torch.load(model_dic,map_location=torch.device('cpu'))
    # model.encoder_state_dict = {k: v for k, v in checkpoint.items() if k.startswith('encoder')}
    # model.decoder_state_dict = {k: v for k, v in checkpoint.items() if k.startswith('decoder')}

    # model = Net(config['model_grid'])
    model = model.to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank],find_unused_parameters=True)

    # Define the optimizer for the DDP model
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=4e-4, betas=(0.9, 0.999))
    # scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0)
    scheduler = CyclicLR(optimizer,base_lr=4*1e-4,max_lr=4*1e-2,step_size_up=100,mode='triangular',cycle_momentum=False)
    loss_object = SobelMse(local_rank, alpha=config['model']['alpha'],beta=config['model']['beta'])
    # Create the Trainer instance
    trainer = ddpTrainer(ddp_model, train_dataloader, test_dataloader, optimizer,loss_object,config,local_rank, world_size,scheduler=None)
    
    # Run the training and testing
    ### start training ###
    trainer.eval()

    start = time.time()
    print(f'rank {rank} starts training\n')
    trainer.run()
    end = time.time()
    print(f'running time:{(end-start)/60} mins')
    trainer.save(False)
   
    # Clean up
    cleanup()
if __name__ == '__main__':
    main()