import torch
import torch.multiprocessing  as mp
import torch.distributed      as dist
from torch.autograd           import Variable
from torch.utils.data         import DataLoader, TensorDataset,DistributedSampler
from torch.nn.parallel        import DistributedDataParallel as DDP
from torch.profiler           import profile, record_function, ProfilerActivity
from torch.optim.lr_scheduler import StepLR
from utils.preprocessing  import preProcessing
from utils.ResNet3DModel  import Net
from utils.loss           import SobelMse,FreqMae,SobelMae,mean_absolute_percentage_error, calculate_ssim_batch
from utils.trainclass     import ddpTrainer
from utils.config         import parse_args
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

    config = parse_args()
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    data_gen = preProcessing('/home/dc-su2/rds/rds-dirac-dr004/Magritte/random_grid64_data0.hdf5')
    x,y = data_gen.get_data()
    # train test split
    xtr, xte, ytr,yte = train_test_split(x,y,test_size=0.2,random_state=42)
    xtr = torch.tensor(xtr,dtype=torch.float32)
    ytr = torch.tensor(ytr,dtype=torch.float32)
    xte = torch.tensor(xte,dtype=torch.float32)
    yte = torch.tensor(yte,dtype=torch.float32)

    train_dataset = TensorDataset(xtr, ytr)
    test_dataset = TensorDataset(xte, yte)

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    train_dataloader = DataLoader(train_dataset, config['batch_size'],sampler=sampler, pin_memory=True, num_workers=num_workers, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=num_workers, shuffle=False)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)

    model = Net(config['model_grid'])
    model_dic = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/results/best/best_model.pth'
    checkpoint = torch.load(model_dic,map_location=torch.device('cpu'))
    model.encoder_state_dict = {k: v for k, v in checkpoint.items() if k.startswith('encoder')}
    model.decoder_state_dict = {k: v for k, v in checkpoint.items() if k.startswith('decoder')}

    # model = Net(config['model_grid'])
    model = model.to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank],find_unused_parameters=True)

    # Define the optimizer for the DDP model
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    cheduler = StepLR(optimizer, step_size=50, gamma=0.2)
    loss_object = SobelMse(local_rank, alpha=config['alpha'],beta=config['beta'])
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