import torch
import torch.multiprocessing  as mp
import torch.distributed      as dist
from torch.autograd           import Variable
from torch.utils.data         import DataLoader, TensorDataset,DistributedSampler
from torch.nn.parallel        import DistributedDataParallel as DDP
from torch.profiler           import profile, record_function, ProfilerActivity
from torch.optim.lr_scheduler import StepLR
from utils.preprocessing  import preProcessing
from utils.trainclass     import ddpTrainer
from utils.ResNet3DModel  import Net
from utils.loss           import SobelMse,FreqMae,mean_absolute_percentage_error, calculate_ssim_batch
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

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed)
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
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--model_name', type = str, default = '3dResNet')
    parser.add_argument('--dataset', type = str, default = 'p3droslo')
    parser.add_argument('--save_path',type =str, default = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/results/')
    parser.add_argument('--logfile',type = str, default = 'trainLonger_log_file')
    parser.add_argument('--epochs', type = int, default = 3000)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--lr_decay', type = float, default = 1e-4)


    args = parser.parse_args()
    
    config = OrderedDict([
            ('path_dir', args.path_dir),
            ('model_name', args.model_name),
            ('dataset', args.dataset),
            ('save_path',args.save_path),
            ('logfile',args.logfile),
            ('epochs', args.epochs),
            ('batch_size', args.batch_size),
            ('lr', args.lr),
            ('lr_decay', args.lr_decay)
            ])
    
    return config
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
    data_gen = preProcessing('/home/dc-su2/rds/rds-dirac-dr004/Magritte/faceon_grid64_data0.hdf5')
    x,y = data_gen.get_data()
    # train test split
    # x,y = np.random.rand(100,3,64,64,64),np.random.rand(100,1,64,64)
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
    ### set a model ###
    model = Net(64)
    model = model.to(local_rank)
    model_dict = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/results/model.pth'
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    model.load_state_dict(torch.load(model_dict, map_location=map_location))
    ddp_model = DDP(model, device_ids=[local_rank],find_unused_parameters=True)

    # Define the optimizer for the DDP model
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    loss_object = SobelMse(local_rank, alpha=0.8,beta=0.2)
    # Create the Trainer instance
    trainer = ddpTrainer(ddp_model, train_dataloader, test_dataloader, optimizer,loss_object,config,local_rank, world_size)
    
    # Run the training and testing
    ### start training ###
    start = time.time()
    print(f'rank {rank} starts training')
    history = trainer.run()
    end = time.time()
    print(f'running time:{(end-start)/60} mins')
    trainer.save(model_path = config['save_path'] + 'trainLonger_model.pth', 
                 history_path = config['save_path'] + 'trainLonger_history.pkl',
                 history = history,
                 world_size = world_size)
   
    # Clean up
    cleanup()
if __name__ == '__main__':
    main()