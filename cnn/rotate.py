import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from utils.so3_model      import SO3Net
import torch
import torch.nn as nn
import torch.multiprocessing  as mp
import torch.distributed      as dist
from torch.autograd           import Variable
from torch.utils.data         import DataLoader, TensorDataset,DistributedSampler,random_split

from torch.nn.parallel        import DistributedDataParallel as DDP
from torch.profiler           import profile, record_function, ProfilerActivity
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR,CyclicLR

from utils.dataloader     import PreProcessingTransform,IntensityDataset,AddGaussianNoise1,CustomCompose
from utils.loss           import SobelMse,FreqMae,SobelMae,mean_absolute_percentage_error, calculate_ssim_batch
from utils.trainclass     import ddpTrainer
from utils.config         import parse_args,load_config,merge_config
from utils.ResNet3DModel  import Net,FinetuneNet
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

    args = parse_args()
    config = load_config(args.config)
    config = merge_config(args, config)
    
    np.random.seed(config['model']['seed'])
    torch.manual_seed(config['model']['seed'])
    torch.cuda.manual_seed_all(config['model']['seed'])

    transform = PreProcessingTransform(config['dataset']['statistics']['path'],config['dataset']['statistics']['values'])
    train_file_paths = [f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/sgl_freq/grid64/Rotation/train_{i}.hdf5' for i in range(40)]
    test_file_paths = [f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/sgl_freq/grid64/Rotation/test_{i}.hdf5' for i in range(10)]
    # train test split
    train_dataset = IntensityDataset(train_file_paths,transform=transform)
    test_dataset = IntensityDataset(test_file_paths,transform=transform)
    # Don't del following lines
    # train_size = int(0.8 * len(dataset))
    # test_size = int(0.2 * len(dataset))
    # val_size = len(dataset) - train_size - test_size
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Apply the noise transformation only to the training dataset
    # train_dataset.dataset.transform = noise_transform
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    train_dataloader = DataLoader(train_dataset, config['dataset']['batch_size'],sampler=sampler, pin_memory=True, num_workers=num_workers, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], num_workers=num_workers, shuffle=False)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)

    model = Net(64)
    model = model.to(local_rank)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    model_dic = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/results/best/pretrained.pth'
    model.load_state_dict(torch.load(model_dic, map_location=map_location))
    # checkpoint = torch.load(model_dic,map_location=torch.device('cpu'))
    # model.encoder_state_dict = {k: v for k, v in checkpoint.items() if k.startswith('encoder')}
    # model.decoder_state_dict = {k: v for k, v in checkpoint.items() if k.startswith('decoder')}
    ddp_model = DDP(model, device_ids=[local_rank],find_unused_parameters=True)

    # Define the optimizer for the DDP model
    optimizer_params = config['optimizer']['params']
    optimizer = torch.optim.Adam(ddp_model.parameters(), **optimizer_params)
    # scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0)
    # scheduler = CyclicLR(optimizer,base_lr=1e-3,max_lr=0.1,step_size_up=100,mode='triangular',cycle_momentum=False)
    # scheduler = StepLR(optimizer,step_size=10,gamma=0.1)
    loss_object = FreqMae(alpha=config['model']['alpha'],beta=config['model']['beta'])
    # Create the Trainer instance
    trainer = ddpTrainer(ddp_model, train_dataloader, test_dataloader, optimizer,loss_object,config,rank,local_rank, world_size,scheduler=None)
    
    # Run the training and testing
    ### start training ###
    # trainer.eval()

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