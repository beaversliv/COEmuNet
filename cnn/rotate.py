import torch
import torch.distributed      as dist
from torch.autograd           import Variable
from torch.utils.data         import DataLoader, TensorDataset,DistributedSampler,random_split
from torch.nn.parallel        import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR

from utils.dataloader     import MultiEpochsDataLoader,ChunkLoadingDataset
from utils.loss           import FreqMse
from utils.trainclass     import ddpTrainer
from utils.config         import parse_args,load_config,merge_config
from utils.ResNet3DModel  import Net,Net3D
from utils.utils           import load_encoder_pretrained,load_model_checkpoint,load_optimizer_checkpoint,HistoryShow,Logging

import h5py as h5
import numpy as np
import datetime
import os
import sys
import json
import time
import pickle
import argparse
from collections import OrderedDict
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
    dist.init_process_group("nccl", rank=rank, world_size=world_size,timeout=datetime.timedelta(seconds=2400))

def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()
def main(config): 
    ## Initialize any necessary components for DDP
    world_size    = int(os.environ.get("SLURM_NTASKS"))
    rank          = int(os.environ.get("SLURM_PROCID"))
    gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE",torch.cuda.device_count()))
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    logger        = Logging(config['output']['save_path'], config['output']['log_file'])
    
    logger.info(config)
    logger.info(f'number of gpus: {torch.cuda.device_count()}')
    if gpus_per_node == 0:
        logger.info("No GPUs available on this node.")
    else:
        logger.info(f"GPUs available: {gpus_per_node}")
    logger.info(f'rank {rank} num workers:{num_workers}')
    logger.info(f"Hello from rank {rank} of {world_size} on {socket.gethostname()} where there are {gpus_per_node} allocated GPUs per node.")

    setup(rank, world_size)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    logger.info(f'Rank {rank} of local rank {local_rank}')
    logger.info('finish setup')
    
    np.random.seed(config['model']['seed'])
    torch.manual_seed(config['model']['seed'])
    torch.cuda.manual_seed_all(config['model']['seed'])   
    # Don't del following lines
    # train_size = int(0.8 * len(dataset))
    # test_size = int(0.2 * len(dataset))
    # val_size = len(dataset) - train_size - test_size
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_file_list_path= f"dataset/data/{config['dataset']['name']}/train.txt"
    test_file_list_path = f"dataset/data/{config['dataset']['name']}/test.txt"
    train_dataset = ChunkLoadingDataset(train_file_list_path,config['dataset']['batch_size'])
    test_dataset = ChunkLoadingDataset(test_file_list_path,config['dataset']['batch_size'])
    
    sampler_train = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    train_dataloader = MultiEpochsDataLoader(train_dataset, 1,sampler=sampler_train, pin_memory=True, num_workers=num_workers, shuffle=False)

    sampler_test = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    test_dataloader = MultiEpochsDataLoader(test_dataset, 1, sampler=sampler_test,num_workers=num_workers, shuffle=False)
    torch.cuda.set_device(local_rank)

    if config['dataset']['name'] == 'mulfreq':
        model = Net3D(7).to(local_rank)
        
        if config['model']['stage'] == 1:
            pretrained_encoder_path = config['model'].get('pretrained_encoder', None)
            if pretrained_encoder_path:
                load_encoder_pretrained(model, pretrained_encoder_path, map_location=f'cuda:{local_rank}')
            else:
                logger.info("No pretrained encoder checkpoint provided. Starting from scratch.")
            
            # Start training from scratch with only the encoder loaded
            start_epoch, best_loss = 0, float('inf')
            logger.info(f"Start training Stage 1 (encoder-only) from epoch {start_epoch}")

        elif config['model']['stage'] == 2:
            # Load the full model and optimizer at the second stage (resuming training)
            full_model_checkpoint_path = config['model'].get('resume_checkpoint',None)
            if full_model_checkpoint_path:
                start_epoch, best_loss = load_model_checkpoint(model, full_model_checkpoint_path, map_location=f'cuda:{local_rank}')
            else:
                logger.info("No full model checkpoint provided. Starting from scratch.")
                start_epoch, best_loss = 0, float('inf')  # Start from scratch
                
            logger.info(f"Resuming from checkpoint {full_model_checkpoint_path}. Start Epoch: {start_epoch}, Best Loss: {best_loss}")

        ddp_model = DDP(model, device_ids=[local_rank],find_unused_parameters=True)

        optimizer_params = config['optimizer']['params']
        optimizer = torch.optim.Adam(ddp_model.parameters(), **optimizer_params)
        if config['model']['stage'] == 2 and full_model_checkpoint_path:
            load_optimizer_checkpoint(optimizer, checkpoint_path=full_model_checkpoint_path, map_location=f'cuda:{local_rank}')

    else:
        model = Net(64).to(local_rank)
        checkpoint_path = config['model'].get('resume_checkpoint', None)
        
        if checkpoint_path:
            # Load the model and optimizer states before DDP wrapping
            start_epoch, best_loss = load_model_checkpoint(model, checkpoint_path=checkpoint_path, map_location=f'cuda:{local_rank}')
            logger.info(f"Resuming from checkpoint {checkpoint_path}. Start Epoch: {start_epoch}, Best Loss: {best_loss}")
        else:
            start_epoch, best_loss = 0, float('inf')
            logger.info(f"Starting training from scratch. Start Epoch: {start_epoch}, Best Loss: {best_loss}")

        ddp_model = DDP(model, device_ids=[local_rank],find_unused_parameters=True)

        optimizer_params = config['optimizer']['params']
        optimizer = torch.optim.Adam(ddp_model.parameters(), **optimizer_params)
        if checkpoint_path:
            load_optimizer_checkpoint(optimizer, checkpoint_path=checkpoint_path, map_location=f'cuda:{local_rank}')

    # scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0)
    # scheduler = CyclicLR(optimizer,base_lr=1e-3,max_lr=0.1,step_size_up=100,mode='triangular',cycle_momentum=False)
    scheduler = StepLR(optimizer,step_size=20,gamma=0.1)
    loss_object = FreqMse(alpha=config['model']['alpha'],beta=config['model']['beta'])
    # Create the Trainer instance
    trainer = ddpTrainer(ddp_model, train_dataloader, test_dataloader, optimizer,loss_object,config,rank,local_rank, world_size,logger,scheduler=None)
    
    # Run the training and testing
    ### start training ###
    # trainer.eval()

    start = time.time()
    if rank == 0:
        logger.info(f'Rank {rank} starts training\n')
    trainer.run()
    end = time.time()
    # Synchronize all processes and stop the timer
    dist.barrier()
    if rank == 0:
        logger.info(f'running time: {(end - start) / 3600:.2f} hours')
    # trainer.save(True)

    # if rank == 0:
    #     log_file = config['output']['log_file']
    #     pkl_file = config['output']['pkl_file'] # only saved 20 target and pred
    #     save_dir = config['ouput']['history_img']
    #     img_dir  = config['output']['img_dir']
    #     history_plot = HistoryShow(log_file,pkl_file)
    #     history_plot.history_show(save_dir,single=True)
    #     history_plot.single_impl(img_dir)
    # # Clean up
    cleanup()
if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    config = merge_config(args, config)
    main(config)
