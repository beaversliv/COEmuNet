import torch
import torch.distributed      as dist
from torch.autograd           import Variable
from torch.utils.data         import DistributedSampler,random_split
from torch.nn.parallel        import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR

from utils.dataloader     import MultiEpochsDataLoader,ChunkLoadingDataset
from utils.loss           import FreqMse
from utils.trainclass     import Trainer
from utils.config         import str_to_bool,load_config,merge_config
from utils.ResNet3DModel  import Net,Net3D
from utils.utils           import LoadCheckPoint,Logging,setup,cleanup

# helper packages
import h5py as h5
import numpy as np
import os
import time
import socket    
import yaml
import argparse
def main(config): 
    logger        = Logging(config['output']['save_dir'], config['output']['logfile_name'])
    if config['ddp_on']:
        ## Initialize DDP components
        world_size    = int(os.environ.get("SLURM_NTASKS"))
        rank          = int(os.environ.get("SLURM_PROCID"))
        gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE",torch.cuda.device_count()))  # GPUs per node

        setup(rank, world_size)
        local_rank = int(os.environ['SLURM_LOCALID'])

        device = torch.device("cuda:0")         # Always 0 now
        torch.cuda.set_device(device)
        print(f"[Rank {rank}] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"[Rank {rank}] Model will use device: {torch.cuda.current_device()} - {torch.cuda.get_device_name()}")
    else:
        world_size = 1
        rank = 0
        local_rank = 0
        gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", 0))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Initialize any necessary components for DDP
    world_size    = int(os.environ.get("SLURM_NTASKS"))
    rank          = int(os.environ.get("SLURM_PROCID"))
    gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE",torch.cuda.device_count()))
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    logger        = Logging(config['output']['save_path'], config['output']['log_file'])

    setup(rank, world_size)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)

    # num_workers is shared across gpu and multi-gpu mode
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    logger.info(f'device name:{device}',rank,True)
    # write configuration out
    if rank == 0:
        logger.info(yaml.dump(config, default_flow_style=False, sort_keys=False))
    logger.info(f"There are {world_size} processes on {socket.gethostname()} where there are {gpus_per_node} allocated GPUs and {num_workers} numb workers per node.",rank,True)
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
    
    if config['ddp_on']:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        shuffle_train = False  # Sampler handles shuffling
        shuffle_test  = False  # No need to shuffle test data
    else:
        train_sampler = None
        test_sampler  = None
        shuffle_train = True   # Enable shuffling for training
        shuffle_test  = False  # No shuffling for test data
    # Training DataLoader
    train_dataloader = MultiEpochsDataLoader(
        train_dataset,
        batch_size = 1 if config['dataset']['name'] == 'mulfreq' else config['model']['batch_size'],
        sampler    = train_sampler,
        shuffle    = shuffle_train,
        num_workers= num_workers,
        pin_memory = True
    )

    # Testing DataLoader
    test_dataloader = MultiEpochsDataLoader(
        test_dataset,
        batch_size  = 1 if config['dataset']['name'] == 'mulfreq' else config['model']['batch_size'],
        sampler     = test_sampler,
        shuffle     = shuffle_test,
        num_workers = num_workers,
        pin_memory  = True
    )
    # init model and optimizer
    if config['dataset']['name'] == 'mulfreq':
        model = Net3D(7,config['dataset']['grid']).to(local_rank)
    else:
        model = Net(config['dataset']['grid'],in_channels=64,out_channels=1).to(local_rank)

    optimizer_params = config['optimizer']['params']
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)

    # Initialize Scheduler (if enabled)
    scheduler = None
    scheduler_config = config.get('use_scheduler', False)  
    if scheduler_config:
        selected_type = config.get('scheduler_type', None)  # Use `args.scheduler_type` if dynamic selection
        if selected_type is None:
            raise ValueError("Scheduler type must be specified in 'scheduler_type' or passed as an argument.")
        scheduler_type = scheduler_config['type']
        scheduler_params = scheduler_config['params']
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)

    # print model arams
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params}")
        logger.info(f"Trainable parameters: {trainable_params}")
    
    # Load checkpoint
    if config.get('use_checkpoint', False):
        checkpoint_loading = LoadCheckPoint(
            learning_model=model,
            optimizer=optimizer,
            scheduler=None, # Will initialize below
            file_path=config['resume_checkpoint'],
            stage=config['model']['stage'],
            logger=logger,
            local_rank=device,
            ddp_on=config.get('ddp_on', True) # whether remove 'module.' in checkpoint
        )
        checkpoint_loading.load_checkpoint(model_only=False)  # Load Model, Optimizer, Scheduler

        ## Use checkpointed values
        checkpoint_loading.model = model
        checkpoint_loading.optimizer = optimizer
        checkpoint_loading.scheduler = scheduler
    # move model to device 
    model.to(device)
    if config.get('ddp_on',False):
        model = DDP(model, device_ids=[0], find_unused_parameters=False)

    loss_object = FreqMse(alpha=config['model']['alpha'])
    # Create the Trainer instance
    trainer = Trainer(model, train_dataloader, test_dataloader, optimizer,loss_object,config,rank,local_rank, world_size,logger,scheduler=scheduler)

    start = time.time()
    logger.info(f'Rank {rank} starts training\n')
    trainer.run()
    end = time.time()
    # Synchronize all processes and stop the timer
    if config.get('ddp_on',False):
        dist.barrier()
    if rank == 0:
        logger.info(f'running time: {(end - start) / 3600:.2f} hours')

    # Clean up
    if config['ddp_on']:
        cleanup()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/mulfreq_dataset.yaml', help='Path to the YAML configuration file')
    parser.add_argument('--grid',type=int,choices=[32,64,128],help='grid of hydro model:[32,64,128]')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--seed', type=int, help='Override seed')

    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--alpha', type=float, help='weight for MSE ,then (1-alpha) is the weight for frequency loss')
    parser.add_argument('--stage',type=int,choices=[1,2],help='stage 1: start from pretrained faceon model; stage 2: load whole model')
    parser.add_argument('--use_checkpoint',type=str_to_bool, default=False,help="Enable or disable the loading checkpoint")
    parser.add_argument('--resume_checkpoint',type=str,help='checkpoint path')

    parser.add_argument("--use_scheduler", type=str_to_bool, default=None, help="Enable or disable the scheduler")
    parser.add_argument("--scheduler_type", type=str, choices=["StepLR","CosineAnnealingLR"], default="StepLR", help="Override scheduler type (e.g., stepLR, cosineAnnealingLR)")
    parser.add_argument("--step_size", type=int, default=50, help="Step size for StepLR")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma value for StepLR")
    parser.add_argument("--T_max", type=int, default=20, help="Maximum number of iterations for CosineAnnealingLR")

    parser.add_argument("--ddp_on",type=str_to_bool,default=False, help="whether use distributed learning")
    
    parser.add_argument('--save_path', type=str,help='path for history.png, history.pkl and model.pth')
    parser.add_argument('--log_file', type=str,help='training history')
    parser.add_argument('--model_file', type=str,help='saved model name')
    parser.add_argument('--pkl_file', type=str,help='saved target and pred in pkl')
    parser.add_argument('--history_img', type=str,help='train test value vs epoch, history.png')
    parser.add_argument('--img_dir', type=str,help='save target img vs pred img in which dir')
    
    args = parser.parse_args()
    config = load_config(args.config)
    config = merge_config(args, config)
    main(config)
