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
from utils.utils           import LoadCheckPoint,Logging,setup,cleanup

import numpy as np
import os

import time
import socket    
import yaml
def main(config): 
    ## Initialize any necessary components for DDP
    world_size    = int(os.environ.get("SLURM_NTASKS"))
    rank          = int(os.environ.get("SLURM_PROCID"))
    gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE",torch.cuda.device_count()))
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    logger        = Logging(config['output']['save_path'], config['output']['log_file'])

    setup(rank, world_size)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)

    if rank == 0:
        logger.info(yaml.dump(config, default_flow_style=False, sort_keys=False))

        if gpus_per_node == 0:
            logger.info("No GPUs available on this node.")
        else:
            logger.info(f"GPUs available: {gpus_per_node}")

    logger.info(f"Hello from rank {rank} of {world_size} on {socket.gethostname()} where there are {gpus_per_node} allocated GPUs and {num_workers} numb workers per node.")
    logger.info(f'Rank {rank} of local rank {local_rank} finishes setup')
    
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
        model = Net(config['dataset']['grid'],in_channels=64,out_channels=7).to(local_rank)
    else:
        model = Net(config['dataset']['grid'],in_channels=64,out_channels=1).to(local_rank)

    if local_rank == 0:
        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params}")
        logger.info(f"Trainable parameters: {trainable_params}")

    
    file_path = config['model'].get('resume_checkpoint',None)
    # load checkpoint that is unwrapped with ddp, so no module.
    checkpoint_loading = LoadCheckPoint(
        learning_model=model,
        optimizer=None,
        file_path=file_path,
        stage=config['model']['stage'],
        logger=logger,
        local_rank=f'cuda:{local_rank}',
        ddp_on=False
    )
    checkpoint_loading.load_checkpoint(model_only=True)
    # Wrap the model with DDP
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer_params = config['optimizer']['params']
    optimizer = torch.optim.Adam(ddp_model.parameters(), **optimizer_params)

    checkpoint_loading.optimizer = optimizer
    checkpoint_loading.load_checkpoint(model_only=False)
        
    # scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0)
    # scheduler = CyclicLR(optimizer,base_lr=1e-3,max_lr=0.1,step_size_up=100,mode='triangular',cycle_momentum=False)
    scheduler = StepLR(optimizer,step_size=20,gamma=0.1)
    loss_object = FreqMse(alpha=config['model']['alpha'])
    # Create the Trainer instance
    trainer = ddpTrainer(ddp_model, train_dataloader, test_dataloader, optimizer,loss_object,config,rank,local_rank, world_size,logger,scheduler=None)

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
