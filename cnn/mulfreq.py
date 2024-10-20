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
from utils.utils           import HistoryShow,Logging,load_checkpoint

import h5py as h5
import numpy as np

import os
import time
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
def main(config): 
    # Initialize any necessary components for DDP
    world_size    = int(os.environ.get("SLURM_NTASKS"))
    rank          = int(os.environ.get("SLURM_PROCID"))
    gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE"))
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    logger        = Logging(config['output']['save_path'], config['output']['log_file'])

    logger.info(f"Hello from rank {rank} of {world_size} on {socket.gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)
    setup(rank, world_size)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    logger.info(f'Rank {rank} of local rank {local_rank}')
    
    np.random.seed(config['model']['seed'])
    torch.manual_seed(config['model']['seed'])
    torch.cuda.manual_seed_all(config['model']['seed'])

    train_file_paths = [f'/home/dp332/dp332/dc-su2/dc-su2/mulfreq/train_{i}.hdf5' for i in range(174)]
    test_file_paths = [f'/home/dp332/dp332/dc-su2/dc-su2/mulfreq/test_{i}.hdf5' for i in range(22)]
    
    train_dataset = ChunkLoadingDataset(train_file_paths,config['dataset']['batch_size'],config['dataset']['name'])
    test_dataset = ChunkLoadingDataset(test_file_paths,config['dataset']['batch_size'],config['dataset']['name'])
    
    sampler_train = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    train_dataloader = MultiEpochsDataLoader(train_dataset, 1,sampler=sampler_train, pin_memory=True, num_workers=num_workers, shuffle=False)

    sampler_test = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    test_dataloader = MultiEpochsDataLoader(test_dataset, 1, sampler=sampler_test,num_workers=num_workers, shuffle=False)
    torch.cuda.set_device(local_rank)

    model = Net3D(freq=7).to(local_rank)
    
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    model_dic = '/home/dp332/dp332/dc-su2/results/pretrained.pth'
    checkpoint = torch.load(model_dic,map_location=map_location)
    model.encoder_state_dict = {k: v for k, v in checkpoint.items() if k.startswith('encoder')}
    # checkpoint_path = config['model']['checkpoint_path']
    # start_epoch, best_loss = load_checkpoint(model, optimizer=optimizer, checkpoint_path=checkpoint_path,local_rank=local_rank)
    # logger.info(f"Resuming from checkpoint. Start Epoch: {start_epoch}, Best Loss: {best_loss}")
    ddp_model = DDP(model, device_ids=[local_rank],find_unused_parameters=True)

    # Define the optimizer for the DDP model
    optimizer_params = config['optimizer']['params']
    optimizer = torch.optim.Adam(ddp_model.parameters(), **optimizer_params)
    scheduler = StepLR(optimizer,step_size=100,gamma=0.1)
    loss_object = FreqMse(alpha=config['model']['alpha'],beta=config['model']['beta'])
    # Create the Trainer instance
    trainer = ddpTrainer(ddp_model, train_dataloader, test_dataloader, optimizer,loss_object,config,rank,local_rank, world_size,logger,scheduler=None)
    
    # Run the training and testing
    dist.barrier()
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
    #     log_file = config['output']['logfile']
    #     pkl_file = config['output']['results']
    #     save_path = config['ouput']['history_img']
    #     history_plot = HistoryShow(log_file,pkl_file)
    #     history_plot.history_show(save_path)
    #     history_plot.mulfreq_impl(img_dir='/home/dp332/dp332/dc-su2/results/mulfreq/img/')
    
    # Clean up
    cleanup()
    print(f'Rank {rank} clean up process')
if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    config = merge_config(args, config)
    main(config)