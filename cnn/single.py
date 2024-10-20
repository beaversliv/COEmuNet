import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset,random_split
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
# custom helper functions
from utils.dataloader     import SequentialDataset,ChunkLoadingDataset
from utils.ResNet3DModel          import Net,Net3D
from utils.loss           import FreqMse
from utils.config         import parse_args,load_config,merge_config
from utils.trainclass     import Trainer
from utils.utils           import load_encoder_pretrained,load_model_checkpoint,HistoryShow,Logging

# helper packages
import h5py as h5
import numpy as np
import os
import sys
import time
import logging
from tqdm                 import tqdm
import pickle
import argparse
from collections import OrderedDict
print('original face on view')
# Check if a GPU is available
if torch.cuda.is_available():
    # Print the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    # Get the name of the current GPU (if there is more than one)
    current_gpu = torch.cuda.current_device()
    print(f"Name of the current GPU: {torch.cuda.get_device_name(current_gpu)}")
else:
    print("No GPU available, using CPU.")
def main(config):
    np.random.seed(config['model']['seed'])
    torch.manual_seed(config['model']['seed'])
    torch.cuda.manual_seed_all(config['model']['seed'])
    logger        = Logging(config['output']['save_path'], config['output']['log_file'])
    
    # train test split
    train_file_list_path= f"/home/dp332/dp332/dc-su2/physical_informed/dataset/data/{config['dataset']['name']}/train.txt"
    test_file_list_path = f"/home/dp332/dp332/dc-su2/physical_informed/dataset/data/{config['dataset']['name']}/test.txt"
    train_dataset = ChunkLoadingDataset(train_file_list_path,config['dataset']['batch_size'])
    test_dataset = ChunkLoadingDataset(test_file_list_path,config['dataset']['batch_size'])
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,num_workers=4,pin_memory=True,prefetch_factor=2)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=4,pin_memory=True,prefetch_factor=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['dataset']['name'] == 'mulfreq':
        model = Net3D(7).to(device)
        optimizer_params = config['optimizer']['params']
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
        if config['model']['stage'] == 1:
            pretrained_encoder_path = config['model'].get('pretrained_encoder', None)
            if pretrained_encoder_path:
                load_encoder_pretrained(model, pretrained_encoder_path, map_location=device)
            else:
                logger.info("No pretrained encoder checkpoint provided. Starting from scratch.")
            
            # Start training from scratch with only the encoder loaded
            start_epoch, best_loss = 0, float('inf')
            logger.info(f"Start training Stage 1 (encoder-only) from epoch {start_epoch}")

        elif config['model']['stage'] == 2:
            # Load the full model and optimizer at the second stage (resuming training)
            full_model_checkpoint_path = config['model'].get('resume_checkpoint',None)
            if full_model_checkpoint_path:
                start_epoch, best_loss = load_model_checkpoint(model, optimizer, full_model_checkpoint_path, map_location=device)
            else:
                logger.info("No full model checkpoint provided. Starting from scratch.")
                start_epoch, best_loss = 0, float('inf')  # Start from scratch
                
            logger.info(f"Start training Stage 2 (full model) from epoch {start_epoch}")
    else:
        model = Net(64).to(device)
        optimizer_params = config['optimizer']['params']
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
        checkpoint_path = config['model']['resume_checkpoint']
        start_epoch, best_loss = load_model_checkpoint(model, optimizer=optimizer, checkpoint_path=checkpoint_path,map_location=device)
        logger.info(f"Resuming from checkpoint. Start Epoch: {start_epoch}, Best Loss: {best_loss}")
    exit(0)
    # Do not del
    #model_dic = '/home/dc-su2/pretrained.pth'
    #checkpoint = torch.load(model_dic,map_location=device)
    # model.encoder_state_dict = {k: v for k, v in checkpoint.items() if k.startswith('encoder')}

    # checkpoint = '/home/dp332/dp332/dc-su2/results/pretrained.pth'
    # model.load_state_dict(torch.load(checkpoint,map_location=device))
    
    loss_object = FreqMse(alpha=config['model']['alpha'],beta=config['model']['beta'])
    # optimizer_params = config['optimizer']['params']
    # optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    ### start training ###
    print('start training')
    scheduler = StepLR(optimizer,step_size=100,gamma=0.1)
    # Assuming model, loss_object, optimizer, train_dataloader, test_dataloader, config, and device are defined
    trainer = Trainer(model, loss_object, optimizer, train_dataloader, test_dataloader, config, logger,device,None)
    start = time.time()
    trainer.run()
    end = time.time()
    print(f'running time:{(end-start)/3600 :4f} h')
    
    ### validation ###
    trainer.save(False)
if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    config = merge_config(args, config)
    main(config)
