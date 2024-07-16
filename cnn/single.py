import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset,random_split
from torch.autograd import Variable
# custom helper functions
from utils.dataloader     import PreProcessingTransform,IntensityDataset
from utils.ResNet3DModel          import Net
from utils.loss           import FreqMse
from utils.plot           import img_plt,history_plt
from utils.config         import parse_args,load_config,merge_config
from utils.trainclass     import Trainer
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
import matplotlib.pyplot as plt
from sklearn.model_selection      import train_test_split 
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
def main():
    args = parse_args()
    config = load_config(args.config)
    config = merge_config(args, config)
    
    np.random.seed(config['model']['seed'])
    torch.manual_seed(config['model']['seed'])
    torch.cuda.manual_seed_all(config['model']['seed'])
    # x,y = np.random.rand(32,3,64,64,64),np.random.rand(32,1,64,64)
    
    # train test split
    transform = PreProcessingTransform(statistics_path=config['dataset']['statistics']['path'],statistics_values=config['dataset']['statistics']['values'],dataset_name=config['dataset']['name'])
    dataset = IntensityDataset(['/home/dc-su2/rds/rds-dirac-dp012/dc-su2/physical_forward/sgl_freq/grid64/Faceon/faceon_grid64_data0.hdf5'],transform=transform)
    print('train test split')
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size= config['dataset']['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### set a model ###
    print('load model')
    model = Net(config['dataset']['grid']).to(device)
    checkpoint = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/results/best/pretrained.pth'
    model.load_state_dict(torch.load(checkpoint,map_location=device))
    
    loss_object = FreqMse(alpha=config['model']['alpha'],beta=config['model']['beta'])
    optimizer_params = config['optimizer']['params']
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    ### start training ###
    print('start training')
    # Assuming model, loss_object, optimizer, train_dataloader, test_dataloader, config, and device are defined
    trainer = Trainer(model, loss_object, optimizer, train_dataloader, test_dataloader, config, device)
    start = time.time()
    trainer.run()
    end = time.time()
    print(f'running time:{(end-start)/60} mins')
    
    ### validation ###
    trainer.save(False)
if __name__ == '__main__':
    main()
