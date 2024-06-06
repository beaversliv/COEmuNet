import torch
import torch.multiprocessing  as mp
import torch.distributed      as dist
from torch.autograd           import Variable
from torch.utils.data         import DataLoader, TensorDataset,DistributedSampler
from torch.nn.parallel        import DistributedDataParallel as DDP
from torch.profiler           import profile, record_function, ProfilerActivity
from torch.optim.lr_scheduler import StepLR
from utils.preprocessing  import preProcessing
from utils.trainclass     import Trainer
from utils.ResNet3DModel  import Net,FinetuneNet
from utils.loss           import SobelMse,FreqMae,mean_absolute_percentage_error, calculate_ssim_batch
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

def main(): 
    config = parse_args()
    ###set random seed###
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed']) 
    torch.cuda.manual_seed(config['seed'])

    data_gen = preProcessing('/home/dc-su2/rds/rds-dirac-dr004/Magritte/faceon_grid64_data0.hdf5')
    x,y = data_gen.get_data()
    # train test split
    # x,y = np.random.rand(32,3,64,64,64),np.random.rand(32,1,64,64)
    xtr, xte, ytr,yte = train_test_split(x,y,test_size=0.2,random_state=42)
    xtr = torch.tensor(xtr,dtype=torch.float32)
    ytr = torch.tensor(ytr,dtype=torch.float32)
    xte = torch.tensor(xte,dtype=torch.float32)
    yte = torch.tensor(yte,dtype=torch.float32)

    train_dataset = TensorDataset(xtr, ytr)
    test_dataset = TensorDataset(xte, yte)

    ## torch data loader ###
    train_dataloader = DataLoader(train_dataset, batch_size= config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### set a model ###
    model = Net(config['model_grid']).to(device)
    model_dic = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/results/best/best_model.pth'
    model.load_state_dict(torch.load(model_dic,map_location=device))

    # Define the optimizer for the DDP model
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    loss_object = SobelMse(device, alpha=0.8,beta=0.2)
    # Create the Trainer instance
    trainer = Trainer(model, loss_object, optimizer, train_dataloader, test_dataloader, config, device)
    
    # Run the training and testing
    ### start training ###
    start = time.time()
    trainer.run()
    end = time.time()
    print(f'running time:{(end-start)/60} mins')
    trainer.save(True)
   
if __name__ == '__main__':
    main()