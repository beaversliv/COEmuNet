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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--dataset', type = str, default = 'p3droslo')
    parser.add_argument('--model_grid',type=int,default= 64,help='grid of hydro model:[32,64,128]')
    parser.add_argument('--save_path',type =str, default = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/results/')
    parser.add_argument('--logfile',type = str, default = '1gpu_trainlonger_log_file.txt')
    parser.add_argument('--model_name', type = str, default = '1gpu_trainlonger.pth')
    parser.add_argument('--patience',type = int, default = 100, help='early stop patience')
    parser.add_argument('--epochs', type = int, default = 500)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--lr_decay', type = float, default = 0.95)


    args = parser.parse_args()
    
    config = OrderedDict([
            ('path_dir', args.path_dir),
            ('dataset', args.dataset),
            ('model_grid', args.model_grid),
            ('save_path',args.save_path),
            ('logfile',args.logfile),
            ('model_name',args.model_name),
            ('patience',args.patience),
            ('epochs', args.epochs),
            ('batch_size', args.batch_size),
            ('lr', args.lr),
            ('lr_decay', args.lr_decay)
            ])
    
    return config
def main(): 
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
    trainer.save(history_path = config['save_path'] + '1gpu_trainLonger_history.pkl')
   
if __name__ == '__main__':
    main()