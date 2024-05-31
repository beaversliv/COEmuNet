import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
# custom helper functions
from utils.dataloader     import CustomTransform,IntensityDataset
from utils.ResNet3DModel          import Net,FinetuneNet
from utils.loss           import SobelMse,MaxRel, calculate_ssim_batch
from utils.plot           import img_plt,history_plt
from utils.preprocessing  import preProcessing
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

# Global Constants
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--model_name', type = str, default = '3dResNet')
    parser.add_argument('--dataset', type = str, default = 'p3droslo')
    parser.add_argument('--epochs', type = int, default = 10)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--lr_decay', type = float, default = 1e-4)


    args = parser.parse_args()
    
    config = OrderedDict([
            ('path_dir', args.path_dir),
            ('model_name', args.model_name),
            
            ('dataset', args.dataset),
            ('epochs', args.epochs),
            ('batch_size', args.batch_size),
            ('lr', args.lr),
            ('lr_decay', args.lr_decay)
            ])
    
    return config
def postProcessing(y):
    
    min_ = -47.387955
    median = 8.168968
    y = y*median + min_
    y = np.exp(y)
    return y
def main():
    config = parse_args()
    # data_gen = preProcessing('/home/dc-su2/rds/rds-dirac-dr004/Magritte/faceon_grid64_data0.hdf5')
    # x,y = data_gen.get_data()
    # train test split
    x,y = np.random.rand(100,3,64,64,64),np.random.rand(100,1,64,64)
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
    model = FinetuneNet()
    model_dic = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/results/model.pth'
    checkpoint = torch.load(model_dic,map_location=torch.device('cpu'))
    model.encoder_state_dict = {k: v for k, v in checkpoint.items() if k.startswith('encoder')}
    model.decoder_state_dict = {k: v for k, v in checkpoint.items() if k.startswith('decoder')}
    model.to(device)

    loss_object = SobelMse(device,0.8,0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], betas=(0.9, 0.999),weight_decay=config['lr_decay'])


    ### start training ###
    start = time.time()
    # Assuming model, loss_object, optimizer, train_dataloader, test_dataloader, config, and device are defined
    trainer = Trainer(model, loss_object, optimizer, train_dataloader, test_dataloader, config, device)
    tr_losses, vl_losses = trainer.run()
    end = time.time()
    print(f'running time:{(end-start)/60} mins')