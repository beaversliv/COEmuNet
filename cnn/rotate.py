import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
# custom helper functions
from utils.dataloader     import CustomTransform,IntensityDataset
from utils.ResNet3DModel  import Net3D,Net
from utils.loss           import SobelMse,Lossfunction,mean_absolute_percentage_error, calculate_ssim_batch
from utils.plot           import img_plt,history_plt
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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed)
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

# import logging

# logging.basicConfig(filename='/home/s/ss1421/Documents/physical_informed_surrogate_model/cnn/rotate/training.log', 
#                     level=logging.INFO, 
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S')
# logger = logging.getLogger(__name__)

# Global Constants

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--model_name', type = str, default = '3dResNet')
    parser.add_argument('--dataset', type = str, default = 'p3droslo')
    parser.add_argument('--epochs', type = int, default = 1000)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--lr_decay', type = float, default = 0.95)


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
def main():
    config = parse_args()
    data_gen = preProcessing('/home/dc-su2/rds/rds-dirac-dr004/Magritte/random_grid64_data0.hdf5')
    x,y = data_gen.get_data()
    # train test split
    xtr, xte, ytr,yte = train_test_split(x,y,test_size=0.2,random_state=42)

    xtr = torch.tensor(xtr,dtype=torch.float32)
    ytr = torch.tensor(ytr,dtype=torch.float32)
    xte = torch.tensor(xte,dtype=torch.float32)
    yte = torch.tensor(yte,dtype=torch.float32)

    train_dataset = TensorDataset(xtr, ytr)
    test_dataset = TensorDataset(xte, yte)

    ## torch data loader ###
    train_dataloader = DataLoader(train_dataset, batch_size= config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size= 8, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    model = Net().to(device) 

    loss_object = SobelMse(device,alpha=0.8,beta=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], betas=(0.9, 0.999))

    ### start training ###
    start = time.time()
    # Assuming model, loss_object, optimizer, train_dataloader, test_dataloader, config, and device are defined
    trainer = Trainer(model, loss_object, optimizer, train_dataloader, test_dataloader, config, device)
    tr_losses, vl_losses = trainer.run()
    end = time.time()
    print(f'running time:{(end-start)/60} mins')
    
    ### validation ###
    pred, target, test_loss = trainer.test()
    print('Test Epoch: {} Loss: {:.4f}\n'.format(
                config["epochs"], test_loss))
    data = (tr_losses, vl_losses,pred, target)
    
    mean_error, median_error = mean_absolute_percentage_error(target,pred)
    print('mean relative error: {:.4f}\n, median relative error: {:.4f}'.format(mean_error,median_error))
    avg_ssim = calculate_ssim_batch(target,pred)

    print('SSIM: {:.4f}'.format(avg_ssim))
    
    with open("/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/rotate/results/sql/test_history.pkl", "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    # img_plt(target[:200],pred[:200],path='/home/dc-su2/physical_informed/cnn/rotate/results/img/')
    # history_plt(tr_losses,vl_losses,path='/home/dc-su2/physical_informed/cnn/rotate/results/')
    torch.save(model.state_dict(),'/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/rotate/results/sql/test_model.pth')

if __name__ == '__main__':
    main()

