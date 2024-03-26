import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
# custom helper functions
from utils.dataloader     import CustomTransform,IntensityDataset
from utils.grid64Model          import Net
from utils.trainclass     import Train
from utils.loss           import SobelMse,Lossfunction,ResNetFeatures,mean_absolute_percentage_error, calculate_ssim_batch
from utils.plot           import img_plt,history_plt

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
    parser.add_argument('--epochs', type = int, default = 100)
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
    # file paths for train, vali and test
    file_statistics = '/home/dc-su2/physical_informed/cnn/original/clean_statistics.pkl'
    train_file_path = [f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/faceon/new_train_{i}.hdf5' for i in range(4)]
    vali_file_path  = ['/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/faceon/new_vali.hdf5']
    test_file_path  = ['/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/faceon/new_test.hdf5']

    custom_transform = CustomTransform(file_statistics)
    train_dataset= IntensityDataset(train_file_path,transform=custom_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,num_workers=2)

    vali_dataset= IntensityDataset(vali_file_path,transform=custom_transform)
    vali_dataloader = DataLoader(vali_dataset, batch_size=config['batch_size'], shuffle=True,num_workers=2)

    test_dataset= IntensityDataset(test_file_path,transform=custom_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### set a model ###
    model = Net().to(device)
    
    # resnet34 = ResNetFeatures().to(device)
    # loss_object = Lossfunction(resnet34,mse_loss_scale = 0.8,freq_loss_scale=0.2, perceptual_loss_scale=0.0)
    loss_object = SobelMse(device)
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

    # plot and save history
    img_plt(target,pred,path='/home/dc-su2/physical_informed/cnn/original/results/img/')
    history_plt(tr_losses,vl_losses,path='/home/dc-su2/physical_informed/cnn/original/results/')
    with open("/home/dc-su2/physical_informed/cnn/original/results/new_history.pkl", "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    torch.save(model.state_dict(),'/home/dc-su2/physical_informed/cnn/original/results/new_model.pth')


if __name__ == '__main__':
    main()
