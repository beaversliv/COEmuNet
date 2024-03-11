import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
# custom helper functions
from utils.dataloader     import CustomTransform,IntensityDataset
from utils.model          import Net
from utils.loss           import SobelRelative,Lossfunction,ResNetFeatures,mean_absolute_percentage_error, calculate_ssim_batch
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
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 32)
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

# file paths for train, vali and test
### torch data loader ###
# file_statistics = '/home/s/ss1421/Documents/physical_informed_surrogate_model/cnn/rotate/rotate24000_statistics.pkl'
# file_paths = [f'/data/astro1/ss1421/physical_forward/cnn/Batches/rotate24000_{i}.hdf5' for i in range(10)]
# train_file_path = file_paths[:8]
# vali_file_path  = file_paths[2:]

# custom_transform = CustomTransform(file_statistics)
# train_dataset= IntensityDataset(train_file_path,transform=custom_transform)
# train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,num_workers=2)

# vali_dataset= IntensityDataset(vali_file_path,transform=custom_transform)
# vali_dataloader = DataLoader(vali_dataset, batch_size=config['batch_size'], shuffle=True,num_workers=2)

### train step ###
class Trainer:
    def __init__(self,model,loss_object,optimizer,train_dataloader,test_dataloader,config,device):
        self.model = model
        self.loss_object = loss_object
        self.optimizer   = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader  = test_dataloader
        self.config = config
        self.device = device
    
    def train(self):
        total_loss = 0.
        edge_loss  = 0.
        relative1  = 0.
        relative2  = 0.
        self.model.train()
         
        for bidx, samples in enumerate(self.train_dataloader):
            data, target = Variable(samples[0]).to(self.device), Variable(samples[1]).to(self.device)
            self.optimizer.zero_grad()
            latent,output = self.model(data)
            loss_edge,relative_mean,relativeMean,loss_combined = self.loss_object(target, output)
            loss_combined.backward()
            self.optimizer.step()
            total_loss += loss_combined.detach().cpu().numpy()

            edge_loss += loss_edge.detach().cpu().numpy()
            relative1 += relative_mean.detach().cpu().numpy()
            relative2 += relativeMean.detach().cpu().numpy()
        epoch_loss = total_loss / len(self.train_dataloader)  # divide number of batches
        edge_epoch_loss = edge_loss / len(self.train_dataloader)
        relative1_epoch_loss = relative1 / len(self.train_dataloader)
        relative2_epoch_loss = relative2 / len(self.train_dataloader)
        return epoch_loss,edge_epoch_loss,relative1_epoch_loss,relative2_epoch_loss

    
    def test(self):
        self.model.eval()
        P, T, L = [], [], torch.tensor(0.0).to(self.device)
        each_loss = {'edge_loss': torch.tensor(0.0).to(self.device),
                    'relative1': torch.tensor(0.0).to(self.device),
                    'relative2': torch.tensor(0.0).to(self.device)}
        num_batches = 0

        for bidx, samples in enumerate(self.test_dataloader):
            data, target = samples[0].to(self.device), samples[1].to(self.device)
            with torch.no_grad():
                latent, pred = self.model(data)
                loss_edge, relative_mean, relativeMean, loss_combined = self.loss_object(target, pred)

            # Directly accumulate to tensors
            L += loss_combined
            each_loss['edge_loss'] += loss_edge
            each_loss['relative1'] += relative_mean
            each_loss['relative2'] += relativeMean
            num_batches += 1

            # For predictions and targets, consider batch aggregating if feasible
            P.append(pred.detach().cpu().numpy())
            T.append(target.detach().cpu().numpy())

        # Compute means
        L /= num_batches
        each_loss = {k: v / num_batches for k, v in each_loss.items()}

        P = np.vstack(P)
        T = np.vstack(T)
        return P, T, L.cpu().numpy(), each_loss['edge_loss'].cpu().numpy(), each_loss['relative1'].cpu().numpy(), each_loss['relative2'].cpu().numpy()
    def run(self):
    
        loss_history = {
        'train': {'total_loss': [], 'edge_loss': [], 'relative1_loss': [], 'relative2_loss': []},
        'val': {'total_loss': [], 'edge_loss': [], 'relative1_loss': [], 'relative2_loss': []}
        }
        for epoch in tqdm(range(self.config['epochs'])):
            tr_loss,edge_loss_tr,relative1_loss_tr,relative2_loss_tr = self.train()
            torch.cuda.empty_cache()  # Clear cache after training
            
            _, _, val_loss, edge_loss_vl,relative1_loss_vl,relative2_loss_vl = self.test()
            torch.cuda.empty_cache()  # Clear cache after evaluation
            loss_history['train']['total_loss'].append(tr_loss)
            loss_history['train']['edge_loss'].append(edge_loss_tr)
            loss_history['train']['relative1_loss'].append(relative1_loss_tr)
            loss_history['train']['relative2_loss'].append(relative2_loss_tr)

            loss_history['val']['total_loss'].append(val_loss)
            loss_history['val']['edge_loss'].append(edge_loss_vl)
            loss_history['val']['relative1_loss'].append(relative1_loss_vl)
            loss_history['val']['relative2_loss'].append(relative2_loss_vl)

            print(f'Train Epoch: {epoch}/{self.config["epochs"]} Loss: {tr_loss:.4f}')
            print(f'Test Epoch: {epoch}/{self.config["epochs"]} Loss: {val_loss:.4f}\n')
            
        return loss_history


def main():
    config = parse_args()
    # read data
    path = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/data_augment/clean_rotate1200.hdf5'
    with h5.File(path,'r') as file:
        x = np.array(file['input'],np.float32) # shape(1192,3,64,64,64)
        y = np.array(file['output'],np.float32) # shape(1192,1,64,64,64)
    # train test split
    xtr,xte = x[:1000],x[1000:]
    ytr,yte = y[:1000],y[1000:]

    xtr = torch.tensor(xtr,dtype=torch.float32)
    ytr = torch.tensor(ytr,dtype=torch.float32)
    xte = torch.tensor(xte,dtype=torch.float32)
    yte = torch.tensor(yte,dtype=torch.float32)
    train_dataset = TensorDataset(xtr, ytr)
    test_dataset = TensorDataset(xte, yte)

    ### torch data loader ###
    train_dataloader = DataLoader(train_dataset, batch_size= config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size= 8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    ### set a model ###
    model = Net().to(device)
   
    # resnet34 = ResNetFeatures().to(device)
    # loss_object = Lossfunction(resnet34,mse_loss_scale = 0.5,freq_loss_scale=0.5, perceptual_loss_scale=0.0)
    loss_object = SobelRelative(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], betas=(0.9, 0.999))

    ### start training ###
    start = time.time()
    # Assuming model, loss_object, optimizer, train_dataloader, test_dataloader, config, and device are defined
    trainer = Trainer(model, loss_object, optimizer, train_dataloader, test_dataloader, config, device)
    loss_history = trainer.run()
    end = time.time()
    print(f'running time:{(end-start)/60} mins')
    
    ### validation ###
    pred, target, test_loss,edge_loss,relative_loss2,relative_loss2 = trainer.test()
    print('Test Epoch: {} Loss: {:.4f}\n'.format(
                config["epochs"], test_loss))
    
    mean_error, median_error = mean_absolute_percentage_error(target,pred)
    print('mean relative error: {:.4f}\n, median relative error: {:.4f}'.format(mean_error,median_error))
    avg_ssim = calculate_ssim_batch(target,pred)
    print('SSIM: {:.4f}'.format(avg_ssim))

    # # plot pred-targ
    data = (loss_history,pred, target)
    
    with open("/home/dc-su2/physical_informed/cnn/rotate/results/history.pkl", "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    img_plt(target,pred,path='/home/dc-su2/physical_informed/cnn/rotate/results/img/')
    # history_plt(tr_losses,vl_losses,path='/home/dc-su2/physical_informed/cnn/rotate/results/')
    torch.save(model.state_dict(),'/home/dc-su2/physical_informed/cnn/rotate/results/model.pth')

if __name__ == '__main__':
    main()

