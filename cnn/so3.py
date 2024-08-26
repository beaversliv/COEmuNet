import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from utils.so3_model      import SO3Net


import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset,DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity

from utils.dataloader     import CustomTransform,SequentialDataset

from utils.loss           import SobelMse,Lossfunction,ResNetFeatures,mean_absolute_percentage_error, calculate_ssim_batch
from utils.plot           import img_plt,history_plt

import h5py as h5
import numpy as np
from tqdm import tqdm
import os
import sys
import time
import pickle
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from thop import profile
import subprocess
from sklearn.model_selection      import train_test_split 
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if torch.cuda.is_available():
    # Print the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    # Get the name of the current GPU (if there is more than one)
    current_gpu = torch.cuda.current_device()
    print(f"Name of the current GPU: {torch.cuda.get_device_name(current_gpu)}")
else:
    print("No GPU available, using CPU.")


def get_gpu_utilization():
    print('get utilization started')
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True)
        # Splitting lines and extracting the utilization from the first line
        utilization_str = result.stdout.strip().split('\n')[0]
        utilization = float(utilization_str)
        return utilization
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_gpu_memory():
    try:
        # Query the NVIDIA System Management Interface for GPU info
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], encoding='utf-8')
        # Convert output into a list
        gpu_memory = [x.split(',') for x in result.strip().split('\n')]
        gpu_memory = [{'used': int(x[0]), 'total': int(x[1])} for x in gpu_memory]
        return gpu_memory
    except subprocess.CalledProcessError as e:
        print(e.output)
        return None



seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--model_name', type = str, default = '3dResNet')
    parser.add_argument('--dataset', type = str, default = 'magritte')
    parser.add_argument('--epochs', type = int, default = 50)
    parser.add_argument('--batch_size', type = int, default = 8)
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

def get_data(path):
    sample = h5.File(path,'r')
    x  = np.array(sample['input'],np.float32)   # shape(1200,3,64,64,64)
    y = np.array(sample['output'], np.float32)# shape(1200,64,64,1)
    
    meta = {}

    x_t = np.transpose(x, (1, 0, 2, 3, 4))
    for idx in [0]:
        meta[idx] = {}
        meta[idx]['mean'] = x_t[idx].mean()
        meta[idx]['std'] = x_t[idx].std()
        x_t[idx] = (x_t[idx] - x_t[idx].mean())/x_t[idx].std()
    
    for idx in [1, 2]:
        meta[idx] = {}
        meta[idx]['min'] = np.min(x_t[idx])
        meta[idx]['median'] = np.median(x_t[idx])
        x_t[idx] = np.log(x_t[idx])
        
        x_t[idx] = x_t[idx] - np.min(x_t[idx])
        x_t[idx] = x_t[idx]/np.median(x_t[idx])
    
    y_v = y.reshape(-1)
    y = np.where(y == 0, np.min(y_v[y_v != 0]), y)
    y = np.log(y)
    y = y-np.min(y)
    y = y/np.median(y)
    
    return np.transpose(x_t, (1, 0, 2, 3, 4)), np.transpose(y,(0,3,1,2))

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
        self.model.train()
         
        for bidx, samples in enumerate(self.train_dataloader):
            data, target = Variable(samples[0]).to(self.device), Variable(samples[1]).to(self.device)
            # memory_info = get_gpu_memory()
            # print(f'GPU Memory usage after loading data in epoch 1 {memory_info}')
            self.optimizer.zero_grad()
            latent,output = self.model(data)

            loss = self.loss_object(target, output)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach().cpu().numpy()
            # memory_info = get_gpu_memory()
            # print(f'GPU Memory usage after an iteration in epoch 1 {memory_info}')
        epoch_loss = total_loss / len(self.train_dataloader)  # divide number of batches
        # memory_info = get_gpu_memory()
        # print(f'GPU Memory usage after epoch 1 {memory_info}')
        return epoch_loss

    
    def test(self):
        self.model.eval()
        P = []
        T = []
        L = []
        for bidx, samples in enumerate(self.test_dataloader):
            data, target = Variable(samples[0]).to(self.device), Variable(samples[1]).to(self.device)
            latent,pred = self.model(data)
            loss = self.loss_object(target, pred)
            
            P.append(pred.detach().cpu().numpy())
            T.append(target.detach().cpu().numpy())
            L.append(loss.detach().cpu().numpy())
        P = np.vstack(P)
        T = np.vstack(T)
        return P,T,np.mean(L)
    def run(self):
        tr_losses = []
        vl_losses = []
        for epoch in tqdm(range(self.config['epochs'])):
            epoch_loss = self.train()

            torch.cuda.empty_cache()  # Clear cache after training
            
            _, _, val_loss = self.test()
            torch.cuda.empty_cache()  # Clear cache after evaluation
            tr_losses.append(epoch_loss)
            vl_losses.append(val_loss)
            print('Train Epoch: {}/{} Loss: {:.4f}'.format(
                    epoch, self.config['epochs'], epoch_loss))
            print('Test Epoch: {}/{} Loss: {:.4f}\n'.format(
                epoch, self.config["epochs"], val_loss))
            
        return tr_losses, vl_losses


def main():
    config = parse_args()
    # file_statistics = '/home/dc-su2/physical_informed/cnn/rotate/12000_statistics.pkl'
    # file_paths = [f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/data_augment/clean_rotate12000_{i}.hdf5' for i in range(5)]

    # train_file_path = file_paths[:4]
    # test_file_path  = file_paths[4:]

    # custom_transform = CustomTransform(file_statistics)
    # train_dataset= IntensityDataset(train_file_path,transform=custom_transform)
    # train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,num_workers=2)

    # test_dataset= IntensityDataset(test_file_path,transform=custom_transform)
    # test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    path = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/sgl_freq/grid64/random/clean_2400_batches.hdf5'
    with h5.File(path,'r') as file:
        x = np.array(file['input'],np.float32) # shape(1192,3,64,64,64)
        y = np.array(file['output'],np.float32) # shape(1192,1,64,64,64)
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
    model = SO3Net().to(device)
   
    # resnet34 = ResNetFeatures().to(device)
    # loss_object = Lossfunction(resnet34,mse_loss_scale = 0.5,freq_loss_scale=0.5, perceptual_loss_scale=0.0)
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
    
    with open("/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/steerable/random_history.pkl", "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    
    mean_error, median_error = mean_absolute_percentage_error(target,pred)
    print('mean relative error: {:.4f}\n, median relative error: {:.4f}'.format(mean_error,median_error))
    avg_ssim = calculate_ssim_batch(target,pred)
    print('SSIM: {:.4f}'.format(avg_ssim))
    # plot pred-targ
    # img_plt(target,pred,path='/home/dc-su2/physical_informed/cnn/steerable/img/')
    # history_plt(tr_losses,vl_losses,path='/home/dc-su2/physical_informed/cnn/steerable/')
    # torch.save(model.state_dict(),'/home/dc-su2/physical_informed/cnn/steerable/model.pth')

if __name__ == '__main__':
    main()

