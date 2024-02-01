import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset,DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


from torch.profiler import profile, record_function, ProfilerActivity

from utils.so3_model      import ClsSO3Net
# from utils.model import Net
from utils.loss           import Lossfunction,ResNetFeatures,mean_absolute_percentage_error, calculate_ssim_batch
from utils.plot           import img_plt

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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

if torch.cuda.is_available():
    # Print the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    # Get the name of the current GPU (if there is more than one)
    current_gpu = torch.cuda.current_device()
    print(f"Name of the current GPU: {torch.cuda.get_device_name(current_gpu)}")
else:
    print("No GPU available, using CPU.")

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--model_name', type = str, default = '3dResNet')
    parser.add_argument('--dataset', type = str, default = 'magritte')
    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--batch_size', type = int, default = 16)
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
    x  = np.array(sample['input'],np.float32)   # shape(100,3,100,100,100)
    y = np.array(sample['output'][:,:,:,15:16], np.float32)# shape(100,1,256,256)
    
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
            
            # utilization = get_gpu_utilization()
            # if utilization is not None:
            #     print(f"GPU Utilization: {utilization}%")
            # else:
            #     print("Unable to retrieve GPU utilization.")
            # Check memory after loading data
            # memory_info = get_gpu_memory()
            # print(f'GPU Memory usage after loading data in epoch 1: {memory_info}')

            data, target = Variable(samples[0]).to(self.device), Variable(samples[1]).to(self.device)
            self.optimizer.zero_grad()
            latent,output = self.model(data)
            loss = self.loss_object(target, output)
            loss.backward()
            # # Unscales the gradients of optimizer's assigned params in-place
            # self.scaler.unscale_(self.optimizer)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=)
            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            # self.optimizer.step()
            # Check memory after an iteration
            # memory_info = get_gpu_memory()
            # print(f'GPU Memory usage after an iteration in epoch 1: {memory_info}\n')
            total_loss += loss.detach().cpu().numpy()
        epoch_loss = total_loss / len(self.train_dataloader)  # divide number of batches
         #Maybe check memory usage at the end of an epoch
        # memory_info = get_gpu_memory()
        # print(f'GPU Memory usage after epoch 1: {memory_info}')
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
    # start = time.time()
    path2 = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/data_augment/rotate1200.hdf5'
    x, y = get_data(path2)
    # print('time for load data from disk:',time.time()-start)
    xtr,xte = x[:1000],x[1000:]
    ytr,yte = y[:1000],y[1000:]

    xtr = torch.tensor(xtr,dtype=torch.float16)
    ytr = torch.tensor(ytr,dtype=torch.float16)
    xte = torch.tensor(xte,dtype=torch.float16)
    yte = torch.tensor(yte,dtype=torch.float16)

    train_dataset = TensorDataset(xtr, ytr)
    test_dataset = TensorDataset(xte, yte)

    ### torch data loader ###
    train_dataloader = DataLoader(train_dataset, batch_size= config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size= 8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    model = ClsSO3Net().to(device).half()
   
    ### Pre-trained VGG16 ###
    # vgg = VGGFeatures()
    # vgg.to(device)
    # vgg.eval()  # Important to set in evaluation mode!
    resnet34 = ResNetFeatures().to(device).half()
    loss_object = Lossfunction(resnet34,mse_loss_scale = 0.6,freq_loss_scale=0.2, perceptual_loss_scale=0.2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], betas=(0.9, 0.999))

    ### start training ###
    start = time.time()
    # Assuming model, loss_object, optimizer, train_dataloader, test_dataloader, config, and device are defined
    trainer = Trainer(model, loss_object, optimizer, train_dataloader, test_dataloader, config, device)
    tr_losses, vl_losses = trainer.run()

    # world_size = torch.cuda.device_count()
    # mp.spawn(ddp_run, args=(world_size, model, device, config, train_dataloader, test_dataloader, optimizer), nprocs=world_size, join=True)
    end = time.time()
    print(f'running time:{(end-start)/60} mins')

    ### validation ###
    # pred, target, test_loss = trainer.test()
    # print('Test Epoch: {} Loss: {:.4f}\n'.format(
    #             config["epochs"], test_loss))
    # data = (tr_losses, vl_losses,pred, target)
    
    # # losses.update(pred_targ)
    # with open("/home/s/ss1421/Documents/physical_informed_surrogate_model/cnn/steerable/history.pkl", "wb") as pickle_file:
    #     pickle.dump(data, pickle_file)
    

    # mean_error, median_error = mean_absolute_percentage_error(target,pred)
    # print('mean relative error: {:.4f}\n, median relative error: {:.4f}'.format(mean_error,median_error))
    # avg_ssim = calculate_ssim_batch(target,pred)
    # print('SSIM: {:.4f}'.format(avg_ssim))
    # # plot pred-targ
    # img_plt(target,pred,path='/home/s/ss1421/Documents/physical_informed_surrogate_model/cnn/steerable/img/')
    # torch.save(model.state_dict(),'/home/s/ss1421/Documents/physical_informed_surrogate_model/cnn/steerable/model.pth')

if __name__ == '__main__':
    main()