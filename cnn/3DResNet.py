import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
# custom helper functions
from utils.dataloader     import CustomTransform,IntensityDataset
from utils.ResNet3DModel          import Net
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
 
            self.optimizer.zero_grad()
            output = self.model(data)
            # print(target.shape)
            # print(output.shape)
            
            # sys.exit()
            loss = self.loss_object(target, output)
            # sys.exit()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach().cpu().numpy()
        epoch_loss = total_loss / len(self.train_dataloader)  # divide number of batches
        return epoch_loss

    
    def test(self):
        self.model.eval()
        P = []
        T = []
        L = []
        for bidx, samples in enumerate(self.test_dataloader):
            data, target = Variable(samples[0]).to(self.device), Variable(samples[1]).to(self.device)
            pred = self.model(data)
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
class preProcessing:
    def __init__(self,path):
        self.path = path

    def outliers(self):
        with h5.File(self.path,'r') as sample:
            x = np.array(sample['input'],np.float32)   # shape(num_samples,3,64,64,64)
            y = np.array(sample['output'], np.float32)# shape(num_samples,64,64,1)
        # take logrithm
        y[y==0] = np.min(y[y!=0])
        I = np.log(y)
        # difference = max - min
        max_values = np.max(I,axis=(1,2))
        min_values = np.min(I,axis=(1,2))
        diff = max_values - min_values
        # find outliers
        outlier_idx = np.where(diff>20)[0]

        # remove outliers
        removed_x = np.delete(x,outlier_idx,axis=0)
        removed_y = np.delete(y,outlier_idx,axis=0)
        return removed_x, removed_y

    def get_data(self):
        x , y = self.outliers()
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
        
        y[y == 0] = np.min(y[y != 0])
        y = np.log(y)
        print('min',np.min(y))
        y = y-np.min(y)
        print('median',np.median(y))
        y = y/np.median(y)
        return np.transpose(x_t, (1, 0, 2, 3, 4)), np.transpose(y,(0,3,1,2))
def main():
    config = parse_args()
    # file paths for train, vali and test
    # file_statistics = '/home/dc-su2/physical_informed/cnn/original/clean_statistics.pkl'
    # train_file_path = [f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/faceon/new_train_{i}.hdf5' for i in range(4)]
    # vali_file_path  = ['/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/faceon/new_vali.hdf5']
    # test_file_path  = ['/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/faceon/new_test.hdf5']

    # custom_transform = CustomTransform(file_statistics)
    # train_dataset= IntensityDataset(train_file_path,transform=custom_transform)
    # train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,num_workers=2)

    # vali_dataset= IntensityDataset(vali_file_path,transform=custom_transform)
    # vali_dataloader = DataLoader(vali_dataset, batch_size=config['batch_size'], shuffle=True,num_workers=2)

    # test_dataset= IntensityDataset(test_file_path,transform=custom_transform)
    # test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    data_gen = preProcessing('/home/dc-su2/rds/rds-dirac-dr004/Magritte/faceon_grid64_data0.hdf5')
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
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### set a model ###
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
    for freq in range(len(avg_ssim)):
        print(f'frequency {freq + 1} has ssim {avg_ssim[freq]:.4f}')

    # plot and save history
    # img_plt(target,pred,path='/home/dc-su2/physical_informed/cnn/original/results/img/')
    # history_plt(tr_losses,vl_losses,path='/home/dc-su2/physical_informed/cnn/original/results/')
    with open("/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/results/test_history.pkl", "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    # torch.save(model.state_dict(),'/home/dc-su2/physical_informed/cnn/original/results/new_model.pth')


if __name__ == '__main__':
    main()
