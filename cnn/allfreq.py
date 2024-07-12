import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
# custom helper functions
from utils.dataloader     import CustomTransform,IntensityDataset
from utils.ResNet3DModel  import Net3D,Net
from utils.trainclass     import Trainer
from utils.loss           import relativeLoss,RelativeLoss,SobelMse,mean_absolute_percentage_error, calculate_ssim_batch
from utils.plot           import img_plt,history_plt

# helper packages
import h5py as h5
import numpy as np
import os
import sys
import time
import logging
from tqdm                         import tqdm
import pickle
import argparse
from collections                  import OrderedDict
import matplotlib.pyplot          as plt
from sklearn.model_selection      import train_test_split 
from sklearn.preprocessing        import QuantileTransformer
import matplotlib.pyplot          as plt
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
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--lr_decay', type = float, default = 0.95)
    parser.add_argument('--num_freqs',type=int,default=7)


    args = parser.parse_args()
    
    config = OrderedDict([
            ('path_dir', args.path_dir),
            ('model_name', args.model_name),
            
            ('dataset', args.dataset),
            ('epochs', args.epochs),
            ('batch_size', args.batch_size),
            ('lr', args.lr),
            ('lr_decay', args.lr_decay),
            ('num_freqs', args.num_freqs)
            ])
    return config

class preProcessing:
    def __init__(self,path,num_freqs=7):
        self.path = path
        self.num_freqs = num_freqs

    def outliers(self):
        with h5.File(self.path,'r') as sample:
            input_ = np.array(sample['input'],np.float32)   # shape(num_samples,3,64,64,64)
            output_ = np.array(sample['output'], np.float32)# shape(num_samples,64,64,1)
        # return input_, output_
        # take logrithm
        y = output_[:,:,:,15]
        y[y==0] = np.min(y[y!=0])
        I = np.log(y)
        # difference = max - min
        max_values = np.max(I,axis=(1,2))
        min_values = np.min(I,axis=(1,2))
        diff = max_values - min_values
        # find outliers
        outlier_idx = np.where(min_values < -60)[0]

        # remove outliers
        removed_x = np.delete(input_,outlier_idx,axis=0)
        removed_y = np.delete(output_,outlier_idx,axis=0)
        return removed_x, removed_y

    def get_data(self):
        x,y = self.outliers()
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
        y[y==0] = np.min(y[y!=0])
        segment = self.num_freqs//2
        y = y[:,:,:,15-segment:15+segment+1]
        y = np.log(y)
        # set threshold
        y[y<=-40] = -40
        # pre-processing: reflect and take base 10 logrithmn
        print('min:',np.min(y))
        print('max:',np.max(y))
        y = (y - np.min(y))/ (np.max(y) - np.min(y))
        # reflection_point = y.max() + 1
        # reflection_point = 1.0
        # y = reflection_point - y
        # print('reflection point:',reflection_point)
        # y = np.log10(y)
        
        # y = (y - np.min(y))/ (np.max(y) - np.min(y))
        y = np.transpose(y,(0,3,1,2))
        y = y[:,np.newaxis,:,:,:]
        return np.transpose(x_t, (1, 0, 2, 3, 4)), y
# class Trainer:
#     def __init__(self,model,loss_object,optimizer,train_dataloader,test_dataloader,config,device):
#         self.model = model
#         self.loss_object = loss_object
#         self.optimizer   = optimizer
#         self.train_dataloader = train_dataloader
#         self.test_dataloader  = test_dataloader
#         self.config = config
#         self.device = device
    
#     def train(self):
#         total_SM = 0.
#         total_rl = 0.
#         self.model.train()
         
#         for bidx, samples in enumerate(self.train_dataloader):
#             data, target = Variable(samples[0]).to(self.device), Variable(samples[1]).to(self.device)
#             self.optimizer.zero_grad()
#             output = self.model(data)
#             sobleMSE,maxrl = self.loss_object(target, output)
#             loss = sobleMSE + maxrl
#             loss.backward()
#             self.optimizer.step()

#             total_SM += sobleMSE.detach().cpu().numpy()
#             total_rl += maxrl.detach().cpu().numpy()

#         epoch_SM = total_SM / len(self.train_dataloader)  # divide number of batches
#         epoch_rl = total_rl / len(self.train_dataloader)
#         return epoch_SM,epoch_rl

    
#     def test(self):
#         self.model.eval()
#         P = []
#         T = []
#         SM = []
#         rl = []
#         for bidx, samples in enumerate(self.test_dataloader):
#             data, target = Variable(samples[0]).to(self.device), Variable(samples[1]).to(self.device)
#             pred = self.model(data)
#             sobleMSE,maxrl = self.loss_object(target, pred)
            
#             P.append(pred.detach().cpu().numpy())
#             T.append(target.detach().cpu().numpy())
#             SM.append(sobleMSE.detach().cpu().numpy())
#             rl.append(maxrl.detach().cpu().numpy())
#         P = np.vstack(P)
#         T = np.vstack(T)
#         return P,T,np.mean(SM),np.mean(rl)
#     def run(self):
#         tr_losses = []
#         vl_losses = []
#         for epoch in tqdm(range(self.config['epochs'])):
#             tr_sm,tr_rl = self.train()
#             torch.cuda.empty_cache()  # Clear cache after training
            
#             _, _, val_sm,val_rl = self.test()
#             torch.cuda.empty_cache()  # Clear cache after evaluation
#             tr_losses.append((tr_sm,tr_rl))
#             vl_losses.append((val_sm,val_rl))
#             print('Train Epoch: {}/{} Loss: {:.4f}'.format(
#                     epoch, self.config['epochs'], tr_sm+tr_rl))
#             print('Test Epoch: {}/{} Loss: {:.4f}\n'.format(
#                 epoch, self.config['epochs'], val_sm+val_rl))
            
#         return tr_losses, vl_losses

def postprocessing(y):
    min_ = -40.0
    max_ = -27.664127
    a = y * (max_ - min_) + min_
    return a

def meanMaxPercentageError(original_target,original_pred):
    return np.mean( np.abs(original_target-original_pred) / np.max(original_target, axis=1,keepdims=True)) *100


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
    # path = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/mul_freq/clean_random_1200.hdf5'
    # with h5.File(path,'r') as file:
    #     x = np.array(file['input'],np.float32) # shape(1192,3,64,64,64)
    #     y = np.array(file['output'],np.float32) # shape(1192,1,64,64,64)
    # with h5.File('/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/sgl_freq/grid64/random/clean_batches.hdf5','r') as sample:
    #     x = np.array(sample['input'],np.float32)   # shape(num_samples,3,64,64,64)
    #     y = np.array(sample['output'], np.float32)# shape(num_samples,64,64,1)
    data_gen = preProcessing('/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/mul_freq/long_1200.hdf5',config['num_freqs'])
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
    model = Net3D(config['num_freqs']).to(device)

    loss_object = SobelMse(device)
    # loss_object = relativeLoss(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], betas=(0.9, 0.999))

    ### start training ###
    start = time.time()
    # Assuming model, loss_object, optimizer, train_dataloader, test_dataloader, config, and device are defined
    trainer = Trainer(model, loss_object, optimizer, train_dataloader, test_dataloader, config, device)
    tr_losses, vl_losses = trainer.run()
    end = time.time()
    print(f'running time:{(end-start)/60} mins')
    
    ### validation ###
    pred, target, _ = trainer.test()
    # print('Test Epoch: {} Loss: {:.4f}\n'.format(
    #             config["epochs"], test_loss))
    data = (tr_losses, vl_losses,pred, target)
    
    with open(f"/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/rotate/results/mul/test_history.pkl", "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    # post processing
    target = target[:,0,:,:,:]
    pred = pred[:,0,:,:,:]
     
    log_target = postprocessing(target)
    log_pred   = postprocessing(pred)
    original_target = np.exp(log_target)
    original_pred = np.exp(log_pred)
    print('relative loss',meanMaxPercentageError(original_target,original_pred))

    avg_ssim = calculate_ssim_batch(log_target,log_pred)
    for freq in range(len(avg_ssim)):
        print(f'frequency {freq + 1} has ssim {avg_ssim[freq]:.4f}')

    # img_plt(target[:200],pred[:200],path='/home/dc-su2/physical_informed/cnn/rotate/results/img/')
    # history_plt(tr_losses,vl_losses,path='/home/dc-su2/physical_informed/cnn/rotate/results/')
    # torch.save(model.state_dict(),'/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/rotate/results/mul/random_Multi7_model.pth')

if __name__ == '__main__':
    main()

