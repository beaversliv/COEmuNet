import torch
import torch.nn       as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd   import Variable

from utils.ResNet3DModel  import Net
from utils.dataloader     import CustomTransform,IntensityDataset
from utils.loss           import SobelMse

import time
import numpy              as np
import h5py               as h5
import logging
from utils.ResNet3DModel  import Net3D,Net
from thop import profile

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from utils.so3_model      import SO3Net

class preProcessing:
    def __init__(self,path):
        self.path = path

    def outliers(self):
        with h5.File(self.path,'r') as sample:
            input_ = np.array(sample['input'],np.float32)   # shape(num_samples,3,64,64,64)
            output_ = np.array(sample['output'], np.float32)# shape(num_samples,64,64,1)
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
        # y = y[:,:,:,12:19]
        y = np.log(y)
        # set threshold
        y[y<=-60] = -60
        # pre-processing: reflect and take base 10 logrithmn
        y -= np.min(y)
        reflection_point = y.max() + 1
        y = reflection_point - y
        y = np.log10(y)
        
        # y = (y - np.min(y))/ (np.max(y) - np.min(y))
        y = np.transpose(y,(0,3,1,2))
        y = y[:,np.newaxis,:,:,:]
        return np.transpose(x_t, (1, 0, 2, 3, 4)), y

def time_eval(log_file:str,test_dataloader,device,model_path:str,grid:int):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    model = Net3D(freq=31).to(device)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()

    # P = []
    # T = []
    # L = []
    for bidx, samples in enumerate(test_dataloader):
        data, target = Variable(samples[0]).to(device), Variable(samples[1]).to(device)
        start = time.time()
        pred = model(data)
        end  = time.time()
        running_time = end - start
        logging.info(f'Running time: {running_time} seconds')
        # loss = loss_object(target, pred)
        
    #     P.append(pred.detach().cpu().numpy())
    #     T.append(target.detach().cpu().numpy())
    #     L.append(loss.detach().cpu().numpy())

    # P = np.vstack(P)
    # T = np.vstack(T)

def main():
    # file_statistics = '/home/dc-su2/physical_informed/cnn/rotate/12000_statistics.pkl'
    # custom_transform = CustomTransform(file_statistics)
    # test_file_path  = ['/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/grid64/rotation/clean_rotate12000_4.hdf5']
    # test_dataset= IntensityDataset(test_file_path,transform=custom_transform)
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    data_gen = preProcessing('/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/mul_freq/long_1200.hdf5')
    x,y = data_gen.get_data()
    # with h5.File(path,'r') as file:
    #     x = np.array(file['input'],np.float32) # shape(1192,3,64,64,64)
    #     y = np.array(file['output'],np.float32) # shape(1192,1,64,64,64)

    # train test split
    xte = x[-1090:]
    yte = y[-1090:]

    xte = torch.tensor(xte,dtype=torch.float32)
    yte = torch.tensor(yte,dtype=torch.float32)
    test_dataset = TensorDataset(xte, yte)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_dic = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/rotate/results/mul/random_Multi_model.pth'
    time_eval(log_file='/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/freq31_runtime64.log',
          test_dataloader=test_dataloader,
          device=device,
          model_path=model_dic,
          grid=64)

if __name__ == '__main__':
    main()