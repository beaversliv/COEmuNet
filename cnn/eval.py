import torch
import torch.nn       as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd   import Variable

from utils.ResNet3DModel  import Net
from utils.dataloader     import CustomTransform,IntensityDataset
from utils.loss           import SobelMse
from utils.preprocessing  import preProcessing

import time
import numpy              as np
import h5py               as h5
import logging
from utils.ResNet3DModel  import Net3D,Net
from thop                   import profile
from sklearn.model_selection      import train_test_split 
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from utils.so3_model      import SO3Net

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
def eval(test_dataloader,device,model_path:str):
    odel = Net().to(device)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()
    P = []
    T = []
    L = []
    for bidx, samples in enumerate(test_dataloader):
        data, target = Variable(samples[0]).to(device), Variable(samples[1]).to(device)
        start = time.time()
        pred = model(data)
        end  = time.time()
        running_time = end - start
        logging.info(f'Running time: {running_time} seconds')
        # loss = loss_object(target, pred)
        
        P.append(pred.detach().cpu().numpy())
        T.append(target.detach().cpu().numpy())
        L.append(loss.detach().cpu().numpy())

    P = np.vstack(P)
    T = np.vstack(T)
    return P,T
def main():
    # file_statistics = '/home/dc-su2/physical_informed/cnn/rotate/12000_statistics.pkl'
    # custom_transform = CustomTransform(file_statistics)
    # test_file_path  = ['/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/grid64/rotation/clean_rotate12000_4.hdf5']
    # test_dataset= IntensityDataset(test_file_path,transform=custom_transform)
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    data_gen = preProcessing('/home/dc-su2/rds/rds-dirac-dr004/Magritte/faceon_grid64_data0.hdf5')
    x,y = data_gen.get_data()

    # train test split
    xtr, xte, ytr,yte = train_test_split(x,y,test_size=0.2,random_state=42)

    xte = torch.tensor(xte,dtype=torch.float32)
    yte = torch.tensor(yte,dtype=torch.float32)
    test_dataset = TensorDataset(xte, yte)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_dic = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/results/test_model.pth'
    P,T = eval(test_dataloader,device,model_dic)
    # time_eval(log_file='/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/freq31_runtime64.log',
    #       test_dataloader=test_dataloader,
    #       device=device,
    #       model_path=model_dic,
    #       grid=64)

    with open("/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/results/test_history.pkl", "wb") as pickle_file:
        pickle.dump({
            'history':{'train_loss':0.0,'val_loss':0.0},
            'target':target,
            'prediction':pred
        }, pickle_file)
if __name__ == '__main__':
    main()