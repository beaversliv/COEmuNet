import torch
import torch.nn       as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd   import Variable

from utils.model          import Net
from utils.dataloader     import CustomTransform,IntensityDataset
from utils.loss           import SobelMse

import time
import numpy              as np
import logging
logging.basicConfig(filename='/home/dc-su2/physical_informed/cnn/original/runtime.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


file_statistics = '/home/dc-su2/physical_informed/cnn/original/clean_statistics.pkl'
custom_transform = CustomTransform(file_statistics)
test_file_path  = ['/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/faceon/clean_test.hdf5']
test_dataset= IntensityDataset(test_file_path,transform=custom_transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
device = torch.device("cpu")
loss_object = SobelMse(device)


model_dic = '/home/dc-su2/physical_informed/cnn/original/results/old_model.pth'
model = Net()
model.load_state_dict(torch.load(model_dic,map_location=torch.device('cpu')))
model.eval()

P = []
T = []
L = []
for bidx, samples in enumerate(test_dataloader):
    data, target = Variable(samples[0]).to(device), Variable(samples[1]).to(device)
    start = time.time()
    latent,pred = model(data)
    end  = time.time()
    running_time = end - start
    logging.info(f'Running time: {running_time} seconds')
    loss = loss_object(target, pred)
    
    P.append(pred.detach().cpu().numpy())
    T.append(target.detach().cpu().numpy())
    L.append(loss.detach().cpu().numpy())

P = np.vstack(P)
T = np.vstack(T)

