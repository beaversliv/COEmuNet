import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

import h5py as h5
import numpy as np
import os
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
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
np.random.seed(1234)
torch.manual_seed(1234) 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--model_name', type = str, default = '3dResNet')
    parser.add_argument('--dataset', type = str, default = 'p3droslo')
    parser.add_argument('--epochs', type = int, default = 50)
    parser.add_argument('--batch_size', type = int, default = 30)
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
config = parse_args()

def get_data(path):
    with h5.File(path,'r') as sample:
        x  = np.array(sample['input'])   # shape(80,4,128,128,128)
        y =  np.array(sample['output'][:,:,:,15:16]) # shape(80,128,128,31)
    
    meta = {}
    
    x_t = np.transpose(x, (1, 0, 2, 3, 4))
    for idx in [0, 1]:
        meta[idx] = {}
        meta[idx]['mean'] = x_t[idx].mean()
        meta[idx]['std'] = x_t[idx].std()
        x_t[idx] = (x_t[idx] - x_t[idx].mean())/x_t[idx].std()
    
    for idx in [2, 3]:
        meta[idx] = {}
        meta[idx]['min'] = np.min(x_t[idx])
        meta[idx]['median'] = np.median(x_t[idx])
        x_t[idx] = np.log(x_t[idx])
        x_t[idx] = x_t[idx] - np.min(x_t[idx])
        x_t[idx] = x_t[idx]/np.median(x_t[idx])
    
    y = np.log(y)
    y = y-np.min(y)
    y = y/np.median(y)
    
    return np.transpose(x_t, (1, 0, 2, 3, 4)), y.transpose(0,3,1,2)

path = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/pism_forward/Demo/data.hdf5'
x, y = get_data(path)
x = torch.Tensor(x)
y = torch.Tensor(y)

train_dataset = TensorDataset(x, y)

### torch data loader ###
train_dataloader = DataLoader(train_dataset, batch_size= config['batch_size'], shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size= 10, shuffle=True)

### Resnet ###
class Conv_BN_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv_BN_Relu,self).__init__()
        self.layers =  nn.ModuleList()
        self.layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride = stride, padding = padding))
        self.layers.append(nn.BatchNorm3d(out_channels))
        self.layers.append(nn.ReLU())
    
    def forward(self,x):   
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return  x

class residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, skip=False):
        super(residual_Block,self).__init__()
        self.skip = skip
        
        self.CBR0 =  Conv_BN_Relu(in_channels, out_channels, kernel_size, stride, padding)
        self.CBR1 =  Conv_BN_Relu(out_channels, out_channels, kernel_size, stride = 1, padding = 'same')
        self.short_CBR0 = Conv_BN_Relu(out_channels, out_channels, kernel_size = (1, 1, 1), stride = 1, padding = 0)
        
    def forward(self, x):  
        residual = x
        x = self.CBR0(x)
        x = self.CBR1(x)
        
        if self.skip :
            # convolutional residual block
            shortcut = self.short_CBR0(x)
            y = x + shortcut
            return y
        else:
          # identical residual block
            y = x + residual
            return y    
        
class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder,self).__init__()
        self.layers =  nn.ModuleList()
        self.layers.append(Conv_BN_Relu(in_channels, 4, kernel_size=3, stride=1, padding = 0))
        self.layers.append(nn.MaxPool3d(2))
        
        self.layers.append(residual_Block(4, 8, kernel_size=3, stride=2, padding = 1, skip=True))
        self.layers.append(residual_Block(8, 8, kernel_size=3, stride=1, padding = 'same', skip=False))
        self.layers.append(residual_Block(8, 8, kernel_size=3, stride=1, padding = 'same', skip=False))
        
        self.layers.append(residual_Block(8, 16, kernel_size=3, stride=2, padding = 1, skip=True))
        self.layers.append(residual_Block(16, 16, kernel_size=3, stride=1, padding = 'same', skip=False))
        self.layers.append(residual_Block(16, 16, kernel_size=3, stride=1, padding = 'same', skip=False))        

        self.layers.append(residual_Block(16, 32, kernel_size=3, stride=2, padding = 1, skip=True))
        self.layers.append(residual_Block(32, 32, kernel_size=3, stride=1, padding = 'same', skip=False))
        self.layers.append(residual_Block(32, 32, kernel_size=3, stride=1, padding = 'same', skip=False))        
        
        self.layers.append(nn.MaxPool3d(2)) #if not 7*7*7

    def forward(self, x):   
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return  x
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder0 = Encoder(1)
        self.encoder1 = Encoder(1)
        self.encoder2 = Encoder(1)
        self.encoder3 = Encoder(1)
        
        self.to_lat = nn.Linear(32*4*4*4*4, 8*8*8)
        self.to_dec = nn.Linear(8*8*8, 64*8*8) #64*8*8
        
        self.decoder= nn.ModuleList()
        
        in_channel = 64
        
        for idx in range(4):
            sub_seq = nn.ModuleList([nn.ConvTranspose2d(in_channel, int(in_channel/2), 4, 2, 
                                                       padding = 1),#, output_padding = 1),
                                     nn.BatchNorm2d(int(in_channel/2)),
                                     nn.ReLU()])
            self.decoder.append(sub_seq)
            in_channel = int(in_channel/2)

        sub_seq = nn.ModuleList([nn.Conv2d(in_channel, 1, 3, 
                                                   padding = 1)])   
        self.decoder.append(sub_seq)
        
    def forward(self, x):
        x0 = self.encoder0(x[:, 0:1, :, :, :])
        x1 = self.encoder1(x[:, 1:2, :, :, :])
        x2 = self.encoder2(x[:, 2:3, :, :, :])
        x3 = self.encoder3(x[:, 3:4, :, :, :])
        
        x0 = torch.flatten(x0, start_dim=1)   
        x1 = torch.flatten(x1, start_dim=1)   
        x2 = torch.flatten(x2, start_dim=1)   
        x3 = torch.flatten(x3, start_dim=1)   
        
        x = torch.cat([x0, x1, x2, x3], dim = -1)
        
        x = self.to_lat(x)
        x = nn.ReLU()(self.to_dec(x))
        x = x.view(-1, 64, 8, 8)
	# shape (batch_size,64,8,8)
        
        for idx in range(len(self.decoder)):
            for cidx in range(len(self.decoder[idx])):
                x = self.decoder[idx][cidx](x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### set a model ###
model = Net()
pred_y = model.forward(x)
model.to(device)   

loss_object = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, betas=(0.9, 0.999))

### train step ###
def train(epoch):
    epoch_loss = 0.
    model.train() 
    for bidx, samples in enumerate(train_dataloader):
        data, target = Variable(samples[0]).to(device), Variable(samples[1]).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_object(target, output)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().cpu().numpy()
    print('Train Epoch: {}/{} Loss: {:.4f}'.format(
            epoch, config['epochs'], epoch_loss))
    return epoch_loss

### train step ###
def test(epoch):
    model.eval()
    P = []
    T = []
    L = []
    for bidx, samples in enumerate(test_dataloader):
        data, target = Variable(samples[0]).to(device), Variable(samples[1]).to(device)
        pred = model(data)
        loss = loss_object(target, pred)
        
        P.append(pred.detach().cpu().numpy())
        T.append(target.detach().cpu().numpy())
        L.append(loss.detach().cpu().numpy())
    
    print('Test Epoch: {}/{} Loss: {:.4f}'.format(
            epoch, 50, np.mean(L)))
    P = np.vstack(P)
    T = np.vstack(T)
    return P, T, L
### run ###
def run():
    losses = []
    for epoch in range(config['epochs']):
        epoch_loss = train(epoch)
        losses.append(epoch_loss)
        
    return losses

def main():
    losses = run()
    pred, target, _ = test(config['epochs'])
    plt.plot(np.arange(config['epochs']),losses)
    plt.savefig('/home/dc-su2/physical_informed/cnn/img/history.png')


    for i in range(target.shape[0]):
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(target[i][0])
        axs[0].set_title('target')
        axs[1].imshow(pred[i][0])
        axs[1].set_title('prediction')
        plt.savefig('/home/dc-su2/physical_informed/cnn/img/ex{}.png'.format(i))
        plt.close()

if __name__ == '__main__':
    main()
