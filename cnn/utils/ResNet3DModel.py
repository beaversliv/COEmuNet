import torch
import torch.nn as nn
import torchvision.models as models
from collections import namedtuple
import sys
import torch.nn.init as init

# Define custom He initialization function
def he_init(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        init.kaiming_normal_(layer.weight, nonlinearity='relu')  # He initialization for ReLU-based networks
        if layer.bias is not None:
            layer.bias.data.fill_(0)
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
        self.layers.append(Conv_BN_Relu(in_channels, 4, kernel_size=3, stride=1, padding = 0)) # [batch, 4, 62, 62, 62]
        self.layers.append(nn.MaxPool3d(2))                                                    # [batch, 4, 31, 31, 31]
        
        self.layers.append(residual_Block(4, 8, kernel_size=3, stride=2, padding = 1, skip=True)) # [batch, 8, 16, 16, 16]
        self.layers.append(residual_Block(8, 8, kernel_size=3, stride=1, padding = 'same', skip=False)) # [batch, 8, 16, 16, 16]
        self.layers.append(residual_Block(8, 8, kernel_size=3, stride=1, padding = 'same', skip=False)) # [batch, 8, 16, 16, 16]
        # self.layers.append(residual_Block(8, 8, kernel_size=3, stride=1, padding = 'same', skip=False))
        
        self.layers.append(residual_Block(8, 16, kernel_size=3, stride=2, padding = 1, skip=True))        # [batch, 16, 8,8,8]
        self.layers.append(residual_Block(16, 16, kernel_size=3, stride=1, padding = 'same', skip=False)) # [batch, 16, 8,8,8]
        self.layers.append(residual_Block(16, 16, kernel_size=3, stride=1, padding = 'same', skip=False)) # [batch, 16, 8,8,8]
        # self.layers.append(residual_Block(16, 16, kernel_size=3, stride=1, padding = 'same', skip=False))        

        self.layers.append(residual_Block(16, 32, kernel_size=3, stride=2, padding = 1, skip=True)) # [batch, 32, 4,4,4]
        self.layers.append(residual_Block(32, 32, kernel_size=3, stride=1, padding = 'same', skip=False)) # [batch, 32, 4,4,4]
        self.layers.append(residual_Block(32, 32, kernel_size=3, stride=1, padding = 'same', skip=False)) # [batch, 32, 4,4,4]
        # self.layers.append(residual_Block(32, 32, kernel_size=3, stride=1, padding = 'same', skip=False))

        # self.layers.append(residual_Block(32, 64, kernel_size=3, stride=2, padding = 1, skip=True))
        # self.layers.append(residual_Block(64, 64, kernel_size=3, stride=1, padding = 'same', skip=False))
        # self.layers.append(residual_Block(64, 64, kernel_size=3, stride=1, padding = 'same', skip=False))
        # self.layers.append(nn.MaxPool3d(2)) #if not 7*7*7

    def forward(self, x):   
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return  x

class Decoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()

        # Example: Halving the channels and doubling the spatial dimension with each step
        for i in range(3):
            self.layers.append(nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=3, stride=1,padding=1))
            self.layers.append(nn.BatchNorm2d(int(in_channels / 2)))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))  # Upsampling
            in_channels = int(in_channels/2)

        # Final convolution to get the desired number of output channels (1 in this case)
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Net(nn.Module):
    def __init__(self,model_grid=64):
        super(Net, self).__init__()
        self.model_grid = model_grid
        self.encoder0 = Encoder(1)
        self.encoder1 = Encoder(1)
        self.encoder2 = Encoder(1)
        self.relu     = nn.ReLU()
        # grid 64
        if model_grid == 32:
            self.to_lat = nn.Linear(32*2*2*2*3,16*16*16)
            self.to_dec = nn.Linear(16*16*16,64*4*4)
        elif model_grid == 64:
            self.to_lat1 = nn.Linear(32*4*4*4*3,16*16*16)
            self.to_lat2 = nn.Linear(16*16*16,512)
            self.to_dec1 = nn.Linear(512,512)
            self.to_dec2 = nn.Linear(512,64*8*8)
        elif model_grid == 128:
            self.to_lat = nn.Linear(32*8*8*8*3,16*16*16)
            self.to_dec = nn.Linear(16*16*16,64*16*16)
        self.decoder= Decoder(in_channels=64, out_channels=1)
        
        
    def forward(self, x):
        x0 = self.encoder0(x[:, 0:1, :, :, :])
        x1 = self.encoder1(x[:, 1:2, :, :, :])
        x2 = self.encoder2(x[:, 2:3, :, :, :])
      
        # x0 shape (batch size, 32*4*4*4)
        x0 = torch.flatten(x0, start_dim=1)   
        x1 = torch.flatten(x1, start_dim=1)   
        x2 = torch.flatten(x2, start_dim=1) 
        # x shape (batch size, 32*4*4*4*3)
        x = torch.cat([x0, x1, x2], dim = -1)
 
        # (batch, 16*16*16)
        x        = self.relu(self.to_lat1(x)) #dense layer
        x_latent = self.relu(self.to_lat2(x)) #dense layer
        x = self.relu((self.to_dec1(x_latent)))
        x = self.relu((self.to_dec2(x)))
        # grid 64
        if self.model_grid == 32:
            x = x.view(-1, 64, 4, 4)
        elif self.model_grid == 64:
            x = x.view(-1, 64, 8, 8)
        elif self.model_grid == 128:
            x = x.view(-1, 64, 16, 16)
        
	    # shape (batch_size,64,8,8)
        output = self.decoder(x)
        return output

class Decoder3D(nn.Module):
    def __init__(self, in_channels=64, out_channels=1):
        super(Decoder3D, self).__init__()
        self.layers = nn.ModuleList()
        # Example: Halving the channels and doubling the spatial dimension with each step
        for i in range(3):
            # self.layers.append(residual_Block(in_channels, int(in_channels / 2), kernel_size=(1,3,3), stride=(1,1,1), padding = (0,1,1), skip=True))
            # self.layers.append(residual_Block(int(in_channels / 2), int(in_channels / 2), kernel_size=(1,3,3), stride=(1,1,1), padding = 'same', skip=False)) # [batch, 8, 16, 16, 16]
            # self.layers.append(nn.Upsample(scale_factor=(1,2,2), mode='nearest'))  # Upsampling
            self.layers.append(nn.Conv3d(in_channels, int(in_channels / 2), kernel_size=(1,3,3), stride=(1,1,1),padding=(0,1,1)))
            self.layers.append(nn.BatchNorm3d(int(in_channels / 2)))
            self.layers.append(nn.ReLU())

            self.layers.append(nn.Conv3d(int(in_channels / 2), int(in_channels / 2), kernel_size=(1,3,3), stride=(1,1,1),padding=(0,1,1)))
            self.layers.append(nn.BatchNorm3d(int(in_channels / 2)))
            self.layers.append(nn.ReLU())

            self.layers.append(nn.Conv3d(int(in_channels / 2), int(in_channels / 2), kernel_size=(1,3,3), stride=(1,1,1),padding=(0,1,1)))
            self.layers.append(nn.BatchNorm3d(int(in_channels / 2)))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Upsample(scale_factor=(1,2,2), mode='nearest'))  # Upsampling
            
            in_channels = int(in_channels/2)
        # Final convolution to get the desired number of output channels (1 in this case)
        self.layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3), padding=(0,1,1)))
        # self.layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=(1,3,3), padding=(0,1,1)))
        self.layers.append(nn.Tanh())

    def forward(self, x):
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        # print(x.shape)
        return x

class Net3D(nn.Module):
    def __init__(self,freq=31):
        super(Net3D, self).__init__()
        # self.apply(he_init)
        self.freq     = freq     # number of frequencies
        self.encoder0 = Encoder(1)
        self.encoder1 = Encoder(1)
        self.encoder2 = Encoder(1)
        # grid 64
        self.to_lat = nn.Linear(32*4*4*4*3,16*16*16)
        self.to_dec = nn.Linear(16*16*16,64*8*8*self.freq)
        
        self.decoder3d= Decoder3D(in_channels=64, out_channels=1)
        
        
    def forward(self, x):
        x0 = self.encoder0(x[:, 0:1, :, :, :])
        x1 = self.encoder1(x[:, 1:2, :, :, :])
        x2 = self.encoder2(x[:, 2:3, :, :, :])
      
        # x0 shape (batch size, 32*4*4*4)
        x0 = torch.flatten(x0, start_dim=1)   
        x1 = torch.flatten(x1, start_dim=1)   
        x2 = torch.flatten(x2, start_dim=1) 
        # x shape (batch size, 32*4*4*4*3)
        x = torch.cat([x0, x1, x2], dim = -1)
 
        # (batch, 16*16*16)
        x_latent = self.to_lat(x) #dense layer
        x = nn.ReLU()(self.to_dec(x_latent)) # latent space

        x = x.view(-1, 64, self.freq, 8, 8)  # treat 31 as a sequence or depth
        
	    # shape (batch_size,64,8,8)
        output = self.decoder3d(x)
        return output

if __name__ == '__main__':
    batch_size = 32
    z = torch.randn((batch_size,3,64,64,64)) # treat 31 as a sequence or depth
    decoder = Net(64)
    output_imgs = decoder(z)
    print(output_imgs.shape)

