import torch
import torch.nn as nn
import torchvision.models as models
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
        self.layers.append(residual_Block(8, 8, kernel_size=3, stride=1, padding = 'same', skip=False))
        
        self.layers.append(residual_Block(8, 16, kernel_size=3, stride=2, padding = 1, skip=True))
        self.layers.append(residual_Block(16, 16, kernel_size=3, stride=1, padding = 'same', skip=False))
        self.layers.append(residual_Block(16, 16, kernel_size=3, stride=1, padding = 'same', skip=False))
        self.layers.append(residual_Block(16, 16, kernel_size=3, stride=1, padding = 'same', skip=False))        

        self.layers.append(residual_Block(16, 32, kernel_size=3, stride=2, padding = 1, skip=True))
        self.layers.append(residual_Block(32, 32, kernel_size=3, stride=1, padding = 'same', skip=False))
        self.layers.append(residual_Block(32, 32, kernel_size=3, stride=1, padding = 'same', skip=False))
        self.layers.append(residual_Block(32, 32, kernel_size=3, stride=1, padding = 'same', skip=False))

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
    def __init__(self):
        super(Net, self).__init__()
        self.encoder0 = Encoder(1)
        self.encoder1 = Encoder(1)
        self.encoder2 = Encoder(1)
        
        # self.to_lat = nn.Linear(32*4*4*4*3, 8*8*8)
        # self.to_dec = nn.Linear(8*8*8, 64*8*8) #64*8*8
        self.to_lat = nn.Linear(32*4*4*4*3,16*16*16)
        self.to_dec = nn.Linear(16*16*16,64*8*8)
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
        x_latent = self.to_lat(x) #dense layer
        x = nn.ReLU()(self.to_dec(x_latent)) # latent space
        x = x.view(-1, 64, 8, 8)
      
	# shape (batch_size,64,8,8)
        output = self.decoder(x)
        return x_latent, output

class VGGFeatures(torch.nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg_model = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_model[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_model[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_model[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_model[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3