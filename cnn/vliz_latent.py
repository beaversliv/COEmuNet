import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset,random_split
from utils.dataloader     import PreProcessingTransform,AsyncChunkDataset,AddGaussianNoise1,CustomCompose
from utils.config         import parse_args,load_config,merge_config
from torch.autograd import Variable
from utils.ResNet3DModel  import Encoder,Decoder
import h5py     as h5
import numpy    as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
class Latent(nn.Module):
    def __init__(self,input_dim,model_grid=64):
        super(Latent,self).__init__()
        self.layers =  nn.ModuleList()
        if model_grid == 32:
            out_dim = 64 * 4 * 4
        elif model_grid == 64:
            out_dim = 64 * 8 * 8
        elif model_grid == 128:
            out_dim = 64 * 16 * 16

        self.layers.append(nn.Linear(input_dim, 16**3))
        self.layers.append(nn.Linear(16**3, out_dim))
        self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        noise = torch.randn_like(x) * 0.1 + 0.0
        x = x + noise
        return x
class FinetuneNet(nn.Module):
    def __init__(self,model_grid=64):
        super(FinetuneNet, self).__init__()
        self.model_grid = model_grid
        self.encoder0 = Encoder(1)
        self.encoder1 = Encoder(1)
        self.encoder2 = Encoder(1)
        # grid 64
        if model_grid == 32:
            input_dim = 32*2*2*2*3
        elif model_grid == 64:
            input_dim = 32*4*4*4*3
        elif model_grid == 128:
            input_dim = 32*8*8*8*3
        self.latent = Latent(input_dim=input_dim)
        self.decoder= Decoder(in_channels=64, out_channels=1)
    def forward(self,x):
        x0 = self.encoder0(x[:, 0:1, :, :, :])
        x1 = self.encoder1(x[:, 1:2, :, :, :])
        x2 = self.encoder2(x[:, 2:3, :, :, :])
      
        # x0 shape (batch size, 32*4*4*4)
        x0 = torch.flatten(x0, start_dim=1)   
        x1 = torch.flatten(x1, start_dim=1)   
        x2 = torch.flatten(x2, start_dim=1) 
        # x shape (batch size, 32*4*4*4*3)
        features = torch.cat([x0, x1, x2], dim = -1)
        latent_output = self.latent(features)

        if self.model_grid == 32:
            x = latent_output.view(-1, 64, 4, 4)
        elif self.model_grid == 64:
            x = latent_output.view(-1, 64, 8, 8)
        elif self.model_grid == 128:
            x = latent_output.view(-1, 64, 16, 16)
        output = self.decoder(x)
        return latent_output,output

def main():
    args = parse_args()
    config = load_config(args.config)
    config = merge_config(args, config)    
    ### data pre-processing ###
    transform = PreProcessingTransform(config['dataset']['statistics']['path'])
    dataset = AsyncChunkDataset(['/home/dc-su2/rds/rds-dirac-dr004/Magritte/random_grid64_data.hdf5'],transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = int(0.2 * len(dataset))
    print(test_size)
    
    val_size = len(dataset) - train_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size= config['dataset']['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinetuneNet(64).to(device)
    model_dict = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/results/Finetune_scratch.pth'
    state_dict = torch.load(model_dict,map_location=device)
    # Create a new state dictionary without the "module." prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value  # Remove 'module.' prefix
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
    model.eval()
    all_latent_spaces = []
    # Pass your data through the encoder
    with torch.no_grad():
        for bidx, samples in enumerate(test_dataloader):
            data, target = Variable(samples[0]).to(device), Variable(samples[1]).to(device)
            latent_space,pred = model(data)
            all_latent_spaces.append(latent_space.cpu().numpy())

    all_latent_spaces = np.concatenate(all_latent_spaces, axis=0)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=0)
    tsne_result = tsne.fit_transform(all_latent_spaces)

    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)

    plt.title("t-SNE Visualization")
    plt.savefig('/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/images/0.1noise_randomall_latent_space.png')
    plt.show()
if __name__ == '__main__':
    main()
