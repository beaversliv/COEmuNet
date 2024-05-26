import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# custom helper functions
from utils.dataloader     import CustomTransform,IntensityDataset
from utils.ResNet3DModel  import Net
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from utils.so3_model      import SO3Net
from utils.trainclass     import Trainer
from utils.data           import dataset_gen,postprocessing
from utils.loss           import SobelMse,Lossfunction,ResNetFeatures,MaxRel, calculate_ssim_batch
from utils.plot           import img_plt,history_plt
from utils.config         import faceon_args

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

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+config['exp_name']):
        os.makedirs('checkpoints/'+config['exp_name'])
    if not os.path.exists(f'checkpoints/{config['exp_name']}/img/'):
        os.makedirs(f'checkpoints/{config['exp_name']}/img/')
    os.system('cp main.py checkpoints'+'/'+config['exp_name']'/'+'main.py.backup')
def set_seed(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)

def main():
    config = faceon_args()
    # _init_()
    set_seed(args.seed)
    
    ### set dataset ###
    train_dataset,test_dataset = dataset_gen(config)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### set a model ###
    if args.model=='3dResNet':
        model = Net(args.model_grid).to(device)
    elif args.model == 'so3':
        model = ClsSO3Net().to(device)
    ### set loss function ###
    loss_object = SobelMse(device)
    ### set optimizer ###
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], betas=(0.9, 0.999))

    ### start training ###
    start = time.time()
    trainer = Trainer(model, loss_object, optimizer, train_dataloader, test_dataloader, config, device)
    tr_losses, vl_losses = trainer.run()
    end = time.time()
    print(f'running time:{(end-start)/60} mins')


    ### validation ###
    pred, target, test_loss = trainer.test()
    print('Test Epoch: {} Loss: {:.4f}\n'.format(
                args.epochs, test_loss))

    ### save history and preds ###            
    pickle_file_path = f'{args.exp_name}_history.pkl'
    try:
        with open(pickle_file_path, "wb") as pickle_file:
            pickle.dump({
                'history': {'train_loss': tr_losses, 'val_loss': vl_losses},
                'targets': target,
                'predictions': pred
            }, pickle_file)
        print(f"Data successfully saved to {pickle_file_path}")
    except Exception as e:
        print(f"Error saving data to pickle file: {e}")

    original_target = postProcessing(target)
    original_pred = postProcessing(pred)
    print(f'relative loss {MaxRel(original_target,original_pred):.5f}%')

    avg_ssim = calculate_ssim_batch(target,pred)
    for freq in range(len(avg_ssim)):
        print(f'frequency {freq + 1} has ssim {avg_ssim[freq]:.4f}')
    torch.save(model.state_dict(),f'{config['exp_name']}_model.pth')
    # img_plt(target,pred,path=f'/checkpoints/{args.exp_name}/img/')
    # history_plt(tr_losses,vl_losses,path=args.exp_name)

if __name__ == "__main__":
    main()