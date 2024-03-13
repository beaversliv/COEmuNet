



# custom helper functions
from utils.dataloader     import CustomTransform,IntensityDataset
from utils.ResNet3DModel  import Net
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from utils.so3_model      import SO3Net
from utils.trainclass     import Trainer
from data                 import load_face_on_view_grid64, load_random_view_grid64_12000, load_face_on_view_grid32, load_random_view_grid32
from utils.loss           import SobelMse,Lossfunction,ResNetFeatures,mean_absolute_percentage_error, calculate_ssim_batch
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
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists(f'checkpoints/{args.exp_name}/img/'):
        os.makedirs(f'checkpoints/{args.exp_name}/img/')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)

def main():
    parser = faceon_args()
    args = parser.parse_args()

    _init_()
    set_seed(args.seed)
    
    ### set dataset ###
    if args.model_grid == 32:
        if args.dataset == 'faceon':
            train_dataset, test_dataset = load_face_on_view_grid32()
        elif args.dataset == 'random':
            train_dataset, test_dataset = load_random_view_grid32()
    if args.model_grid == 64:
        if args.dataset == 'faceon':
            train_dataset, test_dataset = load_face_on_view_grid64()
        elif args.dataset == 'random':
            train_dataset, test_dataset = load_random_view_grid64()
    # if args.model_grid == 128:
    #     if args.dataset == 'faceon':
    #         train_dataset, test_dataset = load_face_on_view_grid64()
    #     elif args.dataset == 'random':
    #         train_dataset, test_dataset = load_random_view_grid64()
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=2)
    vali_dataloader = DataLoader(vali_dataset, batch_size=args.batch_size, shuffle=True,num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### set a model ###
    if args.model=='3dResNet':
        model = Net().to(device)
    elif args.model == 'so3':
        model = ClsSO3Net().to(device)
    ### set loss function ###
    loss_object = SobelMse(device)
    ### set optimizer ###
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas=(0.9, 0.999))

    ### start training ###
    start = time.time()
    trainer = Trainer(model, loss_object, optimizer, train_dataloader, test_dataloader, args, device)
    tr_losses, vl_losses = trainer.run()
    end = time.time()
    print(f'running time:{(end-start)/60} mins')


    ### validation ###
    pred, target, test_loss = trainer.test()
    print('Test Epoch: {} Loss: {:.4f}\n'.format(
                args.epochs, test_loss))
    data = (tr_losses, vl_losses,pred, target)

    mean_error, median_error = mean_absolute_percentage_error(target,pred)
    print('mean relative error: {:.4f}\n, median relative error: {:.4f}'.format(mean_error,median_error))
    avg_ssim = calculate_ssim_batch(target,pred)
    print('SSIM: {:.4f}'.format(avg_ssim))

    # plot and save history
    img_plt(target,pred,path=f'/checkpoints/{args.exp_name}/img/')
    history_plt(tr_losses,vl_losses,path=args.exp_name)
    with open(f'{args.exp_name}_history.pkl', "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    torch.save(model.state_dict(),f'{args.exp_name}_model.pth')

if __name__ == "__main__":
    main()