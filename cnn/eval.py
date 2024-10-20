import torch
import torch.nn       as nn
from torch.utils.data import DataLoader
from torch.autograd   import Variable
import matplotlib.pyplot as plt

from utils.ResNet3DModel  import Net,Net3D
from utils.dataloader     import ChunkLoadingDataset
from utils.loss     import calculate_ssim_batch,MaxRel,FreqMse
from utils.config         import parse_args,load_config,merge_config

from utils.trainclass     import Trainer

import time
import numpy              as np
import h5py               as h5
import logging
from utils.ResNet3DModel  import Net3D,Net
from utils.utils          import HistoryShow,check_dir,Logging,load_model_checkpoint
import pickle
import statistics
import os
save_dir = '/home/dp332/dp332/dc-su2/results/mulfreq/'
check_dir(save_dir)
logger = Logging(save_dir, 'evaluation.txt')
def img_pl(img_data,path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    title = ['target','highest_MaxRel','lowest_MaxRel']
    for i in range(3):
        data = img_data[i]
        img = axes[i].imshow(data[0])
        axes[i].set_title(title[i])
        fig.colorbar(img)
        plt.savefig(path)

def MaxRel_sample_distribution(original_target,original_high_pred,original_low_pred,fig_path):
    '''the maxrel for each sample in histogram'''
    results = {'high': [], 'low': []}
    for i in range(len(original_target)):
        sample_high_maxrel = np.mean(np.abs(original_target[i]-original_high_pred[i]) / np.max(original_target[i], axis=0,keepdims=True)) * 100
        sample_low_maxrel = np.mean(np.abs(original_target[i]-original_low_pred[i]) / np.max(original_target[i], axis=0,keepdims=True))* 100
        results['high'].append(sample_high_maxrel)
        results['low'].append(sample_low_maxrel)

    logger.info(
        f"maxrel statistic values across samples for highest model,"
        f"max MaxRel {max(results['high'])},"
        f"min MaxRel {min(results['high'])},"
        f"mean MaxRel {statistics.mean(results['high'])},"
        f"median MaxRel {statistics.median(results['high'])},"
        )
    logger.info(
        f"maxrel statistic values across samples for lowest model,"
        f"max MaxRel {max(results['low'])},"
        f"min MaxRel {min(results['low'])},"
        f"mean MaxRel {statistics.mean(results['low'])},"
        f"median MaxRel {statistics.median(results['low'])},"
        )
    fig,axes = plt.subplots(1,2, figsize=(10,5))
    axes[0].hist(results['high'])
    axes[0].set_title('Highest model Maxrel sample distribution')

    axes[1].hist(results['low'])
    axes[1].set_title('Lowes model Maxrel sample distribution')
    plt.savefig(fig_path)
    
def locate_min_max_maxrel(save_dir,log_files):

    history_plot = HistoryShow(log_files,pkl_file=None,single=False)
    _,_,_,val_maxrel,ssim_transposed = history_plot.read_training_message()
    # history_plot.history_show(save_dir)
    val_maxrel = val_maxrel[:30]
    ssim = ssim_transposed[0][:30]
    highest_maxrel = max(val_maxrel)
    lowest_maxrel = min(val_maxrel)
    highest_ssim = max(ssim)
    lowest_ssim = min(ssim)

    logger.info(f'higest maxrel :{highest_maxrel:.4f}% and lowest maxrel {lowest_maxrel:.4f}% during training')
    logger.info(f'higest ssim :{highest_ssim:.4f} and lowest ssim {lowest_ssim:.4f} during training')

    highest_epoch = val_maxrel.index(highest_maxrel)
    lowest_epoch = val_maxrel.index(lowest_maxrel)
    # high_checkpoint = f'{save_dir}/model_{highest_epoch}.pth'
    # low_checkpoint = f'{save_dir}/model_{lowest_epoch}.pth'
    logger.info(f'the checkpiont for highest MaxRel appears at epoch {highest_epoch} and lowest at epoch {lowest_epoch}')

    # return high_checkpoint,low_checkpoint
def mulfreq_imshow(target,pred,save_path):
    '''
    input: single paried target and pred
    '''
    

    target = target.reshape(7,64,64)
    pred   = pred.reshape(7,64,64)

    rows = 2
    colms = 7
    fig,axes = plt.subplots(rows,colms,figsize=(16,6))
    for colm in range(colms):
        img1 = axes[0,colm].imshow(target[colm],vmin=np.min(target[colm]), vmax=np.max(target[colm]))
        axes[0,colm].set_title(rf"frequency $\nu_{{{colm + 13}}}$", fontsize=10)
        
        img2 = axes[1,colm].imshow(pred[colm],vmin=np.min(pred[colm]), vmax=np.max(pred[colm]))
        fig.colorbar(img1, ax=axes[0, colm], orientation='vertical', fraction=0.05, pad=0.04)
        fig.colorbar(img2, ax=axes[1, colm], orientation='vertical', fraction=0.05, pad=0.04)

    axes[0,0].set_ylabel('target')
    axes[1,0].set_ylabel('pred')
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce space between subplots
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path)
def save_pkl(model,loss_object,optimizer,train_dataloader,test_dataloader,config,device,save_dir,scheduler=None):
    trainer = Trainer(model,loss_object,optimizer,train_dataloader,test_dataloader,config,device,logger,scheduler=None)
    preds,targets,total_loss,freq_loss,mse = trainer.test()
    assert len(targets) == len(preds), "Targets and predictions must have the same length"
    
    original_targets,original_preds = trainer.postProcessing(targets,preds)
    original_targets = original_targets.cpu().numpy()
    original_preds   = original_preds.cpu().numpy()
    targets          = targets.cpu().numpy()
    preds          = preds.cpu().numpy()

    img_dir = os.path.join(save_dir,'img/')
    check_dir(img_dir)
    for sample_idx in range(20):
        target = targets[sample_idx]
        pred   = preds[sample_idx] #(7,64,64)
        save_path = os.path.join(img_dir,f'img_{sample_idx}.png')
        mulfreq_imshow(target,pred,save_path)

    # history_path = os.path.join(save_dir,'pickl.pkl')
    # if history_path is not None:
    #     with open(history_path, "wb") as pickle_file:
    #         pickle.dump({
    #             "predictions": pred[:20],
    #             "targets": target[:20],
    #             "or_targets": original_targets[:20],
    #             "or_preds": original_preds[:20]
    #         }, pickle_file)
    #     logger.info(f'saved {history_path}')
    return preds,targets,original_targets,original_preds

def single_model_eval_main():
    args = parse_args()
    config = load_config(args.config)
    config = merge_config(args, config)

    train_file_list_path= f"dataset/data/{config['dataset']['name']}/train.txt"
    test_file_list_path = f"dataset/data/{config['dataset']['name']}/test.txt"
    train_dataset = ChunkLoadingDataset(train_file_list_path,config['dataset']['batch_size'])
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,num_workers=4,pin_memory=True,prefetch_factor=2)
    test_dataset = ChunkLoadingDataset(test_file_list_path,config['dataset']['batch_size'])
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = '/home/dp332/dp332/dc-su2/results/mulfreq/ssim/best_sofar_ssim/'
    if config['dataset']['name'] == 'mulfreq':
        model = Net3D(7).to(device)
    else:
        model = Net(64).to(device)
    full_model_checkpoint_path = '/home/dp332/dp332/dc-su2/results/mulfreq/ssim/best_sofar_ssim.pth'
    start_epoch, best_loss = load_model_checkpoint(model, full_model_checkpoint_path, map_location=device)
    

    optimizer_params = config['optimizer']['params']
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    loss_object = FreqMse(alpha=config['model']['alpha'],beta=config['model']['beta'])

    pred,target,original_target,original_pred = save_pkl(model,loss_object,optimizer,train_dataloader,test_dataloader,config,device,save_dir,scheduler=None)
    
def comparison_main():
    args = parse_args()
    config = load_config(args.config)
    config = merge_config(args, config)

    train_file_paths = [f'/home/dp332/dp332/dc-su2/dc-su2/Rotation/train_{i}.hdf5' for i in range(1)]
    test_file_paths = [f'/home/dp332/dp332/dc-su2/dc-su2/Rotation/test_{i}.hdf5' for i in range(1)]
    train_dataset = ChunkLoadingDataset(train_file_paths,config['dataset']['batch_size'],config['dataset']['name'])
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,num_workers=4,pin_memory=True,prefetch_factor=2)
    test_dataset = ChunkLoadingDataset(test_file_paths,config['dataset']['batch_size'],config['dataset']['name'])
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = '/home/dp332/dp332/dc-su2/results/rotation/full_dataset'
    log_files = [f'{save_dir}/full_dataset.txt']
    img_dir   = f'{save_dir}/img'
    check_dir(img_dir)

    high_checkpoint,low_checkpoint = locate_min_max_maxrel(save_dir,log_files)
    
    high_model = Net(64).to(device)
    low_model  = Net(64).to(device)
    high_model = load_checkpoints(high_model,device,high_checkpoint)
    low_model  = load_checkpoints(low_model,device,low_checkpoint)

    optimizer_params = config['optimizer']['params']
    optimizer = torch.optim.Adam(high_model.parameters(), **optimizer_params)
    loss_object = FreqMse(alpha=config['model']['alpha'],beta=config['model']['beta'])
    high_history_paths = f'{save_dir}/high.pkl'
    low_history_paths = f'{save_dir}/low.pkl'
    high_pred,high_target,original_high_target,original_high_pred = save_pkl(high_model,loss_object,optimizer,train_dataloader,test_dataloader,config,device,high_history_paths,scheduler=None)
    low_pred,low_target,original_low_target,original_low_pred = save_pkl(low_model,loss_object,optimizer,train_dataloader,test_dataloader,config,device,low_history_paths,scheduler=None)
    
    for i in range(30):
        img_data = [high_target[i],high_pred[i],low_pred[i]]
        path     = f'{img_dir}/high_low_{i}.png'
        img_pl(img_data,path)
    MaxRel_sample_distribution(original_high_target,original_high_pred,original_low_pred,fig_path=f'{img_dir}/maxrel_distribution.png')

if __name__ == '__main__':
    single_model_eval_main()
    