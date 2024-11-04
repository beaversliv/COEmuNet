from ruamel.yaml import YAML
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import re
import torch
from collections import OrderedDict
import socket 
import torch.distributed      as dist
import h5py as h5
import datetime
from utils.loss import single_maxrel
def check_file(file_path):
    return os.path.isfile(file_path)

def check_dir(file_dir):
    try:
        # Attempt to create the directory if it does not exist
        os.makedirs(file_dir, exist_ok=True)  # exist_ok=True avoids errors if directory already exists
    except Exception as e:
        print(f"Warning: Failed to create directory {file_dir}. {e}")

def load_yaml(path):
    with open(path, 'rb') as f:
        yaml = YAML()
        dt = yaml.load(f)
        return dt
def load_txt(path):
    file_list = []
    with open(path,'r') as lines:
        for line in lines:
            file_list.append(line.strip())
    return file_list

def save_pickle(store_dir, file):
    with open(store_dir, 'wb') as fo:
        pickle.dump(file, fo)

def load_pickle(file, encoding="bytes"):
    with open(file, "rb") as fo:
        return pickle.load(fo, encoding=encoding)

def load_log_files(log_file):
        with open(log_file, 'r') as file:
            return file.read()
        
def load_statistics(statistics_path,statistics_values):
    statistics = {}
    with h5.File(statistics_path, 'r') as f:
        for value in statistics_values:
            feature = value['name']
            stats_to_read = value['stats']
            statistics[feature] = {stat: f[feature][stat][()] for stat in stats_to_read}
    return statistics

def setup(rank, world_size):
    "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"

    def get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    # Ensure MASTER_ADDR is set
    if "MASTER_ADDR" not in os.environ:
        raise ValueError("MASTER_ADDR environment variable is not set")
    
    # Set MASTER_PORT to a free port
    if "MASTER_PORT" not in os.environ:
        free_port = get_free_port()
        os.environ["MASTER_PORT"] = str(free_port)
    
    # Ensure MASTER_PORT is set
    if "MASTER_PORT" not in os.environ or os.environ["MASTER_PORT"] is None:
        raise ValueError("MASTER_PORT environment variable is not set")

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size,timeout=datetime.timedelta(seconds=2400))

def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()

class LoadCheckPoint():
    def __init__(self,learning_model,optimizer,file_path,stage,logger,local_rank,ddp_on=True):
        self.learning_model = learning_model
        self.optimizer      = optimizer
        self.file_path      = file_path
        self.stage          = stage
        self.logger         = logger
        self.local_rank     = local_rank
        self.ddp_on         = ddp_on

    def get_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if self.ddp_on:
                if not k.startswith('module.'):
                    new_state_dict['module.' + k] = v  # Add 'module.' if missing (for DDP)
                else:
                    new_state_dict[k] = v
            else:
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # Remove 'module.' if present (for non-DDP)
                else:
                    new_state_dict[k] = v
        return new_state_dict
    
    def load_state(self,state):
        state = self.get_state_dict(state)
        try:
            if self.stage==1:
                self.logger.info(f"* only load encoder")
                self.learning_model.encoder_state_dict = {k: v for k, v in state.items() if k.startswith('encoder')}
            else:
                self.learning_model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            self.logger.info(e)
            self.learning_model.load_state_dict(state, strict=False)

    def load_checkpoint(self, model_only=True):
        if os.path.isfile(self.file_path):
            self.logger.info(f"=> loading checkpoint '{self.file_path}'")
            try:
                checkpoint = torch.load(self.file_path, map_location=self.local_rank, weights_only=True)
            except RuntimeError as e:
                self.logger.info(e)
                with open(self.file_path, 'rb') as f:
                    checkpoint = pickle.load(f, weights_only=True)
            # set strict to True for omitting key missmatch in the checkpoint and the current model
            if 'model_state_dict' not in checkpoint: # only happens in pre_trained model,stage 1
                self.load_state(checkpoint)
                self.logger.info(f"=> loaded checkpoint '{self.file_path}'")
            else:
                self.load_state(checkpoint['model_state_dict'])
                if model_only:
                    self.logger.info(f"=> loaded checkpoint '{self.file_path}'")
                    return
                if self.stage == 2:
                    self.start_epoch = checkpoint['epoch'] + 1
                    if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        self.logger.info(f'=> loaded optimizer')
                self.logger.info(f"=> loaded checkpoint '{self.file_path}' (trained {checkpoint['epoch']}+1 epochs)")
        else:
            info = f"=> no checkpoint found at '{self.file_path}'"
            self.logger.info(info)
            raise FileNotFoundError(info)


class HistoryShow:
    def __init__(self,save_dir,single=True):
  
        self.save_dir = save_dir
        self.single   = single

    def img_plt(self,pkl_file):
        img_folder = os.path.join(self.save_dir,'img/')
        check_dir(img_folder)

        data   = load_pickle(pkl_file)
        pred   = data['or_preds']
        target = data['or_targets']
        maxrel_0 = single_maxrel(target, pred, centre_size=0)
        maxrel_2 = single_maxrel(target, pred, centre_size=2)
        maxrel_4 = single_maxrel(target, pred, centre_size=4)
        maxrel_6 = single_maxrel(target, pred, centre_size=6)

        for sample_idx in range(len(target)):
            save_path = os.path.join(img_folder,f'img_{sample_idx}.png')
            if self.single:
                self._sgl_imshow(target[sample_idx],pred[sample_idx],save_path)
            else:
                self._mulfreq_imshow(target[sample_idx],pred[sample_idx],maxrel_0[sample_idx],maxrel_2[sample_idx],maxrel_4[sample_idx],maxrel_6[sample_idx],save_path)

    def read_training_message(self,log_files):
        total_epochs = 0
        epochs = []
        train_loss = []
        train_feature = []
        train_mse = []
        val_loss = []
        val_feature = []
        val_mse = []
        val_maxrel = []
        val_ssim = []
        is_multiple_files = len(log_files) > 1
        for log_file in log_files:
            training_messages = load_log_files(log_file)
            if self.single:
                pattern = re.compile(
                r"epoch:\s*(\d+),\s*"
                r"train_loss:\s*([\d\.e\-]+),\s*"
                r"train_feature:\s*([\d\.e\-]+),\s*"
                r"train_mse:\s*([\d\.e\-]+),\s*"
                r"train_maxrel:\s*([\d\.e\-]+),\s*"
                r"train_ssim:\s*\[([\d\.,\s\-e]+)\],\s*"
                r"val_loss:\s*([\d\.e\-]+),\s*"
                r"val_feature:\s*([\d\.e\-]+),\s*"
                r"val_mse:\s*([\d\.e\-]+),\s*"
                r"val_maxrel:\s*([\d\.e\-]+),\s*"
                r"val_ssim:\s*\[([\d\.,\s\-e]+)\]"
                )
                for match in pattern.finditer(training_messages):
                    epoch_in_file = int(match.group(1))
                    if is_multiple_files:
                        cumulative_epoch = epoch_in_file + total_epochs
                    else:
                        cumulative_epoch = epoch_in_file
                    # Adjust epoch to be cumulative
                    epochs.append(cumulative_epoch)

                    train_loss.append(float(match.group(2)))
                    train_feature.append(float(match.group(3)))
                    train_mse.append(float(match.group(4)))
                    val_loss.append(float(match.group(7)))
                    val_feature.append(float(match.group(8)))
                    val_mse.append(float(match.group(9)))
                    val_maxrel.append(float(match.group(10)))
                    # Parse the nested SSIM values
                    ssim_values = match.group(11)
                    ssim_list = [float(ssim) for ssim in ssim_values.split(', ')]
                    val_ssim.append(ssim_list)
                if is_multiple_files:
                    total_epochs = epochs[-1] + 1

            else:
                pattern = re.compile(
                r"epoch:\s*(\d+),\s*"
                r"train_loss:\s*([\d\.e\-]+),\s*"
                r"train_feature:\s*([\d\.e\-]+),\s*"
                r"train_mse:\s*([\d\.e\-]+),\s*"
                r"val_loss:\s*([\d\.e\-]+),\s*"
                r"val_feature:\s*([\d\.e\-]+),\s*"
                r"val_mse:\s*([\d\.e\-]+),\s*"
                r"val_maxrel:\s*([\d\.e\-]+),\s*"
                r"val_ssim:\s*\[(.*?)\]"
                )
                for match in pattern.finditer(training_messages):
                    epoch_in_file = int(match.group(1))
                    if is_multiple_files:
                        cumulative_epoch = epoch_in_file + total_epochs
                    else:
                        cumulative_epoch = epoch_in_file
                    # Adjust epoch to be cumulative
                    epochs.append(cumulative_epoch)

                    train_loss.append(float(match.group(2)))
                    train_feature.append(float(match.group(3)))
                    train_mse.append(float(match.group(4)))
                    val_loss.append(float(match.group(5)))
                    val_feature.append(float(match.group(6)))
                    val_mse.append(float(match.group(7)))
                    val_maxrel.append(float(match.group(8)))
                    # Parse the nested SSIM values
                    ssim_values = match.group(9)
                    ssim_list = [float(ssim) for ssim in ssim_values.split(', ')]
                    val_ssim.append(ssim_list)
                if is_multiple_files:
                    total_epochs = epochs[-1] + 1
        ssim_transposed = list(map(list, zip(*val_ssim)))
        
        return epochs,train_loss,val_loss,val_maxrel,ssim_transposed
    def history_show(self,log_files): 
        history_save_path = f'{self.save_dir}/history.png'
        epochs,train_loss,val_loss,val_maxrel,ssim_transposed = self.read_training_message(log_files)

        fig,axes = plt.subplots(1,3, figsize=(20,5))
        axes[0].plot(epochs,train_loss, label='train')
        axes[0].plot(epochs,val_loss,label='val')
        axes[0].legend()
        axes[0].set_title('Training History')
        axes[0].set_xlabel('epoch')
        axes[0].set_ylabel('Feature loss + MSE loss')
        axes[0].grid(True)

        axes[1].plot(epochs, val_maxrel)
        axes[1].set_ylim(0,100)
        axes[1].set_title('Val MaxRel History')
        axes[1].set_xlabel('epoch')
        axes[1].grid(True)
        num_freqs = 7
        for freq_idx, freq_ssim in enumerate(ssim_transposed):
            plt.plot(epochs, freq_ssim, label=f'Frequency {freq_idx}')
        axes[2].set_title('Val SSIM History')
        axes[2].set_xlabel('epoch')
        axes[2].grid(True)
        plt.savefig(history_save_path)

    def _sgl_imshow(self,target,pred,save_path):
        '''
        single paired target and pred in original space
        '''
        target = target.reshape(1,64,64)
        pred   = pred.reshape(1,64,64)
        maxrel = np.abs(target-pred) / np.max(target, axis=0,keepdims=True) * 100
        maxrel = maxrel[0]
        lg_target = np.log(target[0])
        lg_pred   = np.log(pred[0])
        plt.rc('axes', titlesize=12)  # Title font size
        plt.rc('axes', labelsize=10)  # Axis label font size
        plt.rc('xtick', labelsize=8)  # X-tick font size
        plt.rc('ytick', labelsize=8)  # Y-tick font size


        fig, axs = plt.subplots(3,1,figsize=(5,8))
        im1 = axs[0].imshow(lg_target,vmin=np.min(lg_target), vmax=np.max(lg_target))
        im2 = axs[1].imshow(lg_pred,vmin=np.min(lg_target), vmax=np.max(lg_target))
        im3 = axs[2].imshow(maxrel,vmin=np.min(maxrel), vmax=np.max(maxrel))

        axs[2].text(
                0.5, -0.15,  # Adjust position
                f'min: {np.min(maxrel):.2f}, max: {np.max(maxrel):.2f}',  # Format min/max
                color='black', fontsize=10, ha='center', va='top',  # Center the text
                transform=axs[2].transAxes  # Set relative to the axis
            )
        fig.colorbar(im1,ax=axs[0],orientation='vertical', fraction=0.05, pad=0.1)
        fig.colorbar(im2,ax=axs[1],orientation='vertical', fraction=0.05, pad=0.1)
        fig.colorbar(im3,ax=axs[2],orientation='vertical', fraction=0.05, pad=0.1)

        axs[0].set_ylabel('target')
        axs[1].set_ylabel('pred')
        axs[2].set_ylabel('maxrel(%)')
        plt.savefig(save_path,bbox_inches='tight')
        plt.close() 

    def _mulfreq_imshow(self,target,pred,maxrel_0,maxrel_2,maxrel_4,maxrel_6,save_path):
        '''
        input: single paried target and pred, in original space
        '''

        target = target.reshape(7,64,64)
        pred   = pred.reshape(7,64,64)


        lg_target = np.log(target)
        lg_pred   = np.log(pred)

        plt.rc('axes', titlesize=16)  # Title font size
        plt.rc('axes', labelsize=15)  # Axis label font size
        plt.rc('xtick', labelsize=12)  # X-tick font size
        plt.rc('ytick', labelsize=12)  # Y-tick font size

        rows = 6
        colms = 7
        fig,axes = plt.subplots(rows,colms,figsize=(34,18))
        for colm in range(colms):
            img1 = axes[0,colm].imshow(lg_target[colm],vmin=np.min(lg_target[colm]), vmax=np.max(lg_target[colm]))
            axes[0,colm].set_title(rf"frequency $\nu_{{{colm + 13}}}$", fontsize=16)
            
            img2 = axes[1,colm].imshow(lg_pred[colm],vmin=np.min(lg_target[colm]), vmax=np.max(lg_target[colm]))
            img3 = axes[2,colm].imshow(maxrel_0[colm],vmin=np.min(maxrel_0[colm]), vmax=np.max(maxrel_0[colm]))
            img4 = axes[3,colm].imshow(maxrel_2[colm],vmin=np.min(maxrel_2[colm]), vmax=np.max(maxrel_2[colm]))
            img5 = axes[4,colm].imshow(maxrel_4[colm],vmin=np.min(maxrel_4[colm]), vmax=np.max(maxrel_4[colm]))
            img6 = axes[5,colm].imshow(maxrel_6[colm],vmin=np.min(maxrel_6[colm]), vmax=np.max(maxrel_6[colm]))

            # axes[2, colm].text(
            #     0.5, 1.05,  # Adjust position
            #     f'min: {np.min(maxrel_0[colm]):.2f}, max: {np.max(maxrel_0[colm]):.2f}',  # Format min/max
            #     color='black', fontsize=10, ha='center', va='bottom',  # Center the text
            #     transform=axes[2, colm].transAxes  # Set relative to the axis
            # )
            # axes[3, colm].text(
            #     0.5, 1.05,  # Adjust position
            #     f'min: {np.min(maxrel_0[colm]):.2f}, max: {np.max(maxrel_0[colm]):.2f}',  # Format min/max
            #     color='black', fontsize=10, ha='center', va='bottom',  # Center the text
            #     transform=axes[2, colm].transAxes  # Set relative to the axis
            # )
            # axes[4, colm].text(
            #     0.5, 1.05,  # Adjust position
            #     f'min: {np.min(maxrel_0[colm]):.2f}, max: {np.max(maxrel_0[colm]):.2f}',  # Format min/max
            #     color='black', fontsize=10, ha='center', va='bottom',  # Center the text
            #     transform=axes[2, colm].transAxes  # Set relative to the axis
            # )
            fig.colorbar(img1, ax=axes[0, colm], orientation='vertical', fraction=0.05, pad=0.04)
            fig.colorbar(img2, ax=axes[1, colm], orientation='vertical', fraction=0.05, pad=0.04)
            fig.colorbar(img3, ax=axes[2, colm], orientation='vertical', fraction=0.05, pad=0.04)
            fig.colorbar(img4, ax=axes[3, colm], orientation='vertical', fraction=0.05, pad=0.04)
            fig.colorbar(img5, ax=axes[4, colm], orientation='vertical', fraction=0.05, pad=0.04)
            fig.colorbar(img6, ax=axes[5, colm], orientation='vertical', fraction=0.05, pad=0.04)

        axes[0,0].set_ylabel('target')
        axes[1,0].set_ylabel('pred')
        axes[2,0].set_ylabel('maxrel(%)')
        axes[3,0].set_ylabel('maxrel\nexclude centre 2x2 (%)')
        axes[4,0].set_ylabel('maxrel\nexlude centre 4x4 (%)')
        axes[5,0].set_ylabel('maxrel\nexclude cetre 6x6 (%)')
        # plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce space between subplots
        plt.tight_layout(pad=0.1)
        plt.savefig(save_path)
        plt.close()
                
class Logging:
    def __init__(self, file_dir:str, file_name:str):
        try:
            # Attempt to create the directory if it does not exist
            os.makedirs(file_dir, exist_ok=True)  # exist_ok=True avoids errors if directory already exists
        except Exception as e:
            print(f"Warning: Failed to create directory {file_dir}. {e}")
        self.log_file = os.path.join(file_dir, file_name)
        open(self.log_file, 'a').close()

    def info(self, message, gpu_rank=0, console=True):
        # only log rank 0 GPU if running with multiple GPUs/multiple nodes.
        if gpu_rank is None or gpu_rank == 0:
            if console:
                print(message)

            with open(self.log_file, 'a') as f:  # a for append to the end of the file.
                print(message, file=f)
if __name__=='__main__':
    
    pickle_path = '/home/dp332/dp332/dc-su2/results/mulfreq/ro_200_2/pickl.pkl'
    save_dir    = '/home/dp332/dp332/dc-su2/results/mulfreq/ro_200_2/'
    history_plot = HistoryShow(save_dir,single=False)
    history_plot.img_plt(pickle_path)