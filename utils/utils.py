# from ruamel.yaml import YAML
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import re
import torch
from collections  import OrderedDict
import socket 
import torch.distributed      as dist
import h5py as h5
import json
import datetime
from utils.loss import SingleMaxRel
def check_file(file_path):
    return os.path.isfile(file_path)

def check_dir(file_dir):
    try:
        # Attempt to create the directory if it does not exist
        os.makedirs(file_dir, exist_ok=True)  # exist_ok=True avoids errors if directory already exists
    except Exception as e:
        print(f"Warning: Failed to create directory {file_dir}. {e}")

# def load_yaml(path):
#     with open(path, 'rb') as f:
#         yaml = YAML()
#         dt = yaml.load(f)
#         return dt
def load_txt(path):
    file_list = []
    with open(path,'r') as lines:
        for line in lines:
            file_list.append(line.strip())
    return file_list
def load_json(json_file):
    with open(json_file,'r') as file:
        list_ = json.load(file)
    return list_

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
    def __init__(self,learning_model,optimizer,scheduler,file_path,stage,logger,local_rank,ddp_on=True):
        self.learning_model = learning_model
        self.optimizer      = optimizer
        self.scheduler      = scheduler
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
                # self.logger.info(f"load pre-trained autoencoder but ingore the last layer")
                # new_state_dict1 = self.learning_model.state_dict()

                # # Update the state dict while ignoring the mismatch in the last layer
                # for key in state:
                #     if key in new_state_dict1 and state[key].shape == new_state_dict1[key].shape:
                #         new_state_dict1[key] = state[key]

                # # Load the updated state dict into the new model
                # self.learning_model.load_state_dict(new_state_dict1)
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

                    if self.scheduler is not None:
                        if 'scheduler_state_dict' in checkpoint:
                            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                            self.logger.info("=> Scheduler state loaded from checkpoint.")
                        else:
                            self.logger.info("=> No scheduler state found in checkpoint. Scheduler initialized from scratch.")
                self.logger.info(f"=> loaded checkpoint '{self.file_path}' (trained {checkpoint['epoch']}+1 epochs)")
        else:
            info = f"=> no checkpoint found at '{self.file_path}'"
            self.logger.info(info)
            raise FileNotFoundError(info)


class HistoryShow:
    def __init__(self,save_dir):
  
        self.save_dir = save_dir

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
        val_zncc = []
        val_ssim = []
        is_multiple_files = len(log_files) > 1
        for log_file in log_files:
            training_messages = load_log_files(log_file)
            
            pattern = re.compile(
                r"epoch:\s*(\d+),\s*?"
                r"train_loss:\s*?([\d\.e\-]+),\s*?"
                r"train_feature:\s*?([\d\.e\-]+),\s*?"
                r"train_mse:\s*?([\d\.e\-]+),\s*?"
                r"val_loss:\s*?([\d\.e\-]+),\s*?"
                r"val_feature:\s*?([\d\.e\-]+),\s*?"
                r"val_mse:\s*?([\d\.e\-]+),\s*?"
                r"val_maxrel:\s*?([\d\.e\-]+),\s*?"
                r"val_zncc:\s*?([\d\.e\-]+),\s*?"
                r"val_ssim:\s*?([\d\.e\-]+)"
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
                val_maxrel.append(float(match.group(8))/100)
                val_zncc.append(float(match.group(9)))
                # Parse the nested SSIM values
                val_ssim.append(float(match.group(10)))
                # ssim_values = match.group(10)
                # ssim_list = [float(ssim) for ssim in ssim_values.split(', ')]
                # val_ssim.append(ssim_list)
            if is_multiple_files:
                total_epochs = epochs[-1] + 1
                # ssim_transposed = list(map(list, zip(*val_ssim)))
        return epochs,train_loss,val_loss,val_maxrel,val_zncc,val_ssim
    def history_show(self,log_files): 
        history_save_path = f'{self.save_dir}/history.png'
        epochs,train_loss,val_loss,val_maxrel,val_zncc,val_ssim = self.read_training_message(log_files)
        fig,axes = plt.subplots(1,2, figsize=(8,4))

        axes[0].plot(epochs,train_loss,label='training loss',color='#1E88E5')
        axes[0].plot(epochs,val_loss,label='validation loss',color='#E53935')
        axes[0].legend(loc='upper right')
        axes[0].set_xlabel('Epoch',fontsize=14)
        axes[0].set_ylabel('Loss',fontsize=14)
        axes[0].grid(True)

        axes[1].plot(epochs,val_maxrel,label='validation MaxRel',color='#4CAF50')
        axes[1].plot(epochs,val_zncc,label='validation ZNCC',color='#B0B0B0')
        axes[1].plot(epochs,val_ssim,label='validation SSIM',color='#2596be')
        axes[1].legend()
        axes[1].set_xlabel('Epoch',fontsize=14)
        axes[1].set_ylabel('MaxRel/ZNCC/SSIM',fontsize=14)
        axes[1].grid(True)

        fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.3)
        plt.savefig(history_save_path)
                
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
def zoomin_imshow(target,pred):
    '''
    input: single paried target and pred, in original space
    '''
    target = target.reshape(7,64,64)
    pred   = pred.reshape(7,64,64)
    lg_target = np.log(target)
    lg_pred   = np.log(pred)
   

    plt.rc('axes', titlesize=13)  # Title font size
    plt.rc('axes', labelsize=13)  # Axis label font size
    plt.rc('xtick', labelsize=18)  # X-tick font size
    plt.rc('ytick', labelsize=18)  # Y-tick font size

    rows = 2
    colms = 7
    fig, axes = plt.subplots(rows, colms, figsize=(30, 8),dpi=100)

    for row in range(rows):
        for colm in range(colms):
            img = axes[row, colm].imshow(
                lg_target[colm][15:15+32, 18:18+32] if row == 0 else lg_pred[colm][15:15+32, 18:18+32]
            )

            # Remove unnecessary axes
            if row == 0 and colm == 0:
                axes[row, colm].set_xticks([])  # Keep y-axis but remove x-axis
            elif row == 0:
                axes[row, colm].axis("off")  # Remove both axes for first-row images except (0,0)
            elif row == 1 and colm != 0:
                axes[row, colm].set_yticks([])  # Remove y-axis for second-row images except (1,0)

            # Add colorbars only for the last column in each row
            if colm == colms - 1:
                divider = make_axes_locatable(axes[row, colm])
                cax = divider.append_axes("right", size="5%", pad=0.2)  # Reduce padding
                fig.colorbar(img, cax=cax, orientation="vertical")

    # Ensure no spacing between images
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.02, hspace=0.02)

    plt.show()

def mulfreq_imshow(target,pred,path):
    '''
    input: single paried target and pred, in original space
    '''
    target = target.reshape(7,64,64)
    pred   = pred.reshape(7,64,64)
    lg_target = np.log(target)
    lg_pred   = np.log(pred)
    maxrel = np.abs(target - pred) / np.max(target,axis=0,keepdims=True)*100
    # lg_target = lg_target[[0,3,6],...]
    # lg_pred = lg_pred[[0,3,6],...]
    plt.rc('axes', titlesize=13)  # Title font size
    plt.rc('axes', labelsize=13)  # Axis label font size
    plt.rc('xtick', labelsize=18)  # X-tick font size
    plt.rc('ytick', labelsize=18)  # Y-tick font size

    rows = 3
    colms = 7
    fig, axes = plt.subplots(rows, colms, figsize=(32,13),dpi=50)

    for row in range(rows):
        for colm in range(colms):
            if row == 0:
                img = axes[row, colm].imshow(lg_target[colm])
            elif row == 1:
                img = axes[row, colm].imshow(lg_pred[colm])
            else:
                img = axes[row, colm].imshow(maxrel[colm],cmap='coolwarm')

            # Remove unnecessary axes
            if row == 0 and colm == 0:
                axes[row, colm].set_xticks([])  # Keep y-axis but remove x-axis
            elif row == 0:
                axes[row, colm].axis("off")  # Remove both axes for first-row images except (0,0)
            elif row == 1 and colm == 0:
                axes[row, colm].set_xticks([])
            elif row == 1:
                axes[row, colm].axis("off")

            elif row == 2 and colm != 0:
                axes[row, colm].set_yticks([])  # Remove y-axis for second-row images except (1,0)

            # Add colorbars only for the last column in each row
            if colm == colms - 1:
                if row !=2:
                    divider = make_axes_locatable(axes[row, colm])
                    cax = divider.append_axes("right", size="5%", pad=0.08)  # Reduce padding
                    fig.colorbar(img, cax=cax, orientation="vertical")
                else:
                    divider = make_axes_locatable(axes[row, colm])
                    cax = divider.append_axes("right", size="5%", pad=0.08)  # Adjust size and padding
                    cbar = fig.colorbar(img, cax=cax, orientation="vertical")
                    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    # Ensure no spacing between images
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.02, hspace=0.02)

    plt.savefig(path)