import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import pickle
import torch

class Logging:
    def __init__(self, file_dir:str, file_name:str):
        try:
            # Attempt to create the directory if it does not exist
            os.makedirs(file_dir, exist_ok=True)  # exist_ok=True avoids errors if directory already exists
        except Exception as e:
            print(f"Warning: Failed to create directory {file_dir}. {e}")
        # if not os.path.exists(file_dir):
        #     os.mkdir(file_dir)
        self.log_file = os.path.join(file_dir, file_name)
        open(self.log_file, 'w').close()

    def info(self, message, gpu_rank=0, console=True):
        # only log rank 0 GPU if running with multiple GPUs/multiple nodes.
        if gpu_rank is None or gpu_rank == 0:
            if console:
                print(message)

            with open(self.log_file, 'a') as f:  # a for append to the end of the file.
                print(message, file=f)

class HistoryShow:
    def __init__(self,log_files,pkl_file=None,single=True):
        self.log_files = log_files
        self.pkl_file = pkl_file
        self.single   = single

    def read_log_files(self,log_file):
        with open(log_file, 'r') as file:
            return file.read()
        
    def read_pkl_imgs(self):
        with open(self.pkl_file, 'rb') as fp:
            return pickle.load(fp)
            
    def img_pl(self,img_dir):
        if self.single:
            self.single_impl(img_dir)
        else:
            self.mul_impl(img_dir)


    def read_training_message(self):
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
        is_multiple_files = len(self.log_files) > 1
        for log_file in self.log_files:
            training_messages = self.read_log_files(log_file)
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
    def history_show(self,save_dir): 
        epochs,train_loss,val_loss,val_maxrel,ssim_transposed = self.read_training_message()
        fig,axes = plt.subplots(1,3, figsize=(20,5))
        axes[0].plot(epochs,train_loss, label='train')
        axes[0].plot(epochs,val_loss,label='val')
        axes[0].legend()
        axes[0].set_title('Training History')
        axes[0].set_xlabel('epoch')
        axes[0].set_ylabel('Feature loss + MSE loss')
        axes[0].grid(True)

        axes[1].plot(epochs, val_maxrel)
        axes[1].set_title('Val MaxRel History')
        axes[1].set_xlabel('epoch')
        axes[1].grid(True)
        for freq_idx, freq_ssim in enumerate(ssim_transposed):
            plt.plot(epochs, freq_ssim, label=f'Frequency {freq_idx}')
        axes[2].set_title('Val SSIM History')
        axes[2].set_xlabel('epoch')
        axes[2].grid(True)
        plt.savefig(f'{save_dir}/history.png')

    def single_impl(self,img_dir):
        check_dir(img_dir)
        data   = self.read_pkl_imgs()
        pred   = data['predictions']
        target = data['targets']
        for i in range(len(pred)):
            fig, axs = plt.subplots(1, 2,figsize=(12, 5))
            im1 = axs[0].imshow(target[i][0],vmin=np.min(target[i][0]),vmax = np.max(target[i][0]))
            axs[0].set_title('target')
            fig.colorbar(im1,ax=axs[0])

            im2 = axs[1].imshow(pred[i][0],vmin=np.min(target[i][0]),vmax = np.max(target[i][0]))
            axs[1].set_title('prediction')
            fig.colorbar(im2,ax=axs[1])
            plt.savefig(f'{img_dir}/ex_{i}.png')
            plt.close() 

    def mul_impl(self,img_dir):
        '''save_tar_pred_path: str, target_pred.png'''
        check_dir(img_dir)
        data = self.read_pkl_imgs()
        pred_sample  = data['predictions']
        target_sample = data['targets']
        # log space  
        for sample_idx in range(len(pred_sample)):  
            target = target_sample[sample_idx]
            pred   = pred_sample[sample_idx] #(7,64,64)
           
            rows = 2
            colms = pred.shape[0]
            fig,axes = plt.subplots(rows,colms,figsize=(16,6))
            for colm in range(colms):
                img1 = axes[0,colm].imshow(target[colm],vmin=np.min(target[colm]), vmax=np.max(target[colm]))
                axes[0,colm].set_title(f'freq {colm+13}')
                
                img2 = axes[1,colm].imshow(pred[colm],vmin=np.min(pred[colm]), vmax=np.max(pred[colm]))
                fig.colorbar(img1, ax=axes[0, colm], orientation='vertical', fraction=0.05, pad=0.04)
                fig.colorbar(img2, ax=axes[1, colm], orientation='vertical', fraction=0.05, pad=0.04)

            axes[0,0].set_ylabel('target')
            axes[1,0].set_ylabel('pred')
            plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce space between subplots
            plt.tight_layout()
            plt.savefig(f'{img_dir}/exp_{sample_idx}.png')
def show(img_data,path):
    data1 = img_data[0]
    data2 = img_data[1]
    data3 = img_data[2]
    # Create the subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Display images
    im1 = axes[0].imshow(data1)
    im2 = axes[1].imshow(data2)
    im3 = axes[2].imshow(data3)
    axes[0].set_ylabel('w FL')
    plt.subplots_adjust(wspace=0.1, hspace=0)
    # Set up the colorbar axes to span the combined width of the three images
    # The [left, bottom, width, height] should be adjusted based on your figure layout
    cbar_ax = fig.add_axes([axes[0].get_position().x0,  # left
                            0.90,                        # bottom
                            axes[-1].get_position().x1 - axes[0].get_position().x0,  # width
                            0.05])                       # height

    # Create the colorbar
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')

    # Place the colorbar ticks on top
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    # Set the colorbar label
    cbar.set_label('Intensity',va='center',fontstyle='normal',size='large')
    plt.savefig(path,bbox_inches='tight', pad_inches=0.1)
    # fig.text(0.5, 0.02, 'Shared X-axis Label', ha='center', va='center', fontsize=12)
    # fig.text(0.02, 0.5, 'Shared Y-axis Label', ha='center', va='center', rotation='vertical', fontsize=12)
    plt.show()

def load_txt(path):
    content = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            content.append(line)
    return content


def load_statistics(statistics_path,statistics_values):
        statistics = {}
        with h5.File(statistics_path, 'r') as f:
            for value in statistics_values:
                feature = value['name']
                stats_to_read = value['stats']
                statistics[feature] = {stat: f[feature][stat][()] for stat in stats_to_read}
        return statistics
def check_dir(file_dir):
    try:
        # Attempt to create the directory if it does not exist
        os.makedirs(file_dir, exist_ok=True)  # exist_ok=True avoids errors if directory already exists
    except Exception as e:
        print(f"Warning: Failed to create directory {file_dir}. {e}")

def load_state_dict(model, state_dict):
    """
    Load state_dict into model, handling both formats with and without 'module.' prefix.
    """
    # Create a new state dictionary to store the adjusted keys
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix if it exists
        if key.startswith('module.'):
            new_key = key[7:]  # Remove the first 7 characters ('module.')
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    # Load the adjusted state dictionary into the model
    model.load_state_dict(new_state_dict)

def load_encoder_pretrained(model, pretrained_path, map_location='cpu'):
    """
    Loads only the encoder part of the model from a pretrained checkpoint.
    Args:
        model (torch.nn.Module): The model instance with an encoder attribute.
        pretrained_path (str): Path to the pretrained checkpoint (e.g., 'pretrained.pth').
        map_location (str): The device to load the checkpoint on.
    """
    print(f"Loading encoder weights from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=map_location)
    model.encoder_state_dict = {k: v for k, v in checkpoint.items() if k.startswith('encoder')}
    print("Encoder weights loaded.")

def load_model_checkpoint(model,checkpoint_path=None,map_location='cpu'):
    """
    Load model and optionally optimizer state_dict from a checkpoint.
    Handles checkpoints with and without 'module.' prefix, and those missing optimizer states.
    
    Args:
        model (torch.nn.Module): The model instance.
        optimizer (torch.optim.Optimizer, optional): The optimizer instance. Default is None.
        checkpoint_path (str, optional): Path to the checkpoint file. Default is None.
        
    Returns:
        int: The epoch from which to resume training (or 0 if no checkpoint found).
        float: The best_loss saved in the checkpoint (or infinity if no checkpoint found).
    """
    # Check if checkpoint path is provided and if the file exists
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        print("No checkpoint found. Starting training from scratch.")
        return 0, float('inf')  # Start from scratch if no checkpoint is found
    
    # Load the checkpoint from the file
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    # Create a new state dictionary to store the adjusted keys (removing 'module.' if necessary)
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix if it exists (for loading from DataParallel models)
        if key.startswith('module.'):
            new_key = key[7:]  # Remove the first 7 characters ('module.')
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    # Load the adjusted state_dict into the model
    model.load_state_dict(new_state_dict)

    # Return other useful information from the checkpoint (epoch, best_loss)
    start_epoch = checkpoint.get('epoch', 0)  # Default to epoch 0 if not available
    best_loss = checkpoint.get('current_maxrel', float('inf'))  # Default to infinity if not available
    
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    return start_epoch, best_loss

def load_optimizer_checkpoint(optimizer, checkpoint_path=None, map_location='cpu'):
    """
    Load the optimizer state_dict from a checkpoint.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer instance.
        checkpoint_path (str, optional): Path to the checkpoint file. Default is None.
        map_location (str): Device to map the checkpoint to. Default is 'cpu'.
    """
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        print("No checkpoint found. Skipping optimizer loading.")
        return

    # Load the checkpoint from the file
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Load the optimizer state_dict if it exists
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state_dict loaded.")
    else:
        print("Optimizer state_dict not found in checkpoint. Skipping optimizer loading.")