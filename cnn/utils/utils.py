import os
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import pickle
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


class HistoryShow:
    def __init__(self,log_file,pkl_file=None):
        self.log_file = log_file
        self.pkl_file = pkl_file
    def read_training_messages(self):
        with open(self.log_file, 'r') as file:
            return file.read()
    def read_pkl_imgs(self):
        with open(self.pkl_file, 'rb') as fp:
            data = pickle.load(fp)
    def check_fir(self,file_dir):
        try:
            # Attempt to create the directory if it does not exist
            os.makedirs(file_dir, exist_ok=True)  # exist_ok=True avoids errors if directory already exists
        except Exception as e:
            print(f"Warning: Failed to create directory {file_dir}. {e}")

    def history_show(self,save_path): 
        training_messages = self.read_training_messages()
        epochs = []
        train_loss = []
        train_feature = []
        train_mse = []
        val_loss = []
        val_feature = []
        val_mse = []
        val_maxrel = []
        val_ssim = []

        pattern = re.compile(
            r"epoch:\s*(\d+),\s*train_loss:\s*([\d\.e\-]+),\s*train_feature:\s*([\d\.e\-]+),\s*train_mse:\s*([\d\.e\-]+),\s*val_loss:\s*([\d\.e\-]+),\s*val_feature:\s*([\d\.e\-]+),\s*val_mse:\s*([\d\.e\-]+),\s*val_maxrel:\s*([\d\.e\-]+),\s*val_ssim:\s*\[(.*?)\]"
        )
        # pattern = re.compile(
        #     r"epoch:\s*(\d+),\s*"
        #     r"train_loss:\s*([\d\.e\-]+),\s*"
        #     r"train_feature:\s*([\d\.e\-]+),\s*"
        #     r"train_mse:\s*([\d\.e\-]+),\s*"
        #     r"train_maxrel:\s*([\d\.e\-]+),\s*"
        #     r"train_ssim:\s*\[([\d\.,\s\-e]+)\],\s*"
        #     r"val_loss:\s*([\d\.e\-]+),\s*"
        #     r"val_feature:\s*([\d\.e\-]+),\s*"
        #     r"val_mse:\s*([\d\.e\-]+),\s*"
        #     r"val_maxrel:\s*([\d\.e\-]+),\s*"
        #     r"val_ssim:\s*\[([\d\.,\s\-e]+)\]"
        # )
        for match in pattern.finditer(training_messages):
            epochs.append(int(match.group(1)))
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
            
        ssim_transposed = list(map(list, zip(*val_ssim)))
        fig,axes = plt.subplots(1,3, figsize=(20,5))
        
        axes[0].plot(epochs,train_loss, label='train')
        axes[0].plot(epochs,val_loss,label='val')
        axes[0].legend()
        axes[0].set_title('Training History')
        axes[0].set_xlabel('epoch')
        axes[0].set_ylabel('Feature loss + MSE loss')
        axes[0].grid(True)

        axes[1].plot(epochs, val_maxrel)
        axes[1].set_title('MaxRel History')
        axes[1].set_xlabel('epoch')
        axes[1].grid(True)
        num_freqs = 7
        for freq_idx, freq_ssim in enumerate(ssim_transposed):
            plt.plot(epochs, freq_ssim, label=f'Frequency {freq_idx}')
        axes[2].set_title('SSIM History')
        axes[2].set_xlabel('epoch')
        axes[2].grid(True)
        plt.savefig(save_path)
    
    def single_impl(self,img_dir):
        self.check_fir(img_dir)
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
            plt.savefig(f'{img_dir}/ex{i}.png')
            plt.close() 

    def mulfreq_impl(self,img_dir):
        '''save_tar_pred_path: str, target_pred.png'''
        self.check_fir(img_dir)
        data = self.read_pkl_imgs()
        # log space    
        pred = data['predictions']
        target = data['targets']
        for sample_idx in range(len(pred)):
            target = target[sample_idx,0]
            pred   = pred[sample_idx,0]
            rows = 2
            colms = pred.shape[0]
            fig,axes = plt.subplots(rows,colms,figsize=(14,7))
            for colm in range(colms):
                img1 = axes[0,colm].imshow(target[colm],vmin=np.min(target[colm]), vmax=np.max(target[colm]))
                axes[0,colm].set_title(f'freq {colm+13}')
                
                img2 = axes[1,colm].imshow(pred[colm],vmin=np.min(pred[colm]), vmax=np.max(pred[colm]))
                fig.colorbar(img2,ax=axes[1])
            axes[0,0].set_ylabel('target')
            axes[1,0].set_ylabel('pred')
            plt.savefig(f'{img_dir}/exp_{sample_idx}.png')