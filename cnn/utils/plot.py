import numpy as np
import matplotlib.pyplot as plt
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


def img_plt(target,pred,path):
    for i in range(0,len(pred),10):
        fig, axs = plt.subplots(1, 2,figsize=(12, 5))
        im1 = axs[0].imshow(target[i][0],vmin=np.min(target[i][0]),vmax = np.max(target[i][0]))
        axs[0].set_title('target')
        fig.colorbar(im1,ax=axs[0])

        im2 = axs[1].imshow(pred[i][0],vmin=np.min(target[i][0]),vmax = np.max(target[i][0]))
        axs[1].set_title('prediction')
        fig.colorbar(im2,ax=axs[1])
        plt.savefig(f'{path}ex{i}.png')
        plt.close()

class HistoryShow:
    def __init__(self,log_file,save_img_path):
        self.log_file = log_file
        self.save_img_path = save_img_path
    def read_training_messages(self):
        with open(self.log_file, 'r') as file:
            return file.read() 
    def history_show(self): 
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

        # Regular expression to extract values from the text
        pattern = re.compile(
            r"epoch:(\d+), train_loss:(\d+\.\d+),train_feature:(\d+\.\d+),train_mse:(\d+\.\d+),val_loss:(\d+\.\d+),val_feature:(\d+\.\d+),val_mse:(\d+\.\d+),val_maxrel:(\d+\.\d+),val_ssim:\[(.*?)\]"
        )
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
        plt.savefig(self.save_img_path)