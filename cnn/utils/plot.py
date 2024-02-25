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

def history_plt(trainLoss:list,valLoss:list,path:str):
    ''' 
    plot training history according to train and validation losses.
    '''
    epoch = len(trainLoss)
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(epoch),np.log(trainLoss),label='train')
    plt.plot(np.arange(epoch),np.log(valLoss),label='val')

    plt.legend()
    plt.title('Training History')
    plt.xlabel('epoch')
    plt.ylabel('log MSE+L1F')
    plt.grid(True)
    plt.savefig(f'{path}history.png')
    plt.show()