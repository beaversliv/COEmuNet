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
