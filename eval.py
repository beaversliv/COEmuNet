import torch
import torch.nn       as nn
import torch.distributed as dist
from torch.utils.data         import DataLoader, TensorDataset,DistributedSampler
from torch.nn.parallel        import DistributedDataParallel as DDP
from torch.autograd   import Variable

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.ResNet3DModel  import Net,Net3D
from utils.dataloader     import ChunkLoadingDataset,SequentialDataset
from utils.loss     import FreqMse,EvaluationMetrics
from utils.config         import parse_args,load_config,merge_config

from utils.trainclass     import ddpTrainer,Trainer

import time
import numpy              as np
import h5py               as h5
import statistics
from utils.ResNet3DModel  import Net3D,Net
from utils.utils          import HistoryShow,check_dir,Logging,LoadCheckPoint,setup,cleanup,mulfreq_imshow
import pickle
import json

import os
import yaml
import socket
import sys  

from collections import OrderedDict

class maxrelDist:
    def __init__(self,model,test_dataloader,device,logger,save_dir):
        self.model = model
        self.test_dataloader = test_dataloader
        self.device          = device
        self.logger          = logger
        self.save_dir    = save_dir
        self.maxrel_files   = None
        file_lengths = [5000]*85
        self.cumulative_file_lengths = np.cumsum(file_lengths)

    def postProcessing(self,y):
        max_   = -27.064306
        min_   = -40.0
        y = y*(max_ - min_)+min_
        y = torch.exp(y)
        return y
    def per_sample_eval(self):
        self.model.eval()

        maxrel_data = []
        
        matrics_calculation = EvaluationMetrics(postprocess_fn=self.postProcessing)
        for bidx, (data, target,file_idx) in enumerate(self.test_dataloader):
            data, target = Variable(data.squeeze(0)).to(self.device), Variable(target.squeeze(0)).to(self.device)
            with torch.no_grad():
                pred = self.model(data)
            if bidx % 50 == 0:
                self.logger.info(f'Test batch [{bidx}/{len(self.test_dataloader)}]')

            batch_start_idx = bidx * 500
            maxrels_batch = matrics_calculation.calculate_maxrel(target, pred)
            for i, maxrel in enumerate(maxrels_batch):
                global_sample_idx = batch_start_idx + i
                if int(file_idx) == 0:
                    local_sample_idx = global_sample_idx
                else:
                    local_sample_idx = global_sample_idx - self.cumulative_file_lengths[file_idx - 1]
                self.logger.info(f'file idx:{file_idx},sample idx {local_sample_idx}')
                maxrel_data.append({
                    'maxrel': maxrel.item(),  # Convert tensor to a Python float
                    'file_idx': int(file_idx),  # Store the corresponding file information
                    'local_sample_idx':int(local_sample_idx)
                })
        return maxrel_data
    def maxrel_evaluate(self):
        maxrel_data = self.per_sample_eval()
        all_maxrel_values = [entry['maxrel'] for entry in maxrel_data]
        # Filter samples with MaxRel > 10%
        MAXRELS = {
            'low_than_10':[],
            '10_20':[],
            '20_30':[],
            '30_40':[],
            '40_50':[],
            '50_60':[],
            '60_70':[],
            '70_80':[],
            '80_90':[],
            '90_100':[],
            'high_than_100':[]
        }
        counts = {
            'low_than_10': 0,
            '10_20': 0,
            '20_30': 0,
            '30_40': 0,
            '40_50': 0,
            '50_60': 0,
            '60_70': 0,
            '70_80': 0,
            '80_90': 0,
            '90_100': 0,
            'high_than_100': 0
        }
        ranges = [
            (0, 10, 'low_than_10'),
            (10, 20, '10_20'),
            (20, 30, '20_30'),
            (30, 40, '30_40'),
            (40, 50, '40_50'),
            (50, 60, '50_60'),
            (60, 70, '60_70'),
            (70, 80, '70_80'),
            (80, 90, '80_90'),
            (90, 100, '90_100'),
        ]

        for entry in maxrel_data:
            maxrel_value = entry['maxrel']
            
            # Check ranges to find the correct bucket
            added = False
            for lower, upper, key in ranges:
                if lower <= maxrel_value < upper:
                    MAXRELS[key].append(entry)
                    counts[key] += 1
                    added = True
                    break
    
            # If above 100, add to the 'high_than_100' category
            if not added and maxrel_value >= 100:
                MAXRELS['high_than_100'].append(entry)
                counts['high_than_100'] += 1

        # Standardize MAXRELS by ensuring each category has a placeholder if empty
        for key in MAXRELS:
            if not MAXRELS[key]:  # If the list is empty, set to None
                MAXRELS[key] = None

        # Save the combined MAXRELS on rank 0
        self.maxrel_files = os.path.join(self.save_dir,'maxrel_distribution.json')
        with open(self.maxrel_files, 'w') as f:
            json.dump(MAXRELS, f)
        
        # Log the counts for each range
        for range_key, count in counts.items():
            self.logger.info(f"rank {range_key}: {count} samples")

        # Plot distribution
        fig, ax = plt.subplots(figsize=(5, 4))            

        ax.hist(all_maxrel_values, bins=120)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        plt.xlabel('MaxRel Value (%)')
        plt.ylabel('Frequency')
        plt.xlim((0,60))
        plt.title('Distribution of MaxRel Values')
        plt.savefig('/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/mulfreq/maxrel/maxrel_distribution.png')

    def high_maxrel_sample_locate(self,selected_range:str):
        dir_ = os.path.join(self.save_dir,selected_range)
        check_dir(dir_)
        # self.logger.info(f'read /home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/mulfreq/maxrel/maxrel_distribution.json')
        with open('/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/mulfreq/maxrel/maxrel_distribution.json','r') as f:
            maxrel_file_sampleidx = json.load(f)

        # Example: Analyze each high MaxRel sample
        self.model.eval()
        # for idx,entry in enumerate(maxrel_file_sampleidx[selected_range][0:30:2]):
            # file_idx = entry['file_idx']
            # local_sample_idx = entry['local_sample_idx']
        file_idx = 1
        local_sample_idx = 552 
        # Load the specific sample from the file
        sample_file = f'/home/dc-su2/rds/rds-dirac-dp012/dc-su2/physical_forward/mul_freq/grid64/Rotation/test_{file_idx}.hdf5'
        with h5.File(sample_file, 'r') as f:
            data   = f['input'][local_sample_idx]  # Load only the specific sample
            target = f['output'][local_sample_idx]  # Load only the specific sample (1,7,64,64)

            data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)  # Add batch dimension
            target = torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(self.device)  # Add batch dimension
        # Pass the sample through the model
        with torch.no_grad():  # Disable gradient calculation for inference
            pred = self.model(data)

        or_target = self.postProcessing(target)
        or_pred = self.postProcessing(pred)

        or_target = or_target.cpu().numpy()
        or_pred = or_pred.cpu().numpy()
        maxrel = np.abs(or_target - or_pred) / np.max(or_target,axis=0,keepdims=True)
        print(np.mean(maxrel))

        path = os.path.join(dir_,f'{file_idx}_{local_sample_idx}.png')
        with open("/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/mulfreq/maxrel/maxrel.pickl", "wb") as pickle_file:
            pickle.dump({
                "or_target": or_target,
                "or_pred": or_pred
            }, pickle_file)
        print('saved pickle')
        self.mulfreq_imshow(or_target,or_pred,path)
    
    def mulfreq_imshow(self,target,pred,path):
        '''
        input: single paried target and pred, in original space
        '''
        target = target.reshape(7,64,64)
        pred   = pred.reshape(7,64,64)

        maxrel = np.abs(target - pred) / np.max(target,axis=0,keepdims=True)

        lg_target = np.log(target[[0,3,6],:,:])
        lg_pred   = np.log(pred[[0,3,6],:,:])
        maxrel  = maxrel[[0,3,6],:,:]


        plt.rc('axes', titlesize=16)  # Title font size
        plt.rc('axes', labelsize=16)  # Axis label font size
        plt.rc('xtick', labelsize=16)  # X-tick font size
        plt.rc('ytick', labelsize=16)  # Y-tick font size

        rows = 3
        colms = 3
        fig, axes = plt.subplots(rows, colms, figsize=(15,13),dpi=100)
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
                    divider = make_axes_locatable(axes[row, colm])
                    cax = divider.append_axes("right", size="5%", pad=0.2)  # Reduce padding
                    fig.colorbar(img, cax=cax, orientation="vertical")

        # Ensure no spacing between images
        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.02, hspace=0.02)
        plt.savefig(path,dpi=1000)


def maxrel_main(config):
    folder = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/mulfreq/maxrel/'
    logger = Logging(folder,'maxrel_distribution.txt')

    """Prepare the training and testing data loaders."""
    test_file_list_path = f"dataset/data/{config['dataset']['name']}/test.txt"
    test_dataset = ChunkLoadingDataset(test_file_list_path, 500)
    test_dataloader = DataLoader(test_dataset, batch_size=1,shuffle=False,num_workers=4, pin_memory=True, prefetch_factor=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''Prepare model and optimizer'''
    if config['dataset']['name'] == 'mulfreq':
        model = Net3D(7, config['dataset']['grid']).to(device)
    else:
        model = Net(config['dataset']['grid']).to(device)
    # load checkpoint but without the last layer
    file_path = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/mulfreq/pretrained_tanh_1/best_sofar_pretrained_tanh_1.pth'
    checkpoint = torch.load(file_path, map_location=device, weights_only=True)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' if present (for non-DDP)
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)

    maxreldist = maxrelDist(
        model=model,
        test_dataloader=test_dataloader,
        device=device,
        logger = logger,
        save_dir=folder)
    maxreldist.maxrel_evaluate()
    maxreldist.high_maxrel_sample_locate('low_than_10')
    maxreldist.high_maxrel_sample_locate('30_40')
    maxreldist.high_maxrel_sample_locate('50_60')

def time_eval(model_grid):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logging('/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/nn_runtimes',f'{device}_mulfreq{model_grid}_1.log')
    logger.info('model grid',model_grid)

    model = Net3D(7, model_grid).to(device)
    batch_size = 100
    test_dataset = SequentialDataset(
        file_paths=[f'/home/dc-su2/rds/rds-dirac-dp012/dc-su2/physical_forward/mul_freq/grid64/Rotation/test_{i}.hdf5' for i in range(2)],transform=None)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    t = []
    # Time the inference
    with torch.no_grad():  # Disable gradient calculation for inference
        for bidx,samples in enumerate(test_dataloader):
            data,target = samples[0].to(device),samples[1].to(device)
            
            start_time = time.time()
            output = model(data)
            end_time = time.time()
            batch_inference_time = end_time - start_time
            t.append(batch_inference_time*1000)
            logger.info(f'Batch {bidx} calculation time:{batch_inference_time*1000:.4f} milliseconds')

    per_sample_times = np.array(t) / batch_size  # Element-wise division

    # Compute mean and standard deviation
    time_per_sample_mean = np.mean(per_sample_times)
    time_per_sample_std = np.std(per_sample_times, ddof=1)  # Use ddof=1 for sample std

    logger.info(f'Inference time per sample: {time_per_sample_mean:.2f} ± {time_per_sample_std:.2f} milliseconds')

if __name__ == '__main__':
    # maxrel_main(config)
    time_eval(model_grid=int(sys.argv[1]))


    