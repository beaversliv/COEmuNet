import torch
import torch.nn       as nn
import torch.distributed as dist
from torch.utils.data         import DataLoader, TensorDataset,DistributedSampler
from torch.nn.parallel        import DistributedDataParallel as DDP
from torch.autograd   import Variable

import matplotlib.pyplot as plt

from utils.ResNet3DModel  import Net,Net3D
from utils.dataloader     import ChunkLoadingDataset,MaxRelDataset
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

class ModelEval(ddpTrainer):
    def __init__(self, ddp_model, train_dataloader,test_dataloader, optimizer,loss_object, config, rank,local_rank, world_size, logger, scheduler,save_dir):
        super().__init__(ddp_model, train_dataloader,test_dataloader, optimizer,loss_object, config, rank,local_rank, world_size, logger, scheduler)
        """
        Initialize the ModelEval class.
        
        Parameters:
        config_path (str): Path to the configuration file.
        save_dir (str): Directory where results will be saved.
        """

        self.save_dir = save_dir
        self.logger   = Logging(save_dir, 'evaluation.txt')
        self.pickle_path  = os.path.join(self.save_dir, 'pickl.pkl')
        self.maxrel_files = os.path.join(self.save_dir, f'{rank}_maxrel_values.json')

    def evaluate(self):
        """Main evaluation function."""
        
        preds, targets, total_loss, freq_loss, mse = super().test()

        assert len(targets) == len(preds), "Targets and predictions must have the same length"
        matrics_calculation = EvaluationMetrics(postprocess_fn=super().postProcessing)
        metrics = matrics_calculation.evaluate(targets, preds)

        # zncc = zncc_batch(preds,targets)
        # self.logger.info(f'zncc : {zncc}')
        
        original_targets, original_preds = super().postProcessing(targets), super().postProcessing(preds)

        # maxrel_calculation = MaxRel(original_targets, original_preds)
        # exclude_maxrel_error_2x2 = maxrel_calculation.maxrel(center_size=2)
        # exclude_maxrel_error_4x4 = maxrel_calculation.maxrel(center_size=4)
        # exclude_maxrel_error_6x6 = maxrel_calculation.maxrel(center_size=6)

        # self.logger.info(f'maxrel exclude centre 2x2: {torch.median(exclude_maxrel_error_2x2):.5f}') 
        # self.logger.info(f'maxrel exclude centre 4x4: {torch.median(exclude_maxrel_error_4x4):.5f}')
        # self.logger.info(f'maxrel exclude centre 6x6: {torch.median(exclude_maxrel_error_6x6):.5f}')
        
        # ssim = calculate_ssim_batch(targets,preds)
        # self.logger.info(f'ssim : {ssim}')
        # Save results
        self.save_results(original_targets, original_preds)

        # record train history and imgs
        # history_plot = HistoryShow(self.save_dir,single=False)
        # _,_,_,val_maxrel,ssim_transposed = history_plot.read_training_message(log_files)
   
        # lowest_maxrel = min(val_maxrel)
        # lowest_epoch = val_maxrel.index(lowest_maxrel)
        
        # corr_highest_ssim = [freq[lowest_epoch] for freq in ssim_transposed]
    
        # self.logger.info(f'epoch {lowest_epoch} has lowest maxrel {lowest_maxrel:.4f}%, the corresponding ssim is {corr_highest_ssim}')
        # history_plot.history_show(log_files)
        # history_plot.img_plt(self.pickle_path)

    def maxrel_evaluate(self):
        maxrel_data = super().per_sample_eval()
        # self.logger.info(maxrel_data)
                
        # original maxrel
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
        with open(self.maxrel_files, 'w') as f:
            json.dump(MAXRELS, f)
        
        # Log the counts for each range
        for range_key, count in counts.items():
            self.logger.info(f"rank {rank} {range_key}: {count} samples")
        # # Plot distribution
        # plt.hist(np.log(all_maxrel_values), bins=20)
        # plt.xlabel('MaxRel Value')
        # plt.ylabel('Frequency')
        # plt.title('Distribution of MaxRel Values in log space')
        # plt.savefig(f'{self.save_dir}/maxrel_distribution.png')

    def high_maxrel_sample_locate(self,selected_range):
        dir_ = os.path.join(self.save_dir,selected_range)
        check_dir(dir_)
        self.logger.info(f'read {self.maxrel_files}')
        with open(self.maxrel_files,'r') as f:
            maxrel_file_sampleidx = json.load(f)

        # Example: Analyze each high MaxRel sample
        self.ddp_model.eval()
        for idx,entry in enumerate(maxrel_file_sampleidx[selected_range][0:60:2]):
            sample_file = entry['file']
            # Load the specific sample from the file
            with h5.File(sample_file, 'r') as f:
                data   = f['input']  # Load only the specific sample
                target = f['output']  # Load only the specific sample (1,7,64,64)
                data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.local_rank)  # Add batch dimension
                target = torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(self.local_rank)  # Add batch dimension
            # Pass the sample through the model
            with torch.no_grad():  # Disable gradient calculation for inference
                pred = self.ddp_model(data)

            or_target = super().postProcessing(target)
            or_pred = super().postProcessing(pred)

            # maxrel_calculation = MaxRel(or_target, or_pred)
            # maxrel = maxrel_calculation.maxrel(center_size=0)

            or_target = or_target.cpu().numpy()
            or_pred = or_pred.cpu().numpy()

            file_name = os.path.basename(sample_file)  # "clean_0770_0.hdf5"

            # Extract the model (model218) and specific identifier (0770_0)
            model = sample_file.split('/')[-3]  # "model218"
            
            identifier = "_".join(file_name.split('_')[1:3]).split('.')[0]

            path = os.path.join(dir_,f'{model}_{identifier}.png')
            self.mulfreq_imshow(or_target,or_pred,path)

    def mulfreq_imshow(self,target,pred,path):
        '''
        input: single paried target and pred, in original space
        '''
        target = target.reshape(7,64,64)
        pred   = pred.reshape(7,64,64)

        maxrel_0 = np.abs(target - pred) / np.max(target,axis=0,keepdims=True) * 100

        lg_target = np.log(target)
        lg_pred   = np.log(pred)

        plt.rc('axes', titlesize=13)  # Title font size
        plt.rc('axes', labelsize=13)  # Axis label font size
        plt.rc('xtick', labelsize=8)  # X-tick font size
        plt.rc('ytick', labelsize=8)  # Y-tick font size

        rows = 3
        colms = 7
        fig,axes = plt.subplots(rows,colms,figsize=(25,10))
        for colm in range(colms):
            img1 = axes[0,colm].imshow(lg_target[colm],vmin=np.min(lg_target[colm]), vmax=np.max(lg_target[colm]))
            axes[0,colm].set_title(rf"frequency $\nu_{{{colm + 13}}}$")
            
            img2 = axes[1,colm].imshow(lg_pred[colm],vmin=np.min(lg_target[colm]), vmax=np.max(lg_target[colm]))
            img3 = axes[2,colm].imshow(maxrel_0[colm],vmin=np.min(maxrel_0[colm]), vmax=np.max(maxrel_0[colm]),cmap='cividis')
            # axes[2, colm].text(
            #     0.5, 1.05,  # Adjust position
            #     f'min: {np.min(maxrel_0[colm]):.2f}, max: {np.max(maxrel_0[colm]):.2f}',  # Format min/max
            #     color='black', fontsize=10, ha='center', va='bottom',  # Center the text
            #     transform=axes[2, colm].transAxes  # Set relative to the axis
            # )

            fig.colorbar(img1, ax=axes[0, colm], orientation='vertical', fraction=0.05, pad=0.04)
            fig.colorbar(img2, ax=axes[1, colm], orientation='vertical', fraction=0.05, pad=0.04)
            fig.colorbar(img3, ax=axes[2, colm], orientation='vertical', fraction=0.05, pad=0.04)
        fig.suptitle(f'sample has maxrel {np.mean(maxrel_0)}%')
        axes[0,0].set_ylabel('target')
        axes[1,0].set_ylabel('pred')
        axes[2,0].set_ylabel('maxrel(%)')
        # plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce space between subplots
        plt.tight_layout(pad=0.1)
   
        plt.savefig(path)

    def save_results(self, original_targets, original_preds):
        """Save evaluation results to a pickle file."""
        with open(self.pickle_path, "wb") as pickle_file:
            pickle.dump({
                "or_targets": original_targets[:400].numpy(),
                "or_preds": original_preds[:400].numpy()
            }, pickle_file)

        self.logger.info(f'Saved results to {self.pickle_path}')

def time_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_grid = 32
    print('model grid',model_grid)
    model = Net(model_grid).to(device)
    input_shape = (3,model_grid,model_grid,model_grid)
    data = torch.randn(500, *input_shape)  # 5000 samples with appropriate shape
    batch_size = 100

    # Time the inference
    start_time = time.time()
    with torch.no_grad():  # Disable gradient calculation for inference
        for i in range(0, data.size(0), batch_size):
            batch = data[i:i+batch_size].to(device)
            print(i,i+batch_size)
            output = model(batch)
    end_time = time.time()

    # Total inference time and per-sample time
    total_time = end_time - start_time
    time_per_sample = total_time / data.size(0)

    print(f'Total inference time: {total_time:.4f} seconds')
    print(f'Inference time per sample: {time_per_sample:.6f} seconds')

def main():
    args = parse_args()
    config = load_config(args.config) # read config
    config = merge_config(args, config)

    world_size    = int(os.environ.get("SLURM_NTASKS"))
    rank          = int(os.environ.get("SLURM_PROCID"))
    gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE",torch.cuda.device_count()))
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    logger        = Logging('/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/rotate/results/mul', 'evaluation.txt')

    setup(rank, world_size)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)

    if rank == 0:
        # logger.info(yaml.dump(config, default_flow_style=False, sort_keys=False))

        if gpus_per_node == 0:
            logger.info("No GPUs available on this node.")
        else:
            logger.info(f"GPUs available: {gpus_per_node}")

    logger.info(f"Hello from rank {rank} of {world_size} on {socket.gethostname()} where there are {gpus_per_node} allocated GPUs and {num_workers} numb workers per node.")
    logger.info(f'Rank {rank} of local rank {local_rank} finishes setup')

    """Prepare the training and testing data loaders."""
    test_file_list_path = '/home/dc-su2/physical_informed_old/aluxinary_files/extra_mulfreq_clean_files.json'
    test_dataset   = MaxRelDataset(test_file_list_path)
    train_dataset  = MaxRelDataset(test_file_list_path)
    # train_file_list_path = f"dataset/data/{config['dataset']['name']}/train.txt"
    # test_file_list_path = f"dataset/data/{config['dataset']['name']}/test.txt"
    # train_dataset = ChunkLoadingDataset(train_file_list_path, 512)
    # test_dataset = ChunkLoadingDataset(test_file_list_path, 500)

    sampler_test = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
    test_dataloader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], sampler=sampler_test,shuffle=False,num_workers=4, pin_memory=True, prefetch_factor=2)

    '''Prepare model and optimizer'''
    if config['dataset']['name'] == 'mulfreq':
        model = Net3D(7, config['dataset']['grid']).to(local_rank)
    else:
        model = Net(config['dataset']['grid']).to(local_rank)


    file_path = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/rotate/results/mul/best_sofar_ro_200_final.pth'
    checkpoint_loading = LoadCheckPoint(
        learning_model=model,
        optimizer=None,
        file_path=file_path,
        stage=config['model']['stage'],
        logger=logger,
        local_rank=f'cuda:{local_rank}',
        ddp_on=False
    )
    checkpoint_loading.load_checkpoint(model_only=True)
    # Wrap the model with DDP
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_object = FreqMse(alpha=config['model']['alpha'])

    logger = None

    save_dir = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/rotate/results/mul'
    log_files = ['/home/dp332/dp332/dc-su2/results/mulfreq/mulfreq_del_later/ro_200_final.txt']

    ranges = ['low_than_10','10_20','20_30','30_40','40_50','50_60','60_70','70_80','80_90','90_100','high_than_100']

    model_eval =  ModelEval(
            ddp_model=ddp_model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=None,
            loss_object=loss_object,
            config=config,
            rank = rank,
            local_rank = local_rank,
            world_size = world_size,
            logger=logger,
            scheduler=None,
            save_dir=save_dir
        )
    # model_eval.evaluate()
    # model_eval.results_record()
    model_eval.maxrel_evaluate()
    if rank == 0:
        # model_eval.high_maxrel_sample_locate(selected_range='10_20')
        # model_eval.high_maxrel_sample_locate(selected_range='20_30')
        # model_eval.high_maxrel_sample_locate(selected_range='30_40')
  
        model_eval.high_maxrel_sample_locate(selected_range='high_than_100')
class OneGpuModelEval(Trainer):
    def __init__(self, model, loss_object,optimizer,train_dataloader,test_dataloader,config,device,logger,scheduler,save_dir):
        super().__init__(model, loss_object,optimizer,train_dataloader,test_dataloader,config,device,logger,scheduler=None)
        """
        Initialize the ModelEval class.
        
        Parameters:
        config_path (str): Path to the configuration file.
        save_dir (str): Directory where results will be saved.
        """

        self.save_dir = save_dir
        self.logger   = Logging(save_dir, 'evaluation.txt')
        self.pickle_path  = os.path.join(self.save_dir, 'pickl.pkl')

    def evaluate(self):
        """Main evaluation function."""
        
        preds, targets, total_loss, freq_loss, mse = super().test()

        assert len(targets) == len(preds), "Targets and predictions must have the same length"
        matrics_calculation = EvaluationMetrics(postprocess_fn=super().postProcessing)
        metrics = matrics_calculation.evaluate(targets, preds)
        self.logger.info(f'metrics : {metrics}')
        
        # original_targets, original_preds = super().postProcessing(targets), super().postProcessing(preds)
        # # Save results
        # self.save_results(original_targets, original_preds)
        # img_path = os.path.join(self.save_dir,'img')
        # check_dir(img_path)
        # for i in range(10):
        #     path = os.path.join(img_path,f'exp_{i}.png')
        #     mulfreq_imshow(original_targets[i].numpy(),original_preds[i].numpy(),path)
    def save_results(self, original_targets, original_preds):
        """Save evaluation results to a pickle file."""
        with open(self.pickle_path, "wb") as pickle_file:
            pickle.dump({
                "or_targets": original_targets[:400].numpy(),
                "or_preds": original_preds[:400].numpy()
            }, pickle_file)

        self.logger.info(f'Saved results to {self.pickle_path}')

def singleGpu_main():
    args = parse_args()
    config = load_config(args.config) # read config
    config = merge_config(args, config)

    """Prepare the training and testing data loaders."""
    train_file_list_path = f"dataset/data/{config['dataset']['name']}/train.txt"
    test_file_list_path = f"dataset/data/{config['dataset']['name']}/test.txt"
    train_dataset = ChunkLoadingDataset(train_file_list_path, 512)
    test_dataset = ChunkLoadingDataset(test_file_list_path, 500)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
    test_dataloader = DataLoader(test_dataset, batch_size=1,shuffle=False,num_workers=4, pin_memory=True, prefetch_factor=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''Prepare model and optimizer'''
    if config['dataset']['name'] == 'mulfreq':
        model = Net3D(7, config['dataset']['grid']).to(device)
    else:
        model = Net(config['dataset']['grid']).to(device)

    file_path = config['model'].get('resume_checkpoint',None)
    save_dir = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/pretrained/'
    logger        = Logging(save_dir, 'evaluation.txt')
    checkpoint_loading = LoadCheckPoint(
        learning_model=model,
        optimizer=None,
        scheduler=None,
        file_path=file_path,
        stage=config['model']['stage'],
        logger=logger,
        local_rank=device,
        ddp_on=False
    )
    checkpoint_loading.load_checkpoint(model_only=True)
    # Wrap the model with DDP
    loss_object = FreqMse(alpha=config['model']['alpha'])

    model_eval =  OneGpuModelEval(
            model=model,
            loss_object=loss_object,
            optimizer=None,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            config=config,
            device = device,
            logger=logger,
            scheduler=None,
            save_dir=save_dir
        )
    model_eval.evaluate()
   
if __name__ == '__main__':
    singleGpu_main()
    # with open('/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/mulfreq/final_3/pickl.pkl', "rb") as fo:
    #     data = pickle.load(fo, encoding="bytes")
    # original_targets = data['or_targets']
    # original_preds   = data['or_preds']
    # img_path = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/mulfreq/final_3/img'
    # for i in range(200):
    #     path = os.path.join(img_path,f'exp_{i}.png')
    #     mulfreq_imshow(original_targets[i],original_preds[i],path)
    