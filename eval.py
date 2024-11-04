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

from utils.ResNet3DModel  import Net3D,Net
from utils.utils          import HistoryShow,check_dir,Logging,LoadCheckPoint
import pickle

import os

class ModelEval:
    def __init__(self, save_dir):
        """
        Initialize the ModelEval class.
        
        Parameters:
        config_path (str): Path to the configuration file.
        save_dir (str): Directory where results will be saved.
        """
        args = parse_args()
        config = load_config(args.config) # read config
        self.config = merge_config(args, config)

        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.logger = Logging(self.save_dir, 'evaluation.txt')

    def setup_model(self):
        """Setup the model and optimizer based on the configuration."""
        if self.config['dataset']['name'] == 'mulfreq':
            self.model = Net3D(7, self.config['dataset']['grid']).to(self.device)
        else:
            self.model = Net(self.config['dataset']['grid']).to(self.device)

        optimizer_params = self.config['optimizer']['params']
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_params)

    def _load_checkpoint(self, checkpoint_path):
        """Load the model checkpoint."""
        self.logger = Logging(self.save_dir, 'evaluation.txt')  # Initialize logger
        checkpoint_loading = LoadCheckPoint(
            learning_model=self.model,
            optimizer=self.optimizer,
            file_path=checkpoint_path,
            stage=2,
            logger=self.logger,
            local_rank=self.device,
            ddp_on=False
        )
        checkpoint_loading.load_checkpoint()

    def prepare_data_loaders(self):
        """Prepare the training and testing data loaders."""
        train_file_list_path = f"dataset/data/{self.config['dataset']['name']}/train.txt"
        test_file_list_path = f"dataset/data/{self.config['dataset']['name']}/test.txt"
        train_dataset = ChunkLoadingDataset(train_file_list_path, 512)
        test_dataset = ChunkLoadingDataset(test_file_list_path, 512)

        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return train_dataloader, test_dataloader
    
    def evaluate(self, full_model_checkpoint_path,log_files):
        """Main evaluation function."""
        self.setup_model()
        self._load_checkpoint(full_model_checkpoint_path)

        train_dataloader, test_dataloader = self.prepare_data_loaders()
        loss_object = FreqMse(alpha=self.config['model']['alpha'])

        self.save_pkl(loss_object, train_dataloader, test_dataloader)

        # pickle_path = os.path.join(self.save_dir, 'pickl.pkl')
        # history_plot = HistoryShow(self.save_dir,single=False)
        # _,_,_,val_maxrel,ssim_transposed = history_plot.read_training_message(log_files)
   
        # lowest_maxrel = min(val_maxrel)
        # lowest_epoch = val_maxrel.index(lowest_maxrel)
        
        # corr_highest_ssim = [freq[lowest_epoch] for freq in ssim_transposed]
    
        # self.logger.info(f'epoch {lowest_epoch} has lowest maxrel {lowest_maxrel:.4f}%, the corresponding ssim is {corr_highest_ssim}')
        # history_plot.history_show(log_files)
        # history_plot.img_plt(pickle_path)


    def save_pkl(self, loss_object, train_dataloader, test_dataloader):
        """Save predictions and targets to a pickle file."""
        trainer = Trainer(self.model, loss_object, self.optimizer, train_dataloader, test_dataloader, self.config, self.device, self.logger)
        preds, targets, total_loss, freq_loss, mse = trainer.test()

        assert len(targets) == len(preds), "Targets and predictions must have the same length"
        
        original_targets, original_preds = trainer.postProcessing(targets, preds)

        maxrel_calculation = MaxRel(original_targets, original_preds)
        exclude_maxrel_error_2x2 = maxrel_calculation.maxrel(center_size=2)
        exclude_maxrel_error_4x4 = maxrel_calculation.maxrel(center_size=4)
        exclude_maxrel_error_6x6 = maxrel_calculation.maxrel(center_size=6)

        self.logger.info(f'maxrel exclude centre 2x2: {exclude_maxrel_error_2x2:.5f},' 
                         f'maxrel exclude centre 4x4: {exclude_maxrel_error_4x4:.5f},'
                         f'maxrel exclude centre 4x4: {exclude_maxrel_error_6x6:.5f},')

        # Save results
        self.save_results(preds, targets, original_targets, original_preds)


    def save_results(self, preds, targets, original_targets, original_preds):
        """Save evaluation results to a pickle file."""
        img_dir = os.path.join(self.save_dir, 'img/')
        check_dir(img_dir)  # Ensure the directory exists

        pickle_path = os.path.join(self.save_dir, 'pickl.pkl')
        with open(pickle_path, "wb") as pickle_file:
            pickle.dump({
                "predictions": preds[100:200].numpy(),
                "targets": targets[100:200].numpy(),
                "or_targets": original_targets[100:200].numpy(),
                "or_preds": original_preds[100:200].numpy()
            }, pickle_file)

        self.logger.info(f'Saved results to {pickle_path}')

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



if __name__ == '__main__':
    full_model_checkpoint_path = '/home/dp332/dp332/dc-su2/results/mulfreq/ro_200_2/best_sofar_ro_200_2.pth'
    save_dir = '/home/dp332/dp332/dc-su2/results/mulfreq/ro_200_2'
    log_files = ['/home/dp332/dp332/dc-su2/results/mulfreq/ro_200/ro_200.txt',
                 '/home/dp332/dp332/dc-su2/results/mulfreq/ro_200_1/ro_200_1.txt',
                 '/home/dp332/dp332/dc-su2/results/mulfreq/ro_200_2/ro_200_2.txt']
    model_eval = ModelEval(save_dir)
    model_eval.evaluate(full_model_checkpoint_path,log_files)

    