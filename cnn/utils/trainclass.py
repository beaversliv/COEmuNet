import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset,DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity

from .loss                 import calculate_ssim_batch,MaxRel
from tqdm                 import tqdm
import numpy as np
import os
import sys
import json
import pickle
import logging
class Logging:
    def __init__(self, file_dir:str, file_name:str):
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        self.log_file = os.path.join(file_dir, file_name)
        print(f'create {log_file}')
        open(self.log_file, 'w').close()
    def info(self, message, gpu_rank=0, console=True):
        # only log rank 0 GPU if running with multiple GPUs/multiple nodes.
        if gpu_rank is None or gpu_rank == 0:
            if console:
                print(message)

            with open(self.log_file, 'a') as f:  # a for append to the end of the file.
                print(message, file=f)
    def write_log_metrics(self, epoch, train_loss, val_loss, relative_loss, ssim_values, gpu_rank=0, console=True):
        # Create a structured log entry
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'relative_loss': relative_loss,
            'ssim': ssim_values
        }
        # Convert log entry to a JSON string
        message = json.dumps(log_entry)
        self.info(message, gpu_rank, console)
class Trainer:
    def __init__(self,model,loss_object,optimizer,train_dataloader,test_dataloader,config,device):
        self.model = model
        self.loss_object = loss_object
        self.optimizer   = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader  = test_dataloader
        self.config = config
        self.device = device
        logger = Logging(config['save_path'], config['logfile'])
        self.logger = logger
    
    def train(self):
        total_loss = 0.
        self.model.train()
         
        for bidx, samples in enumerate(self.train_dataloader):
            data, target = Variable(samples[0]).to(self.device), Variable(samples[1]).to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_object(target, output)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach()
        epoch_loss = total_loss / len(self.train_dataloader)  # divide number of batches

        return epoch_loss

    
    def test(self):
        self.model.eval()
        P = []
        T = []
        total_loss = 0.
        for bidx, samples in enumerate(self.test_dataloader):
            data, target = Variable(samples[0]).to(self.device), Variable(samples[1]).to(self.device)
            pred = self.model(data)
            loss = self.loss_object(target, pred)
            
            P.append(pred.detach().cpu().numpy())
            T.append(target.detach().cpu().numpy())
            total_loss += loss.detach()
        val_loss = total_loss / len(self.test_dataloader)
        P = np.vstack(P)
        T = np.vstack(T)

        original_targets = self.postProcessing(P)   
        original_preds   = self.postProcessing(T)  
        maxrel = MaxRel(original_targets,original_preds)
        return P,T,val_loss,maxrel
    def run(self):
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(self.config['epochs'])):
            epoch_loss = self.train()
            epoch_loss = epoch_loss.cpu().item()
            torch.cuda.empty_cache()  # Clear cache after training
            
            t, p, val_loss,maxrel = self.test()
            val_loss = val_loss.cpu().item()
            torch.cuda.empty_cache()  # Clear cache after evaluation
            self.log_metrics(epoch, epoch_loss, val_loss, p, t)

            if maxrel < best_loss:
                best_loss = maxrel
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config['save_path']+self.config['model_name'])
            else:
                patience_counter += 1

            # if patience_counter >= self.config['patience']:
            #     print("Early stopping triggered")
            #     break

    def log_metrics(self, epoch, epoch_loss, val_loss, preds, targets):
        original_targets = self.postProcessing(targets)   
        original_preds   = self.postProcessing(preds)  
        relative_loss = MaxRel(original_targets,original_preds)
        avg_ssim = calculate_ssim_batch(targets, preds)
        self.logger.write_log_metrics(
            epoch = epoch,
            train_loss = epoch_loss,
            val_loss = val_loss,
            relative_loss = relative_loss,
            ssim_values = avg_ssim)
        
    def save(self,history_path):
        self.model.load_state_dict(torch.load(self.config['save_path']+self.config['model_name']))
        pred, target, test_loss,maxrel = self.test()
        assert len(target) == len(pred), "Targets and predictions must have the same length"
        # Save the training history
        try:
            with open(history_path, "wb") as pickle_file:
                pickle.dump({
                    "predictions": pred,
                    "targets": target
                }, pickle_file)
            print(f"Data successfully saved to {history_path}")
        except Exception as e:
            print(f"Error saving data to pickle file: {e}")
        print('Test Epoch: {} Loss: {:.4f}\n'.format(self.config["epochs"], test_loss.cpu().item()))
        print(f'relative loss {maxrel:.5f}%')

        avg_ssim = calculate_ssim_batch(target,pred)
        for freq in range(len(avg_ssim)):
            print(f'frequency {freq + 1} has ssim {avg_ssim[freq]:.4f}')

    def postProcessing(self,y):
        min_ = -47.387955
        median = 8.168968
        y = y*median + min_
        y = np.exp(y)
        return y 

### train step ###
class ddpTrainer:
    def __init__(self,ddp_model,train_dataloader,test_dataloader,optimizer,loss_object,config,rank,world_size):
        self.ddp_model = ddp_model
        self.train_dataloader = train_dataloader
        self.test_dataloader  = test_dataloader

        self.config = config
        self.rank   = rank # global rank, world_size -1, distinguish each process
        self.world_size = world_size # num GPUs
        # Define the optimizer for the DDP model
        self.optimizer = optimizer
        self.loss_object = loss_object
        logger = Logging(config['save_path'], config['logfile'])
        self.logger = logger
                                       
    def train(self):
        total_loss = 0.
        self.ddp_model.train()
        
        for bidx, samples in enumerate(self.train_dataloader):
            data, target = Variable(samples[0]).to(self.rank), Variable(samples[1]).to(self.rank)
            self.optimizer.zero_grad()
            output = self.ddp_model(data)
            loss = self.loss_object(target, output)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach()
        epoch_loss = total_loss / len(self.train_dataloader)  # divide number of batches
        return epoch_loss

    
    def test(self):
        self.ddp_model.eval()
        P = []
        T = []
        L = []
        for bidx, samples in enumerate(self.test_dataloader):
            data, target = Variable(samples[0]).to(self.rank), Variable(samples[1]).to(self.rank)
            pred = self.ddp_model(data)
            loss = self.loss_object(target, pred)
            
            P.append(pred.detach())
            T.append(target.detach())
            L.append(loss.detach())
        P = torch.cat(P, dim=0)
        T = torch.cat(T,dim=0)
        L = torch.stack(L)
        return P,T,torch.mean(L)
    def log_metrics(self, epoch, aggregated_epoch_loss, val_loss, preds, argets):
        best_loss = float('inf')
        patience_counter = 0
        # Ensure targets and predictions have the same length
        assert len(all_targets) == len(all_preds), "Targets and predictions must have the same length"
        original_target = self.postProcessing(targets)
        original_pred = self.postProcessing(preds)
        relative_loss = self.relativeLoss(original_target, original_pred)
        
        avg_ssim = calculate_ssim_batch(targets, preds)

        self.logger.write_log_metrics(
            epoch = epoch,
            train_loss = aggregated_epoch_loss.cpu().item(),
            val_loss = val_loss.cpu().item(),
            relative_loss = relative_loss,
            ssim_values = avg_ssim)
        # if avg_ssim[0] > 0.98:
        #     return True # Indicate that training should stop
        # else:
        #     return False
        if relative_loss < best_loss:
            best_loss = relative_loss
            patience_counter = 0
            torch.save(self.ddp_model.module.state_dict(), self.config['save_path']+self.config['model_name'])
        else:
            patience_counter += 1
        if patience_counter >= self.config['patience']:
            print("Early stopping triggered")
            return True
        else:
            return False
    def run(self):
        stop_signal = torch.tensor([0], device=self.rank)
        for epoch in tqdm(range(self.config['epochs']), disable=self.rank != 0):  # Disable tqdm progress bar except for rank 0
            epoch_loss = self.train()
            torch.cuda.empty_cache()  # Clear cache after training
            # Update history on master process
            if self.rank == 0:
                pred, target, val_loss = self.test()
                stop_training = self.log_metrics(epoch, aggregated_epoch_loss, val_loss, pred, target)
                if stop_training:
                    stop_signal[0] = 1  # Set stop signal

            # Broadcast the stop signal to all processes
            dist.broadcast(stop_signal, src=0)
            if stop_signal[0].item() == 1:
                break
 
    def save(self, history_path, world_size):
        if self.rank == 0:
            self.ddp_model.load_state_dict(torch.load(self.config['save_path']+self.config['model_name']))
            pred, target, test_loss = self.test()
            assert len(target) == len(pred), "Targets and predictions must have the same length"
            # Save the training history
            with open(history_path, "wb") as pickle_file:
                pickle.dump({
                    "predictions": pred,
                    "targets": target
                }, pickle_file)
            print(f'saved {history_path}!\n')
            print('Test Epoch: {} Loss: {:.4f}\n'.format(self.config["epochs"], test_loss.cpu().item()))
            original_target = self.postProcessing(target)
            original_pred = self.postProcessing(pred)
            print(f'relative loss {self.relativeLoss(original_target,original_pred):.5f}%')

            avg_ssim = calculate_ssim_batch(target,pred)
            for freq in range(len(avg_ssim)):
                print(f'frequency {freq + 1} has ssim {avg_ssim[freq]:.4f}')

            # # Plot and save images
            # img_plt(all_targets, all_preds, path=path)
    def postProcessing(self,y):
        # rotate stats
        # min_ , max_ = -103.27893, -28.09121
        # min_ = -50.24472
        # median = 11.025192
        min_ = -47.387955
        median = 8.168968
        y = y*median + min_
        # y = y * (max_ - min_) + min_
        y = np.exp(y)
        return y    
    def relativeLoss(self,original_target,original_pred):
        return np.mean( np.abs(original_target-original_pred) / np.max(original_target, axis=1,keepdims=True)) * 100
        return relative_loss