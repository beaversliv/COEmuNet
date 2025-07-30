import torch
import torch.distributed as dist
from torch.autograd import Variable

from utils.loss     import EvaluationMetrics
from utils.utils    import load_statistics
from tqdm           import tqdm
import numpy as np
import h5py  as h5
import os
import sys
import json
import pickle
import time
import logging
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
    
class singleTrainer:
    def __init__(self,model,loss_object,optimizer,train_dataloader,test_dataloader,config,device,logger,scheduler=None):
        self.model = model
        self.loss_object = loss_object
        self.optimizer   = optimizer
        self.scheduler        = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader  = test_dataloader
        self.config = config
        self.device = device
        self.logger = logger
        self.best_loss = float('inf')  # Initialize best loss
        self.patience_counter = 0
        statistics_path = config['dataset']['statistics']['path']
        statistics_values = config['dataset']['statistics']['values']
        self.dataset_stats  = load_statistics(statistics_path,statistics_values)
      
    def train(self):
        total_loss = 0.
        total_soble = 0.
        total_mse  = 0.
        P,T = [],[]
        self.model.train()
        start_time = time.time()
        for bidx, samples in enumerate(self.train_dataloader):
            data, target = Variable(samples[0].squeeze(0)).to(self.device), Variable(samples[1].squeeze(0)).to(self.device)
            output = self.model(data)
            soble_loss,mse = self.loss_object(target, output)
            loss = soble_loss + mse
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # end_time = time.time()
            # print(f'computation time for batch {bidx}: {end_time - start_time:.4f}')
            total_loss  += loss.detach()
            total_soble += soble_loss.detach()
            total_mse   += mse.detach()
            P.append(output.detach())
            T.append(target.detach())
            del data, target
            torch.cuda.empty_cache()
        # print(f'background loading file:{self.train_dataloader.dataset.future.result():.4f} s')
        # self.logger.info(self.train_dataloader.dataset.dataset.get_time())
        epoch_loss = total_loss / len(self.train_dataloader)  # divide number of batches
        epoch_soble = total_soble / len(self.train_dataloader)
        epoch_mse = total_mse / len(self.train_dataloader)
        return epoch_loss,epoch_soble,epoch_mse
    
    def get_average_metrics(self,results):
        """
        Calculate the average for each metric across all evaluations.
        
        Returns:
            dict: A dictionary containing the average of each metric.
        """
        average_metrics = {}
        if results['maxrel']:
            average_metrics['maxrel'] = sum(results['maxrel']) / len(results['maxrel'])
        else:
            average_metrics['maxrel'] = 0.0

        # Calculate average for `zncc` and `ssim`, each with 7 sub-lists
        average_metrics['zncc'] = []
        average_metrics['ssim'] = []

        for i in range(7):
            if results['zncc'][i]:
                average_zncc = sum(results['zncc'][i]) / len(results['zncc'][i])
            else:
                average_zncc = 0.0
            average_metrics['zncc'].append(average_zncc)

            if results['ssim'][i]:
                average_ssim = sum(results['ssim'][i]) / len(results['ssim'][i])
            else:
                average_ssim = 0.0
            average_metrics['ssim'].append(average_ssim)

        return average_metrics
    def eval(self):
        self.model.eval()
        results = {
            'maxrel': [],
            'zncc' : [[] for _ in range(7)],
            'ssim': [[] for _ in range(7)]
        }
        matrics_calculation = EvaluationMetrics(postprocess_fn=self.postProcessing)
        for bidx, (data, target) in enumerate(self.test_dataloader):
            data, target = Variable(data.squeeze(0)).to(self.device), Variable(target.squeeze(0)).to(self.device)
            with torch.no_grad():
                pred = self.model(data)

            metrics = matrics_calculation.evaluate(target, pred)
            results['maxrel'].append(metrics['maxrel'].item())
            for i in range(7):
                results['zncc'][i].append(metrics['zncc'][i].item())
                results['ssim'][i].append(metrics['ssim'][i].item())

            # self.logger.info(f"Metrics for batch {bidx}: {metrics}")
        average_metrics = self.get_average_metrics(results)
        return average_metrics


    def test(self):
        self.model.eval()
        P = []
        T = []
        L = []
        SL = []
        MSE = []
        for bidx, samples in enumerate(self.test_dataloader):
            data, target = Variable(samples[0].squeeze(0)).to(self.device), Variable(samples[1].squeeze(0)).to(self.device)
            start_time = time.time()
            with torch.no_grad():
                pred = self.model(data)
                end_time = time.time()
            self.logger.info(f'computation time for batch {bidx}: {end_time - start_time:.4f}')
            soble_loss,mse = self.loss_object(target, pred)
            loss = soble_loss + mse

            P.append(pred.detach())
            T.append(target.detach())
            L.append(loss.detach())
            SL.append(soble_loss.detach())
            MSE.append(mse.detach())
        P = torch.cat(P, dim=0)
        T = torch.cat(T,dim=0)
        L = torch.stack(L)
        SL = torch.stack(SL)
        MSE = torch.stack(MSE)
        return P,T,torch.mean(L),torch.mean(SL),torch.mean(MSE)
    
    def run(self):
        for epoch in tqdm(range(self.config['model']['epochs'])):
            self.logger.info(f'epoch:{epoch+1} starts train')
            train_loss,train_feature,train_mse,tr_p,tr_t = self.train()
            torch.cuda.empty_cache()  # Clear cache after training
            
            val_p,val_t,val_loss, val_feature,val_mse = self.test()
            matrics_calculation = EvaluationMetrics(postprocess_fn=self.postProcessing)
            metrics = matrics_calculation.evaluate(val_p, val_t)

            torch.cuda.empty_cache()  # Clear cache after evaluation

            self.log_metrics(epoch, train_loss, train_feature,train_mse, 
                                    val_loss, val_feature,val_mse,metrics['maxrel'],metrics['ssim'])
            if self.scheduler:
                self.scheduler.step()
            # if patience_counter >= self.config['patience']:
            #     print("Early stopping triggered")
            #     break
        
    def log_metrics(self, epoch, train_loss, train_feature,train_mse,val_loss, val_feature,val_mse,vl_maxrel,vl_ssim):
        self.logger.info(f'epoch:{epoch}, train_loss:{train_loss.cpu().item()}, '
                 f'train_feature:{train_feature.cpu().item()}, '
                 f'train_mse:{train_mse.cpu().item()}, '
                 f'val_loss:{val_loss.cpu().item()}, '
                 f'val_feature:{val_feature.cpu().item()}, '
                 f'val_mse:{val_mse.cpu().item()}, '
                 f'val_maxrel:{vl_maxrel.cpu().item()}, '
                 f'val_ssim:{vl_ssim.cpu()}')
        model_path = os.path.join(self.config['output']['save_path'], self.config['output']['model_file'])
        torch.save(self.model.state_dict(), model_path)
        # if vl_maxrel < self.best_loss:
        #     self.best_loss = vl_maxrel
        #     self.patience_counter = 0
        #     model_path = os.path.join(self.config['output']['save_path'], self.config['output']['model_name'])
        #     torch.save(self.model.state_dict(), model_path)
        # else:
        #     self.patience_counter += 1
        
    def save(self,save_hist=True):
        model_path = os.path.join(self.config['output']['save_path'], self.config['output']['model_file'])
        self.model.load_state_dict(torch.load(model_path,map_location=self.device))
        pred, target, test_loss,test_feature, test_mse = self.test()
        assert len(target) == len(pred), "Targets and predictions must have the same length"
        # Save the training history
        if save_hist:
            history_path = os.path.join(self.config['output']['save_path'],self.config['output']['pkl_file'])
            with open(history_path, "wb") as pickle_file:
                pickle.dump({
                    "predictions": pred,
                    "targets": target
                }, pickle_file)
            self.logger.info(f'saved {history_path}!\n')
            self.logger.info('Test Epoch: {} Loss: {:.4f}\n'.format(self.config["model"]["epochs"], test_loss.cpu().item()))
            self.logger.info('best loss', self.best_loss)
            original_target,original_pred = self.postProcessing(target.cpu().item()),self.postProcessing(pred.cpu().item())
            self.logger.info(f'relative loss {MaxRel(original_target,original_pred):.5f}%')

            avg_ssim = calculate_ssim_batch(target.cpu().item(),pred.cpu().item())
            for freq in range(len(avg_ssim)):
                self.logger.info(f'frequency {freq + 1} has ssim {avg_ssim[freq]:.4f}')
    
    def postProcessing(self,y):
        if self.config['dataset']['name'] =='mulfreq':
            max_   = self.dataset_stats['intensity']['max']
            min_   = self.dataset_stats['intensity']['min']
            y = y*(max_ - min_)+min_
        else:
            min_   = self.dataset_stats['intensity']['min']
            median = self.dataset_stats['intensity']['median']
            y = y*median + min_
        y = torch.exp(y)
        return y

 ### train step ###       
class Trainer:
    def __init__(self,ddp_model,train_dataloader,test_dataloader,optimizer,loss_object,config,rank,device,world_size,logger,scheduler):
        print("Initializing Trainer...")
        start_time = time.time()
        self.ddp_model        = ddp_model
        self.train_dataloader = train_dataloader
        self.test_dataloader  = test_dataloader

        self.config           = config
        self.rank             = rank # global rank, world_size -1, distinguish each process
        self.device       = device
        self.world_size       = world_size # num GPUs
        # Define the optimizer for the DDP model
        self.optimizer        = optimizer
        self.scheduler        = scheduler
        self.loss_object      = loss_object
        self.logger           = logger

        self.best_loss = float('inf')
        self.patience_counter = 0

        statistics_path = config['dataset']['statistics']['path']
        statistics_values = config['dataset']['statistics']['values']
        self.dataset_stats  = load_statistics(statistics_path,statistics_values)
        init_duration = time.time() - start_time
        print(f"Initialization completed in {init_duration:.2f} seconds.")
                                       
    def train(self):
        total_loss = 0.
        total_soble = 0.
        total_mse  = 0.
        self.ddp_model.train()
        
        for bidx, samples in enumerate(self.train_dataloader):
            # data, target = samples[0].squeeze(0).to(self.local_rank), samples[1].squeeze(0).squeeze(dim=1).to(self.local_rank)
            data, target,_ = samples[0].squeeze(0).to(self.device), samples[1].squeeze(0).to(self.device),samples[2]
            self.optimizer.zero_grad()
            output = self.ddp_model(data)
            soble_loss,mse = self.loss_object(target, output)
            loss = soble_loss + mse
           
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach()
            total_soble += soble_loss.detach()
            total_mse += mse.detach()
            if bidx % 50 == 0:
                self.logger.info(f'Train batch [{bidx}/{len(self.train_dataloader)}]: freq loss: {soble_loss:.4f}, mse:{mse:.4f}, total loss:{loss:.4f}',gpu_rank=self.rank)
            # self.logger.info(f'aggreate loss for batch {bidx}')
        epoch_loss = total_loss / len(self.train_dataloader)  # divide number of batches
        epoch_soble = total_soble / len(self.train_dataloader)
        epoch_mse = total_mse / len(self.train_dataloader)
        return epoch_loss,epoch_soble,epoch_mse
    def test(self):
        self.ddp_model.eval()
      
        total_loss = 0.
        total_mse_loss, total_freq_loss = 0.0, 0.0

        relative_loss_epoch = []
        zncc_epoch = []
        ssim_epoch = []
        matrics_calculation = EvaluationMetrics(postprocess_fn=self.postProcessing)
        with torch.no_grad():
            for bidx, samples in enumerate(self.test_dataloader):
                # data, target = samples[0].squeeze(0).to(self.local_rank), samples[1].squeeze(0).squeeze(dim=1).to(self.local_rank)
                data, target,_ = samples[0].squeeze(0).to(self.local_rank), samples[1].squeeze(0).to(self.local_rank),samples[2]
                pred = self.ddp_model(data)
                soble_loss,mse = self.loss_object(target, pred)

                loss = soble_loss + mse
                # (scalar): local median value, [7 elements]
                metrics = matrics_calculation.evaluate(target, pred)

                relative_loss_epoch.append(metrics['maxrel'])
                zncc_epoch.append(metrics['zncc'])
                ssim_epoch.append(metrics['ssim'])
                batch_loss += loss.sum()
                batch_mse += mse.sum()
                batch_freq_loss += soble_loss.sum()
                if bidx % 50 == 0:
                    self.logger.info(f'Test batch [{bidx}/{len(self.test_dataloader)}]: freq loss: {soble_loss:.4f}, mse:{mse:.4f}, total loss:{loss:.4f}',gpu_rank=self.rank)
        if dist.is_initialized():
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = total_loss_tensor.item()

            total_mse_tensor = torch.tensor(total_mse_loss, device=self.device)
            dist.all_reduce(total_mse_tensor, op=dist.ReduceOp.SUM)
            total_mse_loss = total_mse_tensor.item()

            total_freq_tensor = torch.tensor(total_freq_loss, device=self.device)
            dist.all_reduce(total_freq_tensor, op=dist.ReduceOp.SUM)
            total_freq_loss = total_freq_tensor.item()

        val_epoch_loss = total_loss / len(self.test_dataloader)
        val_epoch_mse  = total_mse_loss / len(self.test_dataloader)
        val_epoch_soble = total_freq_loss/len(self.test_dataloader)

        # Aggregate other metrics properly
        if dist.is_initialized():
            val_zncc = self.aggregate_metric(torch.stack(zncc_epoch), reduction="mean", gather=True)
            val_ssim = self.aggregate_metric(torch.stack(ssim_epoch), reduction="mean", gather=True)
            val_maxrel = self.aggregate_metric(torch.stack(relative_loss_epoch), reduction="median", gather=True)
        else:
            val_zncc = torch.mean(torch.stack(zncc_epoch))
            val_ssim = torch.mean(torch.stack(ssim_epoch))
            val_maxrel = torch.median(torch.stack(relative_loss_epoch))

        # maxrel in certain rank
        val_maxrel = torch.stack(relative_loss_epoch)
        # ssim in certain rank
        val_zncc   = torch.stack(zncc_epoch)
        val_ssim   = torch.stack(ssim_epoch)
        
        results = {
            'val_loss': val_epoch_loss,
            'val_feature':val_epoch_soble,
            'val_mse':val_epoch_mse,
            'maxrel':val_maxrel,
            'zncc': val_zncc,
            'ssim': val_ssim
        }

        return results
    def run(self):
        stop_signal = torch.tensor([0], device=self.local_rank)
        for epoch in tqdm(range(self.config['model']['epochs']), disable=self.rank != 0):  # Disable tqdm progress bar except for rank 0
            epoch_loss,epoch_soble,epoch_mse = self.train()
            results = self.test() #todo 
            torch.cuda.empty_cache()  # Clear cache after training            

            # Update history on master process
            if torch.distributed.get_rank() == 0:
                record_values = {
                    'train_loss':epoch_loss.cpu().item(),
                    'train_feature':epoch_soble.cpu().item(),
                    'train_mse':epoch_mse.cpu().item(),
                    'val_loss':results['val_loss'].cpu().item(),
                    'val_feature':results['val_feature'].cpu().item(),
                    'val_mse':results['val_mse'].cpu().item(),
                    'val_maxrel':results['maxrel'].cpu().item(),
                    'val_zncc':results['zncc'].cpu().tolist(),
                    'val_ssim': results['ssim'].cpu().tolist()
                }
                self.log_metrics(epoch, record_values)
            if self.scheduler:
                self.scheduler.step()

    def log_metrics(self, epoch, record_values):
        self.logger.info(f"epoch:{epoch}, train_loss:{record_values['train_loss']},"
                            f"train_feature:{record_values['train_feature']},"
                            f"train_mse:{record_values['train_mse']},"
                            f"val_loss:{record_values['val_loss']},"
                            f"val_feature:{record_values['val_feature']},"
                            f"val_mse:{record_values['val_mse']},"
                            f"val_maxrel:{record_values['val_maxrel']},"
                            f"val_zncc:{record_values['val_zncc']},"
                            f"val_ssim:{record_values['val_ssim']}")
        model_path = os.path.join(self.config['output']['save_path'], self.config['output']['model_file'])
        torch.save( {
            'epoch': epoch,
            'model_state_dict':self.ddp_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'schedular_state_dict':self.scheduler.state_dict() if self.scheduler is not None else None,
            'current_maxrel':record_values['val_maxrel'],
        },model_path)

        # self.logger.info(f"Model saved at epoch {epoch}")
        if record_values['val_maxrel'] < self.best_loss:
            self.best_loss = record_values['val_maxrel']
            # self.patience_counter = 0
            model_path = os.path.join(self.config['output']['save_path'], 'best_sofar_'+self.config['output']['model_file'])
            torch.save( {
                'epoch': epoch,
                'model_state_dict':self.ddp_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'schedular_state_dict':self.scheduler.state_dict() if self.scheduler is not None else None,
                'current_maxrel':record_values['val_maxrel'],
            },model_path)
            self.logger.info(f"Model saved at epoch {epoch}\n")

    def postProcessing(self,y):
        if self.config['dataset']['name'] =='mulfreq':
            max_   = self.dataset_stats['intensity']['max']
            min_   = self.dataset_stats['intensity']['min']
            y = y*(max_ - min_)+min_
        else:
            min_   = self.dataset_stats['intensity']['min']
            median = self.dataset_stats['intensity']['median']
            y = y*median + min_
        y = torch.exp(y)
        return y
    def aggregate_metric(self,metric_tensor,reduction='mean',gather=False):
        '''
        Aggregates a given metric tensor across all DDP ranks.

        Args:
            metric_tensor (torch.Tensor): The tensor to aggregate.
            reduction (str): "mean" for averaging, "sum" for summation, "median" for median aggregation.
            gather (bool): Whether to use `all_gather` instead of `all_reduce` (for full dataset statistics).
        Returns:
            torch.Tensor: Aggregated result (scalar).
        '''
        if gather:
            # Gather values from all processes
            gathered_tensors = [torch.zeros_like(metric_tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered_tensors, metric_tensor)
            gathered_tensors = torch.cat(gathered_tensors, dim=0)  # Combine all values

            if reduction == "mean":
                return torch.mean(gathered_tensors)
            elif reduction == "median":
                return torch.median(gathered_tensors)
        else:
            # Directly reduce using sum or mean
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
            return metric_tensor / self.world_size
    def per_sample_eval(self):
        self.ddp_model.eval()

        maxrel_data = []
        
        matrics_calculation = EvaluationMetrics(postprocess_fn=self.postProcessing)
        for bidx, (data, target,files) in enumerate(self.test_dataloader):
            data, target = Variable(data.squeeze(0)).to(self.local_rank), Variable(target.squeeze(0)).to(self.local_rank)
            with torch.no_grad():
                pred = self.ddp_model(data)
            if bidx % 50 == 0:
                self.logger.info(f'Test batch [{bidx}/{len(self.train_dataloader)}]')

            maxrels_batch = matrics_calculation.calculate_maxrel(target, pred)
            for i, (maxrel, file) in enumerate(zip(maxrels_batch, files)):
                maxrel_data.append({
                    'maxrel': maxrel.item(),  # Convert tensor to a Python float
                    'file': file  # Store the corresponding file information
                })

        return maxrel_data
