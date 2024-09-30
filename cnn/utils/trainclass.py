import torch
import torch.distributed as dist
from torch.autograd import Variable

from utils.loss     import calculate_ssim_batch,MaxRel
from utils.utils    import Logging
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
def load_statistics(statistics_path,statistics_values):
        statistics = {}
        with h5.File(statistics_path, 'r') as f:
            for value in statistics_values:
                feature = value['name']
                stats_to_read = value['stats']
                statistics[feature] = {stat: f[feature][stat][()] for stat in stats_to_read}
        return statistics

class Trainer:
    def __init__(self,model,loss_object,optimizer,train_dataloader,test_dataloader,config,device,scheduler=None):
        print("Initializing Trainer...")
        start_time = time.time()
        self.model = model
        self.loss_object = loss_object
        self.optimizer   = optimizer
        self.scheduler        = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader  = test_dataloader
        self.config = config
        self.device = device
        logger = Logging(config['output']['save_path'], config['output']['log_file'])
        self.logger = logger
        self.best_loss = float('inf')  # Initialize best loss
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
        P = torch.cat(P, dim=0)
        T = torch.cat(T,dim=0)

        return epoch_loss,epoch_soble,epoch_mse,P,T
    
    def test(self):
        self.model.eval()
        P = []
        T = []
        L = []
        SL = []
        MSE = []
        for bidx, samples in enumerate(self.test_dataloader):
            data, target = Variable(samples[0].squeeze(0)).to(self.device), Variable(samples[1].squeeze(0)).to(self.device)
            with torch.no_grad():
                pred = self.model(data)
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
            torch.cuda.empty_cache()  # Clear cache after evaluation
            self.log_metrics(epoch, train_loss, train_feature,train_mse,tr_p,tr_t, 
                                    val_loss, val_feature,val_mse,val_p, val_t)
            if self.scheduler:
                self.scheduler.step()
            # if patience_counter >= self.config['patience']:
            #     print("Early stopping triggered")
            #     break
        
    def log_metrics(self, epoch, train_loss, train_feature,train_mse,tr_p,tr_t,val_loss, val_feature,val_mse,val_p, val_t):
        tr_p = tr_p.cpu().numpy()
        tr_t = tr_t.cpu().numpy()
        val_p = val_p.cpu().numpy()
        val_t = val_t.cpu().numpy()

        original_trt,original_trp = self.postProcessing(tr_t,tr_p)
        original_vlt,original_vlp = self.postProcessing(val_t,val_p) 

        train_maxrel = MaxRel(original_trt,original_trp)
        tr_ssim = calculate_ssim_batch(tr_p, tr_t)
        vl_maxrel = MaxRel(original_vlt,original_vlp)
        vl_ssim = calculate_ssim_batch(val_p, val_t)

        self.logger.info(f'epoch:{epoch}, train_loss:{train_loss.cpu().item()}, '
                 f'train_feature:{train_feature.cpu().item()}, '
                 f'train_mse:{train_mse.cpu().item()}, '
                 f'train_maxrel:{train_maxrel}, '
                 f'train_ssim:{tr_ssim}, '
                 f'val_loss:{val_loss.cpu().item()}, '
                 f'val_feature:{val_feature.cpu().item()}, '
                 f'val_mse:{val_mse.cpu().item()}, '
                 f'val_maxrel:{vl_maxrel}, '
                 f'val_ssim:{vl_ssim}')
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
            original_target,original_pred = self.postProcessing(target.cpu().item(),pred.cpu().item())
            self.logger.info(f'relative loss {MaxRel(original_target,original_pred):.5f}%')

            avg_ssim = calculate_ssim_batch(target.cpu().item(),pred.cpu().item())
            for freq in range(len(avg_ssim)):
                self.logger.info(f'frequency {freq + 1} has ssim {avg_ssim[freq]:.4f}')
    def postProcessing(self, target,pred):
        def transformation(y):
            if self.config['dataset']['name'] =='mulfreq':
                max_   = self.dataset_stats['intensity']['max']
                min_   = self.dataset_stats['intensity']['min']
                y = y*(max_ - min_)+min_
            else:
                min_   = self.dataset_stats['intensity']['min']
                median = self.dataset_stats['intensity']['median']
                y = y*median + min_
            y = np.exp(y)
            return y
        return transformation(target), transformation(pred)

 ### train step ###       
class ddpTrainer:
    def __init__(self,ddp_model,train_dataloader,test_dataloader,optimizer,loss_object,config,rank,local_rank,world_size,scheduler=None):
        print("Initializing Trainer...")
        start_time = time.time()
        self.ddp_model        = ddp_model
        self.train_dataloader = train_dataloader
        self.test_dataloader  = test_dataloader

        self.config           = config
        self.rank             = rank # global rank, world_size -1, distinguish each process
        self.local_rank       = local_rank
        self.world_size       = world_size # num GPUs
        # Define the optimizer for the DDP model
        self.optimizer        = optimizer
        self.scheduler        = scheduler
        self.loss_object      = loss_object
        self.logger           = Logging(config['output']['save_path'], config['output']['log_file'])

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
            data, target = samples[0].squeeze(0).to(self.local_rank), samples[1].squeeze(0).to(self.local_rank)
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
        P = []
        T = []
        L = []
        SL = []
        MSE = []
        with torch.no_grad():
            for bidx, samples in enumerate(self.test_dataloader):
                data, target = samples[0].squeeze(0).to(self.local_rank), samples[1].squeeze(0).to(self.local_rank)
                pred = self.ddp_model(data)
                soble_loss,mse = self.loss_object(target, pred)

                loss = soble_loss + mse
                P.append(pred.detach())
                T.append(target.detach())
                L.append(loss.detach())
                SL.append(soble_loss.detach())
                MSE.append(mse.detach())
                if bidx % 50 == 0:
                    self.logger.info(f'Test batch [{bidx}/{len(self.test_dataloader)}]: freq loss: {soble_loss:.4f}, mse:{mse:.4f}, total loss:{loss:.4f}',gpu_rank=self.rank)
        P = torch.cat(P, dim=0)
        T = torch.cat(T,dim=0)
        L = torch.stack(L)
        SL = torch.stack(SL)
        MSE = torch.stack(MSE)
        return P,T,torch.mean(L),torch.mean(SL),torch.mean(MSE)
    def run(self):
        stop_signal = torch.tensor([0], device=self.local_rank)
        for epoch in tqdm(range(self.config['model']['epochs']), disable=self.rank != 0):  # Disable tqdm progress bar except for rank 0
            # self.logger.info(f'epochs {epoch+1}')
            epoch_loss,epoch_soble,epoch_mse = self.train()
            pred, target, val_loss,val_feature,val_mse = self.test() #todo 
       
            torch.cuda.empty_cache()  # Clear cache after training            
            # Aggregate losses
            dist.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
            val_epoch_loss = val_loss/self.world_size

            dist.all_reduce(val_feature, op=torch.distributed.ReduceOp.SUM)
            val_epoch_soble = val_feature/self.world_size
            dist.all_reduce(val_mse, op=torch.distributed.ReduceOp.SUM)
            val_epoch_mse = val_mse/self.world_size
           
            # self.logger.info(f'{self.rank} finish gartering')
            # Update history on master process
            if torch.distributed.get_rank() == 0:
    
                # self.logger.info('test start')
                self.log_metrics(epoch, epoch_loss, epoch_soble, epoch_mse, val_epoch_loss, val_epoch_soble,val_epoch_mse, pred, target)
            if self.scheduler:
                self.scheduler.step()
            #     if stop_training:
            #         stop_signal[0] = 1  # Set stop signal

            # # Broadcast the stop signal to all processes
            # dist.broadcast(stop_signal, src=0)
            # if stop_signal[0].item() == 1:
            #     break

    def log_metrics(self, epoch, aggregated_epoch_loss, aggregated_epoch_soble, aggregated_epoch_mse, aggregated_val_loss, val_feature,val_mse, all_preds, all_targets):
       
        # Ensure targets and predictions have the same length
        assert len(all_targets) == len(all_preds), "Targets and predictions must have the same length"
        all_targets = all_targets.cpu().numpy()
        all_preds   = all_preds.cpu().numpy()
        original_target = self.postProcessing(all_targets)
        original_pred = self.postProcessing(all_preds)
        relative_loss = MaxRel(original_target, original_pred)
        
        avg_ssim = calculate_ssim_batch(all_targets, all_preds)
        self.logger.info(f'epoch:{epoch}, train_loss:{aggregated_epoch_loss.cpu().item()},'
                            f'train_feature:{aggregated_epoch_soble.cpu().item()},'
                            f'train_mse:{aggregated_epoch_mse.cpu().item()},'
                            f'val_loss:{aggregated_val_loss.cpu().item()},'
                            f'val_feature:{val_feature.cpu().item()},'
                            f'val_mse:{val_mse.cpu().item()},'
                            f'val_maxrel:{relative_loss},'
                            f'val_ssim:{avg_ssim}\n')
        
        model_path = os.path.join(self.config['output']['save_path'], {self.config['output']['model_file']})
        torch.save( {
            'epoch': epoch,
            'model_state_dict':self.ddp_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_maxrel':relative_loss,
        },model_path)

        self.logger.info(f"Model saved at epoch {epoch}")
        if relative_loss < self.best_loss:
            self.best_loss = relative_loss
            # self.patience_counter = 0
            model_path = os.path.join(self.config['output']['save_path'], 'best_sofar_'+self.config['output']['model_file'])
            torch.save( {
                'epoch': epoch,
                'model_state_dict':self.ddp_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'current_maxrel':relative_loss,
            },model_path)
            self.logger.info(f"Model saved at epoch {epoch}")
        # else:
        #     self.patience_counter += 1
        # if patience_counter >= self.config['patience']:
        #     print("Early stopping triggered")
        #     return True
        # else:
        #     return False
    def eval(self):
        dist.barrier()
        if self.rank == 0:
            self.ddp_model.eval()
            P = []
            T = []
            with torch.no_grad():
                for bidx, samples in enumerate(self.test_dataloader):
                    data, target = Variable(samples[0]).to(self.local_rank), Variable(samples[1]).to(self.local_rank)
                    pred = self.ddp_model(data)
                    P.append(pred.detach().cpu().numpy())
                    T.append(target.detach().cpu().numpy())
            P = np.vstack(P)
            T = np.vstack(T)
            original_target = self.postProcessing(P)
            original_pred = self.postProcessing(T)
            print(f'initial relative loss {MaxRel(original_target,original_pred):.5f}%')

    def save(self, save_hist=False):
        if self.rank == 0:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
            model_path = os.path.join(self.config['output']['save_path'], self.config['output']['model_file'])
            self.ddp_model.load_state_dict(torch.load(model_path, map_location=map_location))

            all_preds, all_targets, test_loss,test_feature,test_mse = self.test()
            all_targets = all_targets.cpu().numpy()
            all_preds   = all_preds.cpu().numpy()
            
            assert len(all_preds) == len(all_preds), "Targets and predictions must have the same length"
            # Save the training history
            if save_hist:
                history_path = os.path.join(self.config['output']['save_path'],self.config['output']['pkl_file'])
                with open(history_path, "wb") as pickle_file:
                    pickle.dump({
                        "predictions": all_preds[:20],
                        "targets": all_targets[:20]
                    }, pickle_file)
                print(f'saved {history_path}!\n')
            self.logger.info('Test Epoch: {} Loss: {:.4f}\n'.format(self.config["model"]["epochs"], test_loss.cpu().item()))
            self.logger.info('best loss', self.best_loss)
            original_target = self.postProcessing(all_targets)
            original_pred = self.postProcessing(all_preds)
            self.logger.info(f'relative loss {MaxRel(original_target,original_pred):.5f}%')

            avg_ssim = calculate_ssim_batch(all_targets,all_preds)
            for freq in range(len(avg_ssim)):
                self.logger.info(f'frequency {freq + 1} has ssim {avg_ssim[freq]:.4f}')

            # # Plot and save images
            # img_plt(all_targets, all_preds, path=path)
    def postProcessing(self,y):
        if self.config['dataset']['name'] =='mulfreq':
            max_   = self.dataset_stats['intensity']['max']
            min_   = self.dataset_stats['intensity']['min']
            y = y*(max_ - min_)+min_
        else:
            min_   = self.dataset_stats['intensity']['min']
            median = self.dataset_stats['intensity']['median']
            y = y*median + min_
        y = np.exp(y)
        return y