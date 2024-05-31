import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from tqdm                 import tqdm
import numpy as np

class Trainer:
    def __init__(self,model,loss_object,optimizer,train_dataloader,test_dataloader,config,device):
        self.model = model
        self.loss_object = loss_object
        self.optimizer   = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader  = test_dataloader
        self.config = config
        self.device = device
    
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
            total_loss += loss.detach().cpu().numpy()
        epoch_loss = total_loss / len(self.train_dataloader)  # divide number of batches

        return epoch_loss

    
    def test(self):
        self.model.eval()
        P = []
        T = []
        L = []
        for bidx, samples in enumerate(self.test_dataloader):
            data, target = Variable(samples[0]).to(self.device), Variable(samples[1]).to(self.device)
            pred = self.model(data)
            loss = self.loss_object(target, pred)
            
            P.append(pred.detach().cpu().numpy())
            T.append(target.detach().cpu().numpy())
            L.append(loss.detach().cpu().numpy())
        P = np.vstack(P)
        T = np.vstack(T)
        return P,T,np.mean(L)
    def run(self):
        tr_losses = []
        vl_losses = []
        for epoch in tqdm(range(self.config['epochs'])):
            epoch_loss = self.train()
            torch.cuda.empty_cache()  # Clear cache after training
            
            _, _, val_loss = self.test()
            torch.cuda.empty_cache()  # Clear cache after evaluation
            tr_losses.append(epoch_loss)
            vl_losses.append(val_loss)
            print('Train Epoch: {}/{} Loss: {:.4f}'.format(
                    epoch, self.config['epochs'], epoch_loss))
            print('Test Epoch: {}/{} Loss: {:.4f}\n'.format(
                epoch, self.config['epochs'], val_loss))
            
        return tr_losses, vl_losses

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
        logger = Logging('/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/results', 'log_file0')
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
    def log_metrics(self, epoch, aggregated_epoch_loss, aggregated_val_loss, all_preds, all_targets):
       
        # Ensure targets and predictions have the same length
        assert len(all_targets) == len(all_preds), "Targets and predictions must have the same length"
        
        original_target = self.postProcessing(all_targets)
        original_pred = self.postProcessing(all_preds)
        relative_loss = self.relativeLoss(original_target, original_pred)
        
        avg_ssim = calculate_ssim_batch(all_targets, all_preds)

        self.logger.write_log_metrics(
            epoch = epoch,
            train_loss = aggregated_epoch_loss.cpu().item(),
            val_loss = aggregated_val_loss.cpu().item(),
            relative_loss = relative_loss,
            ssim_values = avg_ssim)
        if avg_ssim[0] > 0.98:
            return True # Indicate that training should stop
        else:
            return False
    def gather_predictions_targets(self, pred, target):
        dist.barrier()

        # Gather predictions and targets from all GPUs
        gathered_preds = [torch.zeros_like(pred) for _ in range(self.world_size)]
        gathered_targets = [torch.zeros_like(target) for _ in range(self.world_size)]

        torch.distributed.all_gather(gathered_preds, pred)
        torch.distributed.all_gather(gathered_targets, target)

        all_preds = torch.cat(gathered_preds, dim=0).cpu().numpy()
        all_targets = torch.cat(gathered_targets, dim=0).cpu().numpy()

        return all_preds, all_targets
    def run(self):
        history = {'train_loss': [], 'val_loss': []} 
        stop_signal = torch.tensor([0], device=self.rank)
        for epoch in tqdm(range(self.config['epochs']), disable=self.rank != 0):  # Disable tqdm progress bar except for rank 0
            epoch_loss = self.train()
            torch.cuda.empty_cache()  # Clear cache after training
            
            pred, target, val_loss = self.test()
            torch.cuda.empty_cache()  # Clear cache after evaluation

            # Aggregate losses
            dist.all_reduce(epoch_loss, op=torch.distributed.ReduceOp.SUM)
            aggregated_epoch_loss = epoch_loss/self.world_size

            dist.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
            aggregated_val_loss = val_loss / self.world_size
            gathered_preds, gathered_targets = self.gather_predictions_targets(pred, target)
            # Update history on master process
            if self.rank == 0:
                history['train_loss'].append(aggregated_epoch_loss.cpu().item())
                history['val_loss'].append(aggregated_val_loss.cpu().item())
                # print(f'Train Epoch: {epoch}/{self.config["epochs"]} Loss: {aggregated_epoch_loss.cpu().item():.4f}')
                # print(f'Test Epoch: {epoch}/{self.config["epochs"]} Loss: {aggregated_val_loss.cpu().item():.4f}\n')
                stop_training = self.log_metrics(epoch, aggregated_epoch_loss, aggregated_val_loss, gathered_preds, gathered_targets)
                if stop_training:
                    stop_signal[0] = 1  # Set stop signal

            # Broadcast the stop signal to all processes
            dist.broadcast(stop_signal, src=0)
            if stop_signal[0].item() == 1:
                break

        return history
 
    def save(self, model_path, history_path,history,path, world_size):
        pred, target, test_loss = self.test()
        dist.barrier()
        
        all_preds, all_targets = self.gather_predictions_targets(pred, target)

        # Aggregate test loss
        torch.distributed.all_reduce(test_loss, op=torch.distributed.ReduceOp.SUM)
        aggregated_loss = test_loss / self.world_size

        if self.rank == 0:
            # Save only the model parameters
            torch.save(self.ddp_model.module.state_dict(), model_path)
            print('saved model!\n')
       
            assert len(target) == len(pred), "Targets and predictions must have the same length"
            # Save the training history
            with open(history_path, "wb") as pickle_file:
                pickle.dump({
                    "history": history,
                    "predictions": all_preds,
                    "targets": all_targets
                }, pickle_file)
            print(f'saved {history_path}!\n')
            print('Test Epoch: {} Loss: {:.4f}\n'.format(self.config["epochs"], aggregated_loss.cpu().item()))
            original_target = self.postProcessing(all_targets)
            original_pred = self.postProcessing(all_preds)
            print(f'relative loss {self.relativeLoss(original_target,original_pred):.5f}%')

            avg_ssim = calculate_ssim_batch(all_targets,all_preds)
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