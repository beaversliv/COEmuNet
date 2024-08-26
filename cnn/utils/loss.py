import torch
import torch.nn as nn
import numpy as np
import sys
# Structural Similarity Index Measure (SSIM) 
from skimage.metrics import structural_similarity as ssim
from collections import namedtuple
import torchvision.models as models
# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft
class SobelLoss(nn.Module):
    '''
    sobelloss for all freq
    '''
    def __init__(self):
        super(SobelLoss, self).__init__()
        # Define Sobel edge detection filters for a 2D convolution
        self.conv_op_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_op_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
        # Initialize filters for edge detection (Sobel kernels)
        sobel_kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view((1, 1, 3, 3))
        sobel_kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view((1, 1, 3, 3))
        
        self.conv_op_x.weight.data = sobel_kernel_x
        self.conv_op_y.weight.data = sobel_kernel_y
        
        # Do not update weights during training
        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False

    def forward(self, pred, target):
        # Check if input is 4D or 5D and reshape appropriately
        if pred.dim() == 5:  # Input is 5D
            # Combine batch and sequence dimensions: (batch, 1, 31, 64, 64) -> (batch*31, 1, 64, 64)
            batch_size, channels, depth, height, width = pred.shape
            pred_reshaped = pred.view(batch_size * depth, channels, height, width)
            target_reshaped = target.view(batch_size * depth, channels, height, width)
        elif pred.dim() == 4:  # Input is 4D
            # No need to reshape if input is already 4D: (batch, 1, 64, 64)
            pred_reshaped = pred
            target_reshaped = target
        else:
            raise ValueError('Input tensor must be 4D or 5D')

        # Apply the Sobel filter to the reshaped predicted and target images
        edge_pred_x = self.conv_op_x(pred_reshaped)
        edge_pred_y = self.conv_op_y(pred_reshaped)
        edge_target_x = self.conv_op_x(target_reshaped)
        edge_target_y = self.conv_op_y(target_reshaped)
        
        # Calculate loss as the L1 difference between edges
        loss_x = nn.functional.l1_loss(edge_pred_x, edge_target_x)
        loss_y = nn.functional.l1_loss(edge_pred_y, edge_target_y)
        
        # Combine losses for both x and y edge detections
        loss = loss_x + loss_y
        
        return loss
class FreqMse(nn.Module):
    def __init__(self,alpha=0.8,beta=0.2):
        super(FreqMse, self).__init__()
        self.alpha     = alpha
        self.beta      = beta

    def calculate_freq_loss(self,target,pred):
        # ensure FFT takes float32
        target = target.to(torch.float32)
        pred   = pred.to(torch.float32)
        target_freq = torch.fft.fft2(target)
        pred_freq = torch.fft.fft2(pred)
        return torch.mean(torch.abs(target_freq - pred_freq))

    def forward(self, target,pred):
        # Calculate the edge loss
        loss_edge = self.calculate_freq_loss(target, pred)
        # Calculate the MSE loss
        loss_mse = nn.functional.mse_loss(pred, target)
        # Combine the losses
        loss_combined = self.alpha * loss_edge + self.beta * loss_mse 
        return self.alpha * loss_edge,self.beta * loss_mse

def MaxRel(original_target,original_pred):
    return np.mean( np.abs(original_target-original_pred) / np.max(original_target, axis=1,keepdims=True)) * 100

def calculate_ssim_batch(target,pred):
    if pred.ndim == 5:  # Input is 5D
        # Combine batch and sequence dimensions: (batch, channels, depth, height, width) -> (batch*depth, channels, height, width)
        batch_size, channels, depth, height, width = pred.shape
        pred_reshaped = pred.reshape(batch_size * channels, depth, height, width)
        target_reshaped = target.reshape(batch_size * channels, depth, height, width)
    elif pred.ndim == 4:  # Input is 4D
        # No need to reshape if input is already 4D: (batch, channels, height, width)
        pred_reshaped = pred
        target_reshaped = target
    else:
        raise ValueError(f"Unsupported input dimensions: {pred.ndim}")


    # Calculate SSIM for each image in the batch
    num_samples, num_freqs, height, width = target_reshaped.shape
    ssim_scores = []

    for freq in range(num_freqs):
        freq_ssim = []
        for i in range(num_samples):
            # Extract the specific frequency images from target and prediction
            pred_img = pred_reshaped[i, freq, :, :]
            target_img = target_reshaped[i, freq, :, :]
            
            # Compute SSIM for this frequency channel and sample
            score = ssim(target_img, pred_img, data_range=target_img.max() - target_img.min())
            freq_ssim.append(score)

        # Average SSIM across all samples for this frequency
        avg_ssim = np.mean(freq_ssim)
        ssim_scores.append(avg_ssim)

    return ssim_scores
