import torch
import torch.nn as nn
import numpy as np
import sys
# Structural Similarity Index Measure (SSIM) 
from skimage.metrics import structural_similarity as ssim
from collections import namedtuple
import torchvision.models as models
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

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
    def __init__(self,alpha=0.8):
        super(FreqMse, self).__init__()
        self.alpha     = alpha

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
        loss_combined = self.alpha * loss_edge + (1-self.alpha) * loss_mse 
        return self.alpha * loss_edge,(1-self.alpha) * loss_mse

# def MaxRel(original_target,original_pred):
#     'relative loss in percentage'
#     # calculate maxrel for each sample and average

#     maxrel_per_sample = np.mean(np.abs(original_target-original_pred) / np.max(original_target, axis=1,keepdims=True), axis=(1,2,3)) * 100
#     # maxrel = np.median(maxrel_per_sample)
#     return torch.tensor(maxrel_per_sample)

#     # return np.mean( np.abs(original_target-original_pred) / np.max(original_target, axis=1,keepdims=True)) * 100
class MaxRel:
    '''
    Return the maxrel in local batch, with or without cutting out centre region
    maxrel: scalar value
    '''
    def __init__(self,original_target, original_pred):
        # Check if inputs are NumPy arrays or PyTorch tensors and convert accordingly

        if isinstance(original_target, np.ndarray):
            self.original_target = torch.from_numpy(original_target)
        elif isinstance(original_target, torch.Tensor):
            self.original_target = original_target
        else:
            raise ValueError("original_target must be a NumPy array or a PyTorch tensor.")

        if isinstance(original_pred, np.ndarray):
            self.original_pred = torch.from_numpy(original_pred)
        elif isinstance(original_pred, torch.Tensor):
            self.original_pred = original_pred
        else:
            raise ValueError("original_pred must be a NumPy array or a PyTorch tensor.")
        self.batch_size = None
        self.channels   = None
    
    def maxrel(self,center_size):
        """Calculate the relative loss in percentage using PyTorch."""
        # Calculate maxrel for each sample
        if self.original_pred.ndim == 5:
            # reshape to (batch, 7, 64,64)
            batch_size, channels, depth, height, width = self.original_pred.shape
            pred_reshaped = self.original_pred.view(batch_size * channels, depth, height, width)
            target_reshaped = self.original_target.view(batch_size * channels, depth, height, width)
        elif self.original_pred.ndim == 4:
            # reshape to (batch, 1, 64,64)
            pred_reshaped = self.original_pred
            target_reshaped = self.original_target
        else:
            raise ValueError(f"Unsupported input dimensions: {self.original_pred.ndim}")
        
        max_per_sample = torch.max(target_reshaped, dim=1, keepdim=True).values  # Max value along the specified dimension
        abs_diff = torch.abs(target_reshaped - pred_reshaped)
        rel_error = abs_diff / max_per_sample  # Element-wise relative error

        filled_rel_error = self.fill_center_with_avg(rel_error, center_size)
        
        maxrel_per_sample = torch.mean(filled_rel_error, dim=(1, 2, 3)) * 100  # Average over dimensions and convert to percentage
        # Calculate median of maxrel across all samples
        maxrel = torch.median(maxrel_per_sample)
        return maxrel
    
    def fill_center_with_avg(self,rel_error,center_size):
        batch_size,channels,h,w = rel_error.shape
        # Create a copy to modify
        filled_rel_error = rel_error.clone()
        center_h, center_w = h // 2, w // 2

        # Calculate start and end indices for the center region
        start_h = center_h - center_size // 2
        end_h = center_h + center_size // 2
        start_w = center_w - center_size // 2
        end_w = center_w + center_size // 2

        # Calculate the average value excluding the center region
        for b in range(batch_size):
            for c in range(channels):
                # Create a mask for the center region
                mask = torch.ones_like(filled_rel_error[b, c], dtype=torch.bool)
                mask[start_h:end_h, start_w:end_w] = False  # Exclude center region
                
                # Compute the average of the remaining pixels (excluding the center)
                avg_value = filled_rel_error[b, c][mask].mean()
                
                # Fill the center region with the average value
                filled_rel_error[b, c, start_h:end_h, start_w:end_w] = avg_value

        return filled_rel_error
    
def single_maxrel(target, pred, centre_size):
    ''' Batched tar and pred in original space, numpy'''
    # Reshape the target and prediction arrays
    if pred.ndim == 5:
        # reshape to (batch, 7, 64,64)
        batch_size, channels, depth, height, width = pred.shape
        pred_reshaped = pred.reshape(batch_size * channels, depth, height, width)
        target_reshaped = target.reshape(batch_size * channels, depth, height, width)
    elif pred.ndim == 4:
        # reshape to (batch, 1, 64,64)
        pred_reshaped = pred
        target_reshaped = target


    # Calculate the relative error
    maxrel = np.abs(target_reshaped - pred_reshaped) / np.max(target_reshaped, axis=1, keepdims=True) * 100
    filled_rel_error = maxrel.copy()
    
    center_h, center_w = height // 2, height // 2

    # Calculate start and end indices for the center region
    start_h = center_h - centre_size // 2
    end_h = center_h + centre_size // 2
    start_w = center_w - centre_size // 2
    end_w = center_w + centre_size // 2

    # Calculate the average value excluding the center region
    for b in range(batch_size):
        for c in range(depth):
            # Create a mask for the center region
            mask = np.ones_like(filled_rel_error[b,c], dtype=bool)  # Corrected to use bool
            mask[start_h:end_h, start_w:end_w] = False  # Exclude center region
            
            # Compute the average of the remaining pixels (excluding the center)
            avg_value = filled_rel_error[b,c][mask].mean()
            
            # Fill the center region with the average value
            filled_rel_error[b,c, start_h:end_h, start_w:end_w] = avg_value
    
    return filled_rel_error

# def MaxRel(original_target, original_pred):
#     """Calculate the relative loss in percentage using PyTorch."""
#     # Calculate maxrel for each sample
#     if original_pred.ndim == 5:
#         batch_size, channels, depth, height, width = original_pred.shape
#         pred_reshaped = original_pred.view(batch_size * channels, depth, height, width)
#         target_reshaped = original_target.view(batch_size * channels, depth, height, width)
#     elif original_pred.ndim == 4:
#         pred_reshaped = original_pred
#         target_reshaped = original_target
#     else:
#         raise ValueError(f"Unsupported input dimensions: {original_pred.ndim}")
    
#     max_per_sample = torch.max(target_reshaped, dim=1, keepdim=True).values  # Max value along the specified dimension
#     abs_diff = torch.abs(target_reshaped - pred_reshaped)
#     rel_error = abs_diff / max_per_sample  # Element-wise relative error
    
#     maxrel_per_sample = torch.mean(rel_error, dim=(1, 2, 3)) * 100  # Average over dimensions and convert to percentage
#     # Calculate median of maxrel across all samples
    # maxrel = torch.median(maxrel_per_sample)
#     return maxrel


def calculate_ssim_batch(target, pred):
    # Reshape the tensors if input is 5D (batch, channels, depth, height, width)
    if pred.ndim == 5:
        batch_size, channels, depth, height, width = pred.shape
        pred_reshaped = pred.view(batch_size * channels, depth, height, width)
        target_reshaped = target.view(batch_size * channels, depth, height, width)
    elif pred.ndim == 4:
        pred_reshaped = pred
        target_reshaped = target
    else:
        raise ValueError(f"Unsupported input dimensions: {pred.ndim}")

    # Initialize a list to store SSIM scores for each frequency
    num_samples, num_freqs, height, width = target_reshaped.shape
    ssim_scores = torch.zeros(num_freqs, device=pred.device)

    # Calculate SSIM for each frequency
    for freq in range(num_freqs):
        # Extract specific frequency images from target and prediction
        pred_img = pred_reshaped[:, freq, :, :].unsqueeze(1)  # Shape: (batch, 1, height, width)
        target_img = target_reshaped[:, freq, :, :].unsqueeze(1)  # Shape: (batch, 1, height, width)

        # Compute SSIM for this frequency channel over the batch
        freq_ssim = ssim(pred_img, target_img, data_range=target_img.max() - target_img.min())

        # Average SSIM across the batch for this frequency
        ssim_scores[freq] = freq_ssim

    return ssim_scores

# def calculate_ssim_batch(target,pred):
#     if pred.ndim == 5:  # Input is 5D
#         # Combine batch and sequence dimensions: (batch, channels, depth, height, width) -> (batch*depth, channels, height, width)
#         batch_size, channels, depth, height, width = pred.shape
#         pred_reshaped = pred.reshape(batch_size * channels, depth, height, width)
#         target_reshaped = target.reshape(batch_size * channels, depth, height, width)
#     elif pred.ndim == 4:  # Input is 4D
#         # No need to reshape if input is already 4D: (batch, channels, height, width)
#         pred_reshaped = pred
#         target_reshaped = target
#     else:
#         raise ValueError(f"Unsupported input dimensions: {pred.ndim}")


#     # Calculate SSIM for each image in the batch
#     num_samples, num_freqs, height, width = target_reshaped.shape
#     ssim_scores = torch.zeros(num_freqs)

#     for freq in range(num_freqs):
#         freq_ssim = []
#         for i in range(num_samples):
#             # Extract the specific frequency images from target and prediction
#             pred_img = pred_reshaped[i, freq, :, :]
#             target_img = target_reshaped[i, freq, :, :]
            
#             # Compute SSIM for this frequency channel and sample
#             score = ssim(target_img, pred_img, data_range=target_img.max() - target_img.min())
#             freq_ssim.append(score)

#         # Average SSIM across all samples for this frequency
#         avg_ssim = np.mean(freq_ssim)
#         ssim_scores[freq] = torch.tensor(avg_ssim)

#     return ssim_scores
