import torch
import torch.nn as nn
import numpy as np
# Structural Similarity Index Measure (SSIM) 
from skimage.metrics import structural_similarity as ssim
from collections import namedtuple
import torchvision.models as models
# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft
class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight

class VGGFeatures(torch.nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg_model = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_model[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_model[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_model[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_model[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
    
class ResNetFeatures(nn.Module):
    def __init__(self):
        super(ResNetFeatures, self).__init__()
        resnet34 = models.resnet34(pretrained=True).eval()
        self.features = nn.Sequential(*list(resnet34.children())[:-2])

    def forward(self, x):
        return self.features(x)
    
class Lossfunction(nn.Module):
    def __init__(self, pretrained_model,use_freq_loss=False,use_perceptual_loss=False,mse_loss_scacle = 1.0,freq_loss_scale=1.0, perceptual_loss_scale=1.0):
        super(Lossfunction, self).__init__()
        self.pretrained_model      = pretrained_model
        self.use_freq_loss         = use_freq_loss
        self.use_perceptual_loss   = use_perceptual_loss
        self.mse_loss_scacle       = mse_loss_scacle
        self.freq_loss_scale       = freq_loss_scale
        self.perceptual_loss_scale = perceptual_loss_scale
    def forward(self,target,pred):
        # Mean Squared Error Loss
        mse_loss = self.mse_loss_scacle * nn.functional.mse_loss(pred, target)
        total_loss = mse_loss

        # Frequency Loss
        if self.use_freq_loss:
            freq_loss = self.calculate_freq_loss(target,pred)
            total_loss += self.freq_loss_scale * freq_loss
        # Perceptual Loss
        if self.use_perceptual_loss:
            perceptual_loss = self.calculate_perceptual_loss(target,pred)
            total_loss += self.perceptual_loss_scale * perceptual_loss
        return total_loss
    
    def calculate_freq_loss(self,target,pred):
        target_freq = torch.fft.fft2(target)
        pred_freq = torch.fft.fft2(pred)
        return torch.mean(torch.abs(target_freq - pred_freq))
    def calculate_perceptual_loss(self,target,pred):
        # repeat the grayscale channel to create a 3-channel input
        pred   = pred.repeat(1,3,1,1)
        target = target.repeat(1,3,1,1)
        generated_features = self.pretrained_model(pred)
        target_features = self.pretrained_model(target)
        # perceptual_loss = torch.stack([nn.functional.mse_loss(gf, tf) for gf, tf in zip(generated_features, target_features)]).mean()
        perceptual_loss = nn.functional.mse_loss(generated_features, target_features)

        return perceptual_loss

### Define the Perceptual Loss Function ###
def perceptual_loss(generated_features, target_features):
    mse_loss = torch.nn.MSELoss()
    perceptual_loss = 0.0

    for gf, tf in zip(generated_features, target_features):
        perceptual_loss += mse_loss(gf, tf)
    return perceptual_loss
def mean_absolute_percentage_error(true,pred):
    # Avoid division by zero
    true, pred = true + 1e-8, pred + 1e-8
    mean_error = np.mean(np.abs((true - pred) / true)) * 100
    median_error = np.median(np.abs((true - pred) / true))*100
    return mean_error, median_error

def calculate_ssim_batch(target,pred):
    # Calculate SSIM for each image in the batch
    batch_size = pred.shape[0]
    ssim_scores = []
    for i in range(batch_size):
        pred_img = pred[i,0,:,:]
        target_img = target[i,0,:,:] #(64,64,1)

        # Calculate SSIM, assuming multichannel (color) images
        score = ssim(target_img,pred_img, data_range=target_img.max() - target_img.min())
        ssim_scores.append(score)

    # Average SSIM over the batch
    avg_ssim = np.mean(ssim_scores)
    return avg_ssim