import h5py as h5
import numpy as np
import pickle
import sys
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)
# torch related
import torch
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision.transforms   import Compose
import logging
import time
from sklearn.model_selection      import train_test_split 
logger = logging.getLogger(__name__)
### custom compose class, handles two arguments x and y
class CustomCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):
        for transform in self.transforms:
            x, y = transform(x, y)
        return x, y

    def __repr__(self):
        return self.__class__.__name__ + f'({self.transforms})'
### custom transformations ###
class PreProcessingTransform:
    def __init__(self,file_statistics):
        self.statistics = self._read_statistics(file_statistics)
    def _read_statistics(self, file_path):
        with h5.File(file_path, 'r') as f:
            statistics = {
                'velocity': {
                    'mean': f['vel']['mean'][()],
                    'std': f['vel']['std'][()]
                },
                # Add other features as needed
                'temperature': {
                    'min': f['temp']['min'][()],
                    'median': f['temp']['median'][()]
                },
                'density': {
                    'min': f['co']['min'][()],
                    'median': f['co']['median'][()]
                },
                'intensity':{
                    'min':f['y']['min'][()],
                    'median': f['y']['median'][()],
                }
            }
        return statistics
    
    def __call__(self,x,y):
        xt = x.copy()
        yt = y.copy()
        # single x, y
        xt[0] = (xt[0]-self.statistics['velocity']['mean'])/self.statistics['velocity']['std']

        
        xt[1] = np.log(xt[1],dtype=np.float32)
        xt[1] = xt[1] - self.statistics['temperature']['min']
        xt[1] = xt[1]/self.statistics['temperature']['median']
        # try:
        xt[2] = np.log(xt[2],dtype=np.float32)
        # except RuntimeWarning:
        #     xt[2] = np.log(xt[2] + 1e-10, dtype=np.float32)
        xt[2] = xt[2] - self.statistics['density']['min']
        xt[2] = xt[2]/self.statistics['density']['median']

        y_v = yt.reshape(-1)
        yt = np.where(yt == 0, np.min(y_v[y_v != 0]), yt)
        yt = np.log(yt)
        yt = yt-self.statistics['intensity']['min']
        yt = yt/self.statistics['intensity']['median']
        yt = np.transpose(yt,(2,0,1))
        
        # convert to tensor
        xt = torch.tensor(xt,dtype=torch.float32)
        yt = torch.tensor(yt,dtype=torch.float32)

        return xt,yt

### custom dataset ###
# class IntensityDataset(Dataset):
#     def __init__(self,file_paths,transform=None):
#         '''
#         arguments:
#         file_paths ([strings]): path to the hdf5 file. 
#         '''
#         self.file_paths = file_paths
#         self.transform = transform
#         self.data      = self.load_data()

#     def load_data(self):
#         data = [] # len = num files
#         for file_path in self.file_paths:
#             print(file_path)
#             with h5.File(file_path, 'r') as hdf5_file:
#                 cubes = np.array(hdf5_file['input'],dtype=np.float32)
#                 intensity = np.array(hdf5_file['output'][:, :, :, 15:16],dtype=np.float32)
#                 data.append((cubes, intensity)) 

#         return data
        
#     def __len__(self):
#         return sum(len(intensity) for cubes, intensity in self.data)
#     def __getitem__(self,idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         # Find the file that contains the requested index
#         file_idx = 0
#         while idx >= len(self.data[file_idx][1]):
#             idx -= len(self.data[file_idx][1])
#             file_idx += 1

#         # Get the data from the corresponding file
#         cubes, intensity = self.data[file_idx]
#         x = cubes[idx]
#         y = intensity[idx]
#             # sys.exit()
#         if self.transform:
#             xt, yt = self.transform(x, y)
#         return xt,yt
class IntensityDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        """
        Arguments:
        file_paths ([str]): Paths to the hdf5 files.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        # Lazily compute the total length
        if not hasattr(self, '_total_len'):
            self._total_len = sum(self.get_file_length(file_path) for file_path in self.file_paths)
        return self._total_len

    def get_file_length(self, file_path):
        with h5.File(file_path, 'r') as hdf5_file:
            intensity = hdf5_file['output']
            return intensity.shape[0]

    def __getitem__(self, idx):
        # start timing
        start_time = time.time()
        file_idx, file_local_idx = self.find_file_index(idx)
        
        # Load the specific data from the file
        with h5.File(self.file_paths[file_idx], 'r') as hdf5_file:
            x = np.array(hdf5_file['input'][file_local_idx], dtype=np.float32)
            y = np.array(hdf5_file['output'][file_local_idx], dtype=np.float32)
        if self.remove_outlier(y):
            return self.__getitem__((idx + 1) % self.__len__())  # Skip to the next sample
        if self.transform:
            xt,yt = self.transform(x, y)
        else: 
            xt = x
            yt = y
        # end timing
        end_time = time.time()
        load_time = end_time - start_time
        # write in log
        # logger.info(f"Data loading time for index {idx}: {load_time} seconds")
        return xt, yt

    def find_file_index(self, global_idx):
        running_total = 0
        for file_idx, file_path in enumerate(self.file_paths):
            file_length = self.get_file_length(file_path)
            if running_total + file_length > global_idx:
                return file_idx, global_idx - running_total
            running_total += file_length
        raise IndexError("Index out of range")
    def remove_outlier(self,y):
        y[y == 0] = np.min(y[y != 0])
        y = np.log(y)
        if np.min(y) <= -50:
            return True
        return False
class AddGaussianNoise1(object):
    '''
    calculate std for each feature each sample.
    '''
    def __init__(self,scale_factor=0.1):
        self.scale_factor = scale_factor

    def __call__(self,x,y):
        noisy_tensor = x.clone()
        for i in range(3):
            # Calculate the standard deviation for each feature slice
            std = torch.std(x[i, :, :, :])
            # Generate noise with mean 0 and std derived from the feature
            mean_ = 0.0
            noise = torch.randn(x[i, :, :, :].size()) * (self.scale_factor * std) + mean_
            noisy_tensor[i, :, :, :] += noise 
        return noisy_tensor,y

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'

class AddGaussianNoise(object):
    def __init__(self, means, stds):
        assert len(means) == len(stds), "Means and stds must have the same length"
        self.means = means
        self.stds = stds

    def __call__(self, tensor):
        noisy_tensor = tensor.clone()
        for i in range(len(self.means)):
            noise = torch.randn(tensor[i, :, :, :].size()) * self.stds[i] + self.means[i]
            noisy_tensor[i, :, :, :] += noise
        return noisy_tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(means={self.means}, stds={self.stds})'
class AddUniformNoise(object):
    def __init__(self, lows=-0.1, highs=0.1):
        self.lows = lows
        self.highs = highs

    def __call__(self, tensor):
       
        noisy_tensor = tensor.clone()
        for i in range(3):
            noise = (torch.rand(tensor[i, :, :, :].size()) * (self.highs[i] - self.lows[i]) + self.lows[i])
            noisy_tensor[i,:,:,:] += noise
        return noisy_tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(low={self.low}, high={self.high})'
class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        with h5.File(self.file_path, 'r') as f:
            self.data_len = f['input'].shape[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        with h5.File(self.file_path, 'r') as f:
            x = np.array(f['input'][idx],dtype=np.float32)
            y = np.array(f['output'][idx],dtype=np.float32)
        x = torch.tensor(x)
        y = torch.tensor(y)
        if self.transform:
            x = self.transform(x)

        return x,y
if __name__ == '__main__':

    transform = CustomCompose( [PreProcessingTransform('/home/dc-su2/physical_informed/cnn/statistic/random.hdf5'),
                    AddGaussianNoise1(scale_factor=0.1)
    ])
    # transform = PreProcessingTransform('/home/dc-su2/physical_informed/cnn/statistic/random.hdf5')
    dataset = IntensityDataset(['/home/dc-su2/rds/rds-dirac-dr004/Magritte/dummy.hdf5'],transform=transform)
    
    # train test split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    # lows,hights = [-0.1,-0.2,-0.2],[0.1,0.2,0.2]
    # noise_transform = AddUniformNoise(lows, hights)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # train_dataset.dataset.transform = noise_transform
    train_dataloader = DataLoader(train_dataset, batch_size=30)

    for bidx,sample in enumerate(train_dataloader):
        x,y = sample[0],sample[1]
        print(y.shape)