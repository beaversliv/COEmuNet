import h5py as h5
import numpy as np
import sys
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)
# torch related
import torch
from torch.utils.data import Dataset

import logging
import time

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
### Custom Dataset ###
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