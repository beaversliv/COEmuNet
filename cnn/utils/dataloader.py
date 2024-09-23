import h5py as h5
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)
# torch related
import torch
from torch.utils.data import Dataset,DataLoader
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager
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
    def __init__(self,statistics_path,statistics_values,dataset_name):
        self.dataset_name = dataset_name
        self.statistics  = self._load_statistics(statistics_path,statistics_values)
        print('read statistic:',self.statistics)
    def _load_statistics(self,statistics_path,statistics_values):
        statistics = {}
        with h5.File(statistics_path, 'r') as f:
            for value in statistics_values:
                feature = value['name']
                stats_to_read = value['stats']
                statistics[feature] = {stat: f[feature][stat][()] for stat in stats_to_read}
        return statistics

    
    def __call__(self,x,y):
        xt = np.zeros_like(x)
        yt = np.zeros_like(y)
        # single x, y
        xt[0] = (x[0]-self.statistics['velocity']['mean'])/self.statistics['velocity']['std']

        
        xt[1] = np.log(x[1],dtype=np.float32)
        xt[1] = xt[1] - self.statistics['temperature']['min']
        xt[1] = xt[1]/self.statistics['temperature']['median']
        # try:
        xt[2] = np.log(x[2],dtype=np.float32)
        # except RuntimeWarning:
        #     xt[2] = np.log(xt[2] + 1e-10, dtype=np.float32)
        xt[2] = xt[2] - self.statistics['density']['min']
        xt[2] = xt[2]/self.statistics['density']['median']

        y_v = y.reshape(-1)
        yt = np.where(y == 0, np.min(y_v[y_v != 0]), y)
        yt = np.log(yt)
        if self.dataset_name == 'mulfreq':
            yt = (yt - self.statistics['intensity']['min']) / (self.statistics['intensity']['max'] - self.statistics['intensity']['min'])
            yt = np.transpose(yt,(2,0,1))
            yt = yt[np.newaxis,:,:,:]

        else:
            yt = yt-self.statistics['intensity']['min']
            yt = yt/self.statistics['intensity']['median']
            yt = np.transpose(yt,(2,0,1))
        
        # convert to tensor
        xt = torch.tensor(xt,dtype=torch.float32)
        yt = torch.tensor(yt,dtype=torch.float32)

        return xt,yt
### Custom Dataset ###
class SequentialDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        """
        Arguments:
        file_paths ([str]): Paths to the hdf5 files.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_paths = file_paths
        self.transform = transform
        self.data_loading_time = 0
        self.data_idx_finding_time = 0
        self.data_transform_time   = 0

    def get_time(self):
        time_str = f"load_time: {self.data_loading_time}, idx_find_time: {self.data_idx_finding_time}, transform time:{self.data_transform_time}"
        self.data_loading_time = 0
        self.data_idx_finding_time = 0
        self.data_transform_time   = 0
        return time_str
    
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
        
        self.data_idx_finding_time += time.time() - start_time

        start_time1 = time.time()
        # Load the specific data from the file
        with h5.File(self.file_paths[file_idx], 'r') as hdf5_file:
            x = np.array(hdf5_file['input'][file_local_idx], dtype=np.float32)
            y = np.array(hdf5_file['output'][file_local_idx], dtype=np.float32)

        self.data_loading_time += time.time() - start_time1

        start_time = time.time()
        if self.transform:
            xt,yt = self.transform(x, y)
        else: 
            xt = x
            yt = y
            # yt = np.transpose(yt,(2,0,1))
            # yt = yt[np.newaxis,:,:,:]
        # end timing
        self.data_transform_time += time.time() - start_time
        # convert to tensor
        xt = torch.tensor(xt,dtype=torch.float32)
        yt = torch.tensor(yt,dtype=torch.float32)

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
        y[y == 0.0] = np.min(y[y != 0.0])
        y = np.log(y)
        if np.min(y) <= -50:
            return True
        return False
    
class AsyncChunkDataset(Dataset):
    def __init__(self, file_paths, dataset_name):
        self.file_paths = file_paths
        self.dataset_name = dataset_name
        self.file_cache = {}
        self.processed_files = set()
        self.current_file_idx = -1
        self.X = None
        self.Y = None
 
        self.data_loading_time = 0
        self.data_idx_finding_time = 0
        self.data_transform_time   = 0

        # self.executor = ThreadPoolExecutor(max_workers=1)
        # self.future = None
        # self.load_file_into_memory(0)
        # self._prefetch_next_file(1)

    def get_time(self):
        time_str = f"load_time: {self.data_loading_time}, idx_find_time: {self.data_idx_finding_time}, transform time:{self.data_transform_time}"
        self.data_loading_time = 0
        self.data_idx_finding_time = 0
        self.data_transform_time   = 0
        return time_str
    
    def load_file_into_memory(self, file_idx):
        """Load the file into memory, but cache the loaded data."""
        if file_idx in self.file_cache:
            self.X, self.Y = self.file_cache[file_idx]
            self.current_file_idx = file_idx
            return
        if len(self.file_cache) > 2:
            self.file_cache.clear()
        start_time = time.time()
        print(f"[Worker {torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 'Main'}] Loading file {self.file_paths[file_idx]} into memory... ")
        with h5.File(self.file_paths[file_idx], 'r') as f:
            self.X = np.array(f['input'], dtype=np.float32)
            self.Y = np.array(f['output'], dtype=np.float32)
        # self.file_cache[file_idx] = (self.X, self.Y)
        self.current_file_idx = file_idx
        print(f"[Worker {torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 'Main'}] Loaded file {self.file_paths[file_idx]} in {time.time() - start_time:.2f} seconds")    
        self.processed_files.add(file_idx)

    def _prefetch_next_file(self, next_file_idx):
        """Asynchronously prefetch the next file."""
        if next_file_idx < len(self.file_paths):
            self.future = self.executor.submit(self.load_file_into_memory, next_file_idx)
    def start_new_epoch(self):
        """Reset processed files at the beginning of a new epoch."""
        self.processed_files.clear()

    def __len__(self):
        # Total length across all files
        return sum(self.get_file_length(fp) for fp in self.file_paths)

    def get_file_length(self, file_path):
        """Get the number of samples in the HDF5 file."""
        with h5.File(file_path, 'r') as f:
            return len(f['input'])

    def __getitem__(self, idx):
        # Determine which file the index corresponds to
        file_idx, file_local_idx = self.find_file_index(idx)

        # if self.future is not None:
        #     self.future.result()
        # Load the file into memory if not already loaded
        if file_idx != self.current_file_idx:
            self.load_file_into_memory(file_idx)
            # Start prefetching the next file asynchronously
            # self._prefetch_next_file(file_idx + 1)

        start_time = time.time()
        # Retrieve the specific data point
        xt = self.X[file_local_idx]
        yt = self.Y[file_local_idx]
        self.data_loading_time += time.time() - start_time

        start_time = time.time()
        if self.dataset_name == 'mulfreq':
            yt = yt[np.newaxis,:,:,:]

        xt = torch.tensor(xt,dtype=torch.float32)
        yt = torch.tensor(yt,dtype=torch.float32)
        self.data_transform_time += time.time() - start_time
        return xt, yt

    def find_file_index(self, global_idx):
        """Find which file and local index correspond to the global index."""
        running_total = 0
        for file_idx, file_path in enumerate(self.file_paths):
            file_length = self.get_file_length(file_path)
            if running_total + file_length > global_idx:
                file_local_idx = global_idx - running_total
                return file_idx, file_local_idx
            running_total += file_length
        raise IndexError("Index out of range")

class ChunkLoadingDataset(Dataset):
    def __init__(self, file_list, mini_batch_size=128,dataset_name='rotation'):
        """
        Args:
            file_list (list): List of file paths where data is stored.
            mini_batch_size (int): The size of the mini-batches for each loaded chunk.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_list = file_list  # List of chunked file paths (e.g., ['batch_0.hdf5', 'batch_1.hdf5', ...])
        self.mini_batch_size = mini_batch_size
        self.dataset_name = dataset_name
        # Store the number of samples in each file

        self.file_lengths = []
        for file_path in file_list:
            with h5.File(file_path, 'r') as f:
                self.file_lengths.append(f['input'].shape[0])

        # Calculate cumulative number of samples (for indexing)
        self.cumulative_file_lengths = np.cumsum(self.file_lengths)

    def __len__(self):
        total_samples = sum(self.file_lengths)
        # Calculate the total number of mini-batches, including partial batches
        num_batches = (total_samples + self.mini_batch_size - 1) // self.mini_batch_size
        return num_batches  # Total mini-batches available
    def __getitem__(self, idx):
        # Figure out which file and which batch inside the file corresponds to the index
        # file_idx = idx // (10000 // self.mini_batch_size)  # Assuming 1000 samples per file
        # mini_batch_idx = idx % (10000 // self.mini_batch_size)
        global_sample_idx = idx * self.mini_batch_size
        file_idx = np.searchsorted(self.cumulative_file_lengths, global_sample_idx, side='right')
        # Adjust the index relative to the file
        if file_idx == 0:
            mini_batch_idx = global_sample_idx
        else:
            mini_batch_idx = global_sample_idx - self.cumulative_file_lengths[file_idx - 1]
        file_path = self.file_list[file_idx]
        with h5.File(file_path, 'r') as f:
            start_idx = mini_batch_idx
            end_idx = min(start_idx + self.mini_batch_size, f['input'].shape[0])
            input_data = f['input'][start_idx:end_idx]
            output_data = f['output'][start_idx:end_idx]

        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output_tensor = torch.tensor(output_data, dtype=torch.float32)
        return input_tensor, output_tensor
    
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)