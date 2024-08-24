import h5py as h5
import numpy as np
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
class SingleSampleDataset(Dataset):
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
        if self.remove_outlier(y):
            return self.__getitem__((idx + 1) % self.__len__())  # Skip to the next sample
        if self.transform:
            xt,yt = self.transform(x, y)
        else: 
            xt = x
            yt = y
        # end timing
        self.data_transform_time += time.time() - start_time

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
    
class FullDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        self.file_cache = {}
        self.current_file_idx = -1
        self.load_call_count = 0
        self.X = None
        self.Y = None
 
        self.data_loading_time = 0
        self.data_idx_finding_time = 0
        self.data_transform_time   = 0

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

        self.load_call_count += 1
        start_time = time.time()
        print(f"[Worker {torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 'Main'}] Loading file {self.file_paths[file_idx]} into memory... (call count: {self.load_call_count})")
        with h5.File(self.file_paths[file_idx], 'r') as f:
            self.X = np.array(f['input'], dtype=np.float32)
            self.Y = np.array(f['output'], dtype=np.float32)
        self.file_cache[file_idx] = (self.X, self.Y)
        self.current_file_idx = file_idx
        print(f"[Worker {torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 'Main'}] Loaded file {self.file_paths[file_idx]} in {time.time() - start_time:.2f} seconds")

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

        # Load the file into memory if not already loaded
        self.load_file_into_memory(file_idx)

        start_time = time.time()
        # Retrieve the specific data point
        x = self.X[file_local_idx]
        y = self.Y[file_local_idx]
        self.data_loading_time += time.time() - start_time

        start_time = time.time()
        if self.remove_outlier(y):
            return self.__getitem__((idx + 1) % self.__len__())  # Skip to the next sample
        if self.transform:
            xt,yt = self.transform(x, y)
        else: 
            xt = x
            yt = y
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
    def remove_outlier(self,y):
        y[y == 0.0] = np.min(y[y != 0.0])
        y = np.log(y)
        if np.min(y) <= -50:
            return True
        return False
class zipDataset(Dataset):
    def __init__(self, file_paths,transform=None):
        self.file_paths = file_paths
        self.transform  = transform
        self.data_loading_time = 0.0
        start_time = time.time()
        # Load the entire ZIP file into memory
        with open(self.file_paths, 'rb') as f:
            self.zip_data = f.read()
        print(f'data loading time:{time.time() - start_time:.4f}')
        start_time = time.time()
        # Decompress the HDF5 file in memory
        with zipfile.ZipFile(io.BytesIO(self.zip_data), 'r') as z:
            # Assuming the HDF5 file is the only file in the zip, or you know its name
            hdf5_filename = z.namelist()[0]
            
            # Load the HDF5 file into an in-memory BytesIO buffer
            with z.open(hdf5_filename) as hdf5_file:
                self.hdf5_buffer = io.BytesIO(hdf5_file.read())
        
        # Open the HDF5 file from the in-memory buffer
        self.hdf5_data = h5.File(self.hdf5_buffer, 'r')
        
        # Assuming 'input' and 'output' are datasets within the HDF5 file
        self.input_data = self.hdf5_data['input']
        self.output_data = self.hdf5_data['output']
        print(f'decompress time:{time.time() - start_time:.4f}')

    def get_time(self):
        time_str = f"load_time: {self.data_loading_time}"

        self.data_loading_time = 0.0
        return time_str
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        start_time = time.time()
        # Get the input and output data for the specified index
        x = self.input_data[idx]
        y = self.output_data[idx]
        self.data_loading_time += time.time() - start_time
        
        if self.remove_outlier(y):
            return self.__getitem__((idx + 1) % self.__len__())  # Skip to the next sample
        if self.transform:
            xt,yt = self.transform(x, y)
        else: 
            xt = x
            yt = y
        return xt, yt
        
    def remove_outlier(self,y):
        y[y == 0.0] = np.min(y[y != 0.0])
        y = np.log(y)
        if np.min(y) <= -50:
            return True
        return False
    

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
if __name__ == '__main__':
    # Assume we have multiple HDF5 files
    file_paths = ['/Users/ss1421/Desktop/dummy_100.hdf5','/Users/ss1421/Desktop/dummy1_100.hdf5']

    # Create the dataset and DataLoader
    dataset = FullDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)  # Using a single worker

    # Iterate over the DataLoader
    for i, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {i+1}: Loaded batch with shape {inputs.shape}")