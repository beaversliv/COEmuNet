import h5py as h5
import numpy as np
import pickle
import sys
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)
# torch related
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

### custom transformations ###
class CustomTransform:
    def __init__(self,file_statistics):
        self.file_statistics = file_statistics
        with open(file_statistics,'rb') as pkl_file:
            self.statistics = pickle.load(pkl_file)

    def __call__(self,x,y):
        xt = x.copy()
        yt = y.copy()
        # single x, y
        xt[0] = (xt[0]-self.statistics['vz'][0])/self.statistics['vz'][1]

    
        xt[1] = np.log(xt[1],dtype=np.float32)
        xt[1] = xt[1] - self.statistics['temp'][0]
        xt[1] = xt[1]/self.statistics['temp'][1]

        xt[2] = np.log(xt[2],dtype=np.float32)
        xt[2] = xt[2] - self.statistics['co'][0]
        xt[2] = xt[2]/self.statistics['co'][1]

        y_v = yt.reshape(-1)
        yt = np.where(yt == 0, np.min(y_v[y_v != 0]), yt)
        yt = np.log(yt)
        yt = yt-self.statistics['y'][0]
        yt = yt/self.statistics['y'][1]
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
        file_idx, file_local_idx = self.find_file_index(idx)

        # Load the specific data from the file
        with h5.File(self.file_paths[file_idx], 'r') as hdf5_file:
            x = np.array(hdf5_file['input'][file_local_idx], dtype=np.float32)
            y = np.array(hdf5_file['output'][file_local_idx, :, :, 15:16], dtype=np.float32)

        if self.transform:
            xt,yt = self.transform(x, y)

        return xt, yt

    def find_file_index(self, global_idx):
        running_total = 0
        for file_idx, file_path in enumerate(self.file_paths):
            file_length = self.get_file_length(file_path)
            if running_total + file_length > global_idx:
                return file_idx, global_idx - running_total
            running_total += file_length
        raise IndexError("Index out of range")
# file_statistics = '/home/dc-su2/physical_informed/cnn/unrotate_statistics.pkl'
# train_file_paths = [f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/Batches/batch_{i}.hdf5'for i in range(8)]
# # val_file_path = ['/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/Batches/vali.hdf5']
# test_file_path = ['/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/Batches/unrotate.hdf5']
# custom_transform = CustomTransform(file_statistics)
# ### dataloader ###
# train_dataset= IntensityDataset(train_file_paths,transform=None)
# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# # vali_dataset= IntensityDataset(val_file_path,transform=custom_transform)
# # vali_dataloader = DataLoader(vali_dataset, batch_size=64, shuffle=True)

# test_dataset= IntensityDataset(test_file_path,transform=custom_transform)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# for i, batch in enumerate(train_dataloader):
#     inputs,target = batch

# # for i, batch in enumerate(vali_dataloader):
#     # inputs,target = batch

# for i, batch in enumerate(test_dataloader):
#     inputs,target = batch
    # print(i, inputs.shape,target.shape)
#     # print(inputs.shape)