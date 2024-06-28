from mpi4py import MPI
import numpy             as np
import h5py              as h5
import pandas            as pd
import os
import math
import json
import random
import sys

def load_dataset_names(json_file_path):
    '''
    locate paths for individual samples
    return: list of paths
    '''
    with open(json_file_path, 'r') as file:
        lists = json.load(file)
    datasets = lists['datasets']   
    # random.shuffle(datasets)
    return datasets

def distribute_datasets(datasets, rank, nproc, segment_size=10903):
    '''
    split list of paths (datasets) into batches by
    marking start and end idx
    '''
    start_index = rank * segment_size
    end_index = (rank + 1) * segment_size if rank < nproc - 1 else len(datasets)
    return datasets[start_index:end_index]

def process_datasets(file_list):
    '''
    pack individual samples into baches based on the splited file paths
    '''
    grid, freq, num_input = 64, 1, 3
    X = np.zeros((len(file_list), num_input, grid, grid, grid))
    Y = np.zeros((len(file_list), grid, grid, freq))
    for idx, file in enumerate(file_list):
        with h5.File(file, 'r') as f:
            co = np.array(f['CO'])
            tmp = np.array(f['temperature'])
            v_z = np.array(f['velocity_z'])
            img = np.array(f['I'])
        
        X[idx] = np.stack((v_z, tmp, co), axis=0)
        Y[idx] = img
    
    return X, Y
def save_processed_data(output_path, X, Y):
    '''
    save batches into hdf5
    '''
    with h5.File(output_path, 'w') as file:
        file['input'] = X
        file['output'] = Y

class preProcessing:
    def __init__(self,data_path,stats_path):
        self.data_path = data_path
        self.meta = {}
        self.stats_path = stats_path

    def outliers(self):
        with h5.File(self.data_path,'r') as sample:
            x = np.array(sample['input'],np.float32)   # shape(num_samples,3,64,64,64)
            y = np.array(sample['output'], np.float32)# shape(num_samples,64,64,1)
        # take logrithm
        y[y==0] = np.min(y[y!=0])
        I = np.log(y)
        # difference = max - min
        max_values = np.max(I,axis=(1,2))
        min_values = np.min(I,axis=(1,2))
        diff = max_values - min_values
        # find outliers
        outlier_idx = np.where(diff>20)[0]

        # remove outliers
        removed_x = np.delete(x,outlier_idx,axis=0)
        removed_y = np.delete(y,outlier_idx,axis=0)
        return removed_x, removed_y

    def get_data(self):
        x , y = self.outliers()

        feature = ['vel','temp','co']
        x_t = np.transpose(x, (1, 0, 2, 3, 4))
        for idx in range(3):
            if idx == 0:
                self.meta[feature[idx]] = {}
                self.meta[feature[idx]]['mean'] = x_t[idx].mean()
                self.meta[feature[idx]]['std'] = x_t[idx].std()
                x_t[idx] = (x_t[idx] - x_t[idx].mean())/x_t[idx].std()
            else:
                self.meta[feature[idx]] = {}
                self.meta[feature[idx]]['min'] = np.min(x_t[idx])
                self.meta[feature[idx]]['median'] = np.median(x_t[idx])
                x_t[idx] = np.log(x_t[idx])
                
                x_t[idx] = x_t[idx] - np.min(x_t[idx])
                x_t[idx] = x_t[idx]/np.median(x_t[idx])
        print(f'pre-processing value:{self.meta}\n')

        y[y == 0] = np.min(y[y != 0])
        y = np.log(y)
        min_y = np.min(y)
        y = y-min_y

        median_y = np.median(y)
        y = y/median_y
        self.meta['y'] = {'min':min_y, 'meidan':median_y}

        self.save_meta_hdf5(self.meta, self.stats_path)
        print(f'post-processing value:{self.meta}')
        return np.transpose(x_t, (1, 0, 2, 3, 4)), np.transpose(y,(0,3,1,2))
    def save_meta_hdf5(self,meta, filename='meta.h5'):
        with h5.File(filename, 'w') as f:
            for key, subdict in meta.items():
                grp = f.create_group(key)
                for subkey, value in subdict.items():
                    grp.create_dataset(subkey, data=value)
# Main MPI execution flow
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    # name_lists = '/home/dc-su2/physical_informed/data_gen/datasets.json'
    name_lists = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/files/grid64_data.json'
    datasets = load_dataset_names(name_lists)
    # process_file_list = distribute_datasets(datasets[10903:10903*2], rank, nproc,len(datasets)//nproc)
    process_file_list = distribute_datasets(datasets[10903:10903*2], rank, nproc,10903)
    X, Y = process_datasets(process_file_list)
    print(Y.shape)
    output_file = f'/home/dc-su2/rds/rds-dirac-dr004/Magritte/random_grid64_data_360z.hdf5'
    save_processed_data(output_file, X, Y)

    data_gen = preProcessing('/home/dc-su2/rds/rds-dirac-dr004/Magritte/random_grid64_data_360z.hdf5','/home/dc-su2/physical_informed/cnn/statistic/random_360z.hdf5')
    x,y = data_gen.get_data()
    with h5.File('/home/dc-su2/rds/rds-dirac-dr004/Magritte/clean_random_grid64_data_360z','w') as file:
        file['input'] = x
        file['output'] = y
    print(x.shape,y.shape)
    print('saved!')
    
