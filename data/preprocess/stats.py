import numpy as np
import h5py  as h5
import os
import sys
import json
from mpi4py import MPI

from utils import check_file
from utils  import Logging
class Stats:
    '''
    Calculate and save statistic of provided data. Suitable for smaller dataset
    stats_path : path for saving stats
    '''
    def __init__(self,chunk,dataset_name):
        self.meta = {}
        self.chunk = chunk
        self.dataset_name = dataset_name
    def chunk_data(self):
        '''
        pack individual samples into baches based on the splited file paths
        chunk: a list of str, subset of non outlier files
        '''
        chunk_X = []
        chunk_Y = []
        for idx,file in enumerate(self.chunk):
            with h5.File(file, 'r') as f:
                co = np.array(f['CO'],dtype=np.float32)
                tmp = np.array(f['temperature'],dtype=np.float32)
                v_z = np.array(f['velocity_z'],dtype=np.float32)
                if self.dataset_name == 'mulfreq':
                    img = np.array(f['I'][:,:,12:19],dtype=np.float32)
                else:
                    img = np.array(f['I'],dtype=np.float32)
            x = np.stack((v_z, tmp, co), axis=0)
            y = img
            chunk_X.append(x)
            chunk_Y.append(y)
        chunk_X = np.stack(chunk_X,axis=0)
        chunk_Y = np.stack(chunk_Y,axis=0)
        return chunk_X,chunk_Y

    def stats_calculator(self):
        chunk_X,chunk_Y = self.chunk_data() #rotation (n,64,64,1); mulfreq (n,64,64,31)
        feature = ['velocity','temperature','density']
        x_t = np.transpose(chunk_X, (1, 0, 2, 3, 4))
        for idx in range(3):
            if idx == 0:
                self.meta[feature[idx]] = {}
                self.meta[feature[idx]]['mean'] = x_t[idx].mean()
                self.meta[feature[idx]]['std'] = x_t[idx].std()
                x_t[idx] = (x_t[idx] - x_t[idx].mean())/x_t[idx].std()
            else:
                self.meta[feature[idx]] = {}
                x_t[idx] = np.log(x_t[idx])
                self.meta[feature[idx]]['min'] = np.min(x_t[idx])
                
                x_t[idx] = x_t[idx] - np.min(x_t[idx])
                self.meta[feature[idx]]['median'] = np.median(x_t[idx])
                x_t[idx] = x_t[idx]/np.median(x_t[idx])

        y = chunk_Y
        y[y == 0] = np.min(y[y != 0])
        y = np.log(y)
        if self.dataset_name == 'mulfreq':
            y[y<=-40] = -40
            min_y = np.min(y)
            max_y = np.max(y)
            self.meta['intensity'] = {'min':min_y, 'max':max_y}
        else:
            min_y = np.min(y)
            y = y-min_y
            median_y = np.median(y)
            y = y/median_y
            self.meta['intensity'] = {'min':min_y, 'median':median_y}
        # print(f'stats :{self.meta}')
        return self.meta
def save_meta_hdf5(meta, filename='meta.h5'):
    with h5.File(filename, 'w') as f:
        for key, subdict in meta.items():
            grp = f.create_group(key)
            for subkey, value in subdict.items():
                grp.create_dataset(subkey, data=value)

def stats_main(dataset_name,logger,non_outlier_file_path):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        try:
            with open(non_outlier_file_path, 'r') as f:
                model_files = json.load(f)
        except Exception as e:
            logger.info(f"Warning: Failed to open file {non_outlier_file_path}. {e}")
            comm.Abort()  # This will terminate all processes
    else:
        model_files = None
    model_files = comm.bcast(model_files, root=0)
    logger.info(f'rank {rank} received {len(model_files)} files')
    chunk_size = 2000
    total_num_files = len(model_files)
    files_per_round = chunk_size * size
    num_rounds = (total_num_files + files_per_round - 1) // files_per_round

    global_t_min = float('inf')
    global_c_min = float('inf')
    global_y_min = float('inf')
    global_y_max = float('-inf')

    all_c_medians = []
    all_t_medians = []
    all_y_medians = []

    global_sum = 0
    global_count = len(model_files)

    all_tuples_per_round = []

    for round_idx in range(num_rounds):
        start_idx = round_idx * files_per_round
        end_idx = min(start_idx + files_per_round, total_num_files)
        
        # Get the subset of files for this round
        files_this_round = model_files[start_idx:end_idx]
        # Local chunk for each process
        local_chunk = files_this_round[rank * chunk_size:(rank + 1) * chunk_size]

        if local_chunk is not None and len(local_chunk) > 0:
            logger.info(f"round {round_idx} Process {rank} received {len(local_chunk)} files.")
            stats_saver = Stats(local_chunk,dataset_name)
            local_stats = stats_saver.stats_calculator()

            # Calculate stats for chunk
            local_vz_mean  = local_stats['velocity']['mean']
            local_vz_std   = local_stats['velocity']['std']
            local_t_min    = local_stats['temperature']['min']
            local_t_median = local_stats['temperature']['median']
            local_c_min    = local_stats['density']['min']
            local_c_median = local_stats['density']['median']
            local_y_min    = local_stats['intensity']['min']
            local_y_max    = local_stats['intensity'].get('max',float('-inf'))
            local_y_median = local_stats['intensity'].get('median',None)

            local_count = len(local_chunk)
            local_sum = local_vz_mean * local_count
            # Tuple for second pass
            tuple_ = (local_vz_mean, local_vz_std**2, local_count)

        else:
            # Process received no data, contribute dummy values
            logger.info(f"round {round_idx} Process {rank} received no data.")
            local_t_min = float('inf')
            local_c_min = float('inf')
            local_y_min = float('inf')
            local_y_max = float('-inf')
            local_t_median = None
            local_c_median = None
            local_y_median = None
            local_sum = 0
            local_count = 0
            tuple_ = (0, 0, 0)  # Dummy tuple

        # Ensure all processes participate in MPI communication
        comm.Barrier()

        # Gather local statistics at the root process
        round_t_min = comm.reduce(local_t_min, op=MPI.MIN, root=0)
        round_c_min = comm.reduce(local_c_min, op=MPI.MIN, root=0)
        round_y_min = comm.reduce(local_y_min, op=MPI.MIN, root=0)
        round_y_max = comm.reduce(local_y_max, op=MPI.MAX, root=0)

        round_t_medians = comm.gather(local_t_median, root=0)
        round_c_medians = comm.gather(local_c_median, root=0)
        round_y_medians = comm.gather(local_y_median, root=0)

        round_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
        round_tuples = comm.gather(tuple_, root=0)  # Gather tuples

        if rank == 0:
            logger.info(f'round {round_idx} starts to gather')
            global_t_min = min(global_t_min, round_t_min) 
            global_c_min = min(global_c_min, round_c_min) 
            global_y_min = min(global_y_min, round_y_min)
            global_y_max = max(global_y_max, round_y_max)

            round_t_medians = [m for m in round_t_medians if m is not None]
            round_c_medians = [m for m in round_c_medians if m is not None]
            round_y_medians = [m for m in round_y_medians if m is not None]

            all_t_medians.extend(round_t_medians)
            all_c_medians.extend(round_c_medians)
            all_y_medians.extend(round_y_medians)
            global_sum += round_sum
            all_tuples_per_round.append(round_tuples)
    if rank == 0:
        logger.info('starts to calculate median, mean and std')
        global_t_median = np.median(all_t_medians) if all_t_medians else np.nan
        global_c_median = np.median(all_c_medians) if all_c_medians else np.nan
        global_y_median = np.median(all_y_medians) if all_y_medians else np.nan
        global_mean = global_sum / global_count if global_count > 0 else np.nan

        flattened_tuples = [tup for round_list in all_tuples_per_round for tup in round_list]
        # Second pass to calculate std
        global_variance_numerator = 0.0
        for local_mean, local_var, local_count in flattened_tuples:
            global_variance_numerator += (local_var * local_count) + local_count * (local_mean - global_mean) ** 2

        # Calculate the global variance
        global_variance = global_variance_numerator / global_count
        global_std = np.sqrt(global_variance)
        if dataset_name == 'mulfreq':
            global_stats = {
            'velocity': {'mean': global_mean, 'std': global_std},
            'temperature': {'min': global_t_min, 'median': global_t_median},
            'density': {'min': global_c_min, 'median': global_c_median},
            'intensity': {'min': global_y_min, 'max': global_y_max}
        }
        else:
            global_stats = {
                'velocity': {'mean': global_mean, 'std': global_std},
                'temperature': {'min': global_t_min, 'median': global_t_median},
                'density': {'min': global_c_min, 'median': global_c_median},
                'intensity': {'min': global_y_min, 'median': global_y_median}
            }
        logger.info(global_stats)
        save_meta_hdf5(global_stats,f'data/preprocess/statistic/{dataset_name}.hdf5')
        logger.info(f'saved new_{dataset_name} stats')