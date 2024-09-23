import numpy as np
import h5py as h5
import os
import sys
from mpi4py import MPI
import json
from config.config         import parse_args,load_config
import logging
args = parse_args()
config = load_config(args.config)

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"data/preprocess/statistic/{config['dataset']['name']}_transform.log", level=logging.INFO)
class PreProcessingTransform:
    def __init__(self,statistics_path,statistics_values,dataset_name):
        self.dataset_name = dataset_name
        self.statistics  = self._load_statistics(statistics_path,statistics_values)
        # print('read statistic:',self.statistics)
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
            yt[yt<=-40] = -40
            yt = (yt - self.statistics['intensity']['min']) / (self.statistics['intensity']['max'] - self.statistics['intensity']['min'])
            yt = np.transpose(yt,(2,0,1))
            yt = yt[np.newaxis,:,:,:]

        else:
            yt = yt-self.statistics['intensity']['min']
            yt = yt/self.statistics['intensity']['median']
            yt = np.transpose(yt,(2,0,1))
        return xt,yt

def find_path():
    with open(f"data/preprocess/statistic/{config['dataset']['name']}_non_outliers.json", 'r') as f:
        model_files = json.load(f)
    # model_files = model_files[:11]
    clean_files = []
    for model_file in model_files:
        listb = model_file.split('/')
        listb[-1] = 'clean_'+ listb[-1]
        clean_file = ('/').join(listb)
        clean_files.append(clean_file)
    with open(f"data/preprocess/statistic/{config['dataset']['name']}_clean_files.json", 'w') as f:
            json.dump(clean_files, f)
    return model_files,clean_files

def process_files(rank, model_files, clean_files):
    transform = PreProcessingTransform(statistics_path=config['dataset']['statistics']['path'],
                                        statistics_values=config['dataset']['statistics']['values'],dataset_name=config['dataset']['name'])

    for model_file, clean_file in zip(model_files, clean_files):
        with h5.File(model_file, 'r') as f:
            co = np.array(f['CO'])
            tmp = np.array(f['temperature'])
            v_z = np.array(f['velocity_z'])
            if config['dataset']['name'] == 'mulfreq':
                y = np.array(f['I'][:,:,12:19],dtype=np.float32)
            else:
                y = np.array(f['I'],dtype=np.float32)

            x = np.stack([v_z,tmp,co],axis=0)

            xt, yt = transform(x, y)
        # logger.info(f'write {clean_file}')
        with h5.File(clean_file, 'w') as h5f:
            h5f.create_dataset('input', data=xt)
            h5f.create_dataset('output', data=yt)
        # os.remove(model_file)
def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Find all file paths
    if rank == 0:
        model_files, clean_files = find_path()
        # model_files,clean_files = model_files[:5],clean_files[:5]
    else:
        model_files,clean_files = None,None
    model_files = comm.bcast(model_files, root=0)
    clean_files = comm.bcast(clean_files, root=0)

    chunk_size = 2000
    total_num_files = len(model_files)
    files_per_round = chunk_size * size
    num_rounds = (total_num_files + files_per_round - 1) // files_per_round
    for round_idx in range(num_rounds):
        start_idx = round_idx * files_per_round
        end_idx = min(start_idx + files_per_round, total_num_files)

        files_this_round1, files_this_round2= model_files[start_idx:end_idx],clean_files[start_idx:end_idx]
        local_model_files = files_this_round1[rank * chunk_size:(rank + 1) * chunk_size]
        local_clean_files = files_this_round2[rank * chunk_size:(rank + 1) * chunk_size]

        if local_model_files is not None and len(local_model_files) > 0:
            logger.info(f"round {round_idx} Process {rank} received {len(local_model_files)} files.")
        else:
            logger.info(f"round {round_idx} Process {rank} received no data.")
            continue

        process_files(rank, local_model_files, local_clean_files)
    # Finalize MPI
    comm.Barrier()
if __name__=='__main__':
    main()
