import h5py  as h5
import numpy as np
class preProcessing:
    '''
    Calculate and save statistic of provided data. Suitable for smaller dataset
    '''
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

        feature = ['velocity','temperature','density']
        x_t = np.transpose(x, (1, 0, 2, 3, 4))
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
        print(f'pre-processing value:{self.meta}\n')

        y[y == 0] = np.min(y[y != 0])
        y = np.log(y)
        min_y = np.min(y)
        y = y-min_y

        median_y = np.median(y)
        y = y/median_y
        self.meta['intensity'] = {'min':min_y, 'median':median_y}

        self.save_meta_hdf5(self.meta, self.stats_path)
        print(f'post-processing value:{self.meta}')
        return np.transpose(x_t, (1, 0, 2, 3, 4)), np.transpose(y,(0,3,1,2))
    def save_meta_hdf5(self,meta, filename='meta.h5'):
        with h5.File(filename, 'w') as f:
            for key, subdict in meta.items():
                grp = f.create_group(key)
                for subkey, value in subdict.items():
                    grp.create_dataset(subkey, data=value)

if __name__ == '__main__':
    data_gen = preProcessing('/home/dc-su2/rds/rds-dirac-dr004/Magritte/random_grid64_data.hdf5','/Users/ss1421/Documents/physical_informed/data/preprocess/statistic/random.hdf5')
    x,y = data_gen.get_data()