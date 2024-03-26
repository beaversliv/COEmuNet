import h5py as h5
import numpy as np

class preProcessing:
    def __init__(self,path):
        self.path = path

    def outliers(self):
        with h5.File(self.path,'r') as sample:
            x = np.array(sample['input'],np.float32)   # shape(num_samples,3,64,64,64)
            y = np.array(sample['output'][:,:,:,15:16], np.float32)# shape(num_samples,64,64,1)
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
        meta = {}

        x_t = np.transpose(x, (1, 0, 2, 3, 4))
        for idx in [0]:
            meta[idx] = {}
            meta[idx]['mean'] = x_t[idx].mean()
            meta[idx]['std'] = x_t[idx].std()
            x_t[idx] = (x_t[idx] - x_t[idx].mean())/x_t[idx].std()
        
        for idx in [1, 2]:
            meta[idx] = {}
            meta[idx]['min'] = np.min(x_t[idx])
            meta[idx]['median'] = np.median(x_t[idx])
            x_t[idx] = np.log(x_t[idx])
            
            x_t[idx] = x_t[idx] - np.min(x_t[idx])
            x_t[idx] = x_t[idx]/np.median(x_t[idx])
        
        y[y == 0] = np.min(y[y != 0])
        y = np.log(y)
        
        y = y-np.min(y)
        y = y/np.median(y)
        return np.transpose(x_t, (1, 0, 2, 3, 4)), np.transpose(y,(0,3,1,2))

def main():
    path = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/grid64/random/batches.hdf5'
    preprocessing = preProcessing(path)
    clean_x,clean_y = preprocessing.get_data()
    print(clean_y.shape)
    with h5.File(f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/grid64/random/clean_batches.hdf5', 'w') as file:
        file['input'] = clean_x
        file['output']= clean_y
    

if __name__ == '__main__':
    main()

