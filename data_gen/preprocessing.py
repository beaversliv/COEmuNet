import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
# read data
path = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/data_augment/rotate1200.hdf5'
with h5.File(path,'r') as sample:
    x = np.array(sample['input'],np.float32)   # shape(100,3,64,64,64)
    y = np.array(sample['output'][:,:,:,15:16], np.float32)# shape(100,64,64,1)
    freqs = np.array(sample['nfreqs'],np.float32)

# take logrithm
y_v = y.reshape(-1)
y = np.where(y == 0, np.min(y_v[y_v != 0]), y)
I = np.log(y)
# difference = max - min
max_values = np.max(I,axis=(1,2))
min_values = np.min(I,axis=(1,2))
diff = max_values - min_values
# find outliers
outlier_idx = np.where(diff>20)[0]

# remove outliers
clean_x = np.delete(x,outlier_idx,axis=0)
clean_y = np.delete(y,outlier_idx,axis=0)

with h5.File(f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/data_augment/clean_rotate1200.hdf5', 'w') as file:
    file['input'] = clean_x
    file['output']= clean_y
    file['nfreqs'] = freqs
