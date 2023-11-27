### Find global mean, std, min and median for pre-processing####
import heapq
import json
import pickle
import h5py  as h5
import numpy as np
class OnlineMedianCalculator:
    def __init__(self):
        self.min_heap = []  # Min heap for the upper half
        self.max_heap = []  # Max heap for the lower half

    def add_data_point(self, values):
        for value in np.nditer(values):
        
            if not self.max_heap or value <= -self.max_heap[0]:
                heapq.heappush(self.max_heap, -value)
            else:
                heapq.heappush(self.min_heap, value)

            # Balance the heaps
            if len(self.max_heap) > len(self.min_heap) + 1:
                heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
            elif len(self.min_heap) > len(self.max_heap):
                heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def get_median(self):
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        else:
            return -self.max_heap[0]
        
def global_vz(file_paths, chunk_size=1000):
    global_mean = 0
    global_sum_squared_diff = 0
    global_count = 0

    for file_path in file_paths:
        with h5.File(file_path,'r') as file:
            x = np.array(file['input'][:,0,:,:,:])
            for i in range(0,x.shape[0],chunk_size):
                chunk = x[i:i+chunk_size]

                # Update global statistics for the current chunk
                chunk_mean = np.mean(chunk)
                global_sum_squared_diff += np.sum((chunk - chunk_mean) ** 2)

                global_count += chunk_size

    # Calculate global mean, std, and median
    global_mean = global_sum_squared_diff / global_count
    global_std = np.sqrt(global_mean)
    # global_median = calculate_global_median(file_paths)

    return global_mean, global_std

def calculate_global_statistics(file_paths, chunk_size=1000,order=1,read_x_condition=True):
    median_calculator = OnlineMedianCalculator()
    global_min = float('inf')

    for file_path in file_paths:
        with h5.File(file_path,'r') as file:
            # read temp, co or Intensity
            if read_x_condition:
                dataset = np.array(file['input'][:,order,:,:,:])
            else:
                dataset = np.array(file['output'])
                # assign zero intensity to the minimum value of nonzero
                dataset[dataset==0.0] = np.min(dataset[dataset!=0.0])

            for i in range(0,dataset.shape[0],chunk_size):
                chunk = dataset[i:i+chunk_size]

                # Update global statistics for the current chunk
                chunk_min = np.min(np.log(chunk))
                global_min = min(global_min, chunk_min)
                for value in chunk:
                    diff = np.log(value) - chunk_min
                    median_calculator.add_data_point(diff)

    return global_min, median_calculator.get_median()

def load(file_paths):
    xs = []
    ys = []
    for file_path in file_paths:
        print('process :',file_path)
        with h5.File(file_path,'r') as file:
            # read temp, co or Intensity 
            x = np.array(file['input'], dtype=np.float32)
            y = np.array(file['output'],dtype=np.float32)
            # assign zero intensity to the minimum value of nonzero
            y[y==0.0] = np.min(y[y!=0.0])
        xs.append(x)
        ys.append(y)
    XS = np.concatenate(xs)
    YS = np.concatenate(ys)
    # vz
    vz_mean = np.mean(XS[:,0,:,:,:],dtype=np.float32)
    vz_std  = np.std(XS[:,0,:,:,:], dtype=np.float32)
    # temp
    temp_min = np.min(np.log(XS[:,1,:,:,:]))
    temp_median = np.median(np.log(XS[:,1,:,:,:]) - temp_min)
    # co
    co_min = np.min(np.log(XS[:,2,:,:,:]))
    co_median = np.median(np.log(XS[:,2,:,:,:]) - co_min)
    # intensity
    y_min = np.min(np.log(YS))
    y_median = np.median(np.log(YS) - y_min)

    return vz_mean,vz_std,temp_min,temp_median,co_min,co_median,y_min,y_median
def main():
    # file_paths = [f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/Batches/batch_{rank}.hsdf5' for rank in range(2)]
    file_paths = ['/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/Batches/rotate_1200.hdf5']
    vz_mean,vz_std,temp_min,temp_median,co_min,co_median,y_min,y_median = load(file_paths)
    # vz_mean,vz_std = global_vz(file_paths, chunk_size=1000)
    # temp_min,temp_median = calculate_global_statistics(file_paths, chunk_size=1000,order=1,read_x_condition=True)
    # co_min,co_median = calculate_global_statistics(file_paths, chunk_size=1000,order=2,read_x_condition=True)
    # y_min,y_median = calculate_global_statistics(file_paths, chunk_size=1000,order=1,read_x_condition=False)

    statistics = {
        'vz': [vz_mean,vz_std],
        'temp': [temp_min,temp_median],
        'co': [co_min,co_median],
        'y': [y_min,y_median]
    }
    print('writing statistic values')
    with open('/home/dc-su2/physical_informed/cnn/rotate/rotate1200_statistics.pkl','wb') as file:
        pickle.dump(statistics,file)

if __name__ == '__main__':
    main()


  