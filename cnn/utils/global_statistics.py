### Find global mean, std, min and median for pre-processing####
import heapq
import json
import pickle
import h5py  as h5
import numpy as np
class OnlineMedianCalculator:
    """Helper class to maintain the running median using two heaps."""
    def __init__(self):
        self.min_heap = []  # Min heap for the upper half
        self.max_heap = []  # Max heap for the lower half

    def add_data_point(self, values):
        """Add a batch of data points to the online median calculator."""
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
        """Get the current median of all added data points."""
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        else:
            return -self.max_heap[0]

class GlobalStatsCalculator:
    def __init__(self,files:list,batch_size:int):
        self.files = files
        self.batch_size = batch_size
        self.global_minI = float('inf')
        self.global_maxI = float('-inf')
        self.global_minT = float('inf')
        self.global_minC = float('inf')
        self.global_medianT = None
        self.global_medianC = None
        self.global_mean = 0.0
        self.global_variance = 0.0
        self.global_std = 0.0
        self.n = 0 
        self.global_median_calculator = OnlineMedianCalculator()
    def outlierRemover(self,output_):
        # removed outlier based on middle frequency
        y = output_[:,:,:,15]
        y[y==0] = np.min(y[y!=0])
        I = np.log(y)
        # difference = max - min
        max_values = np.max(I,axis=(1,2))
        min_values = np.min(I,axis=(1,2))
        diff = max_values - min_values
        # find outliers
        outlier_idx = np.where(min_values < -60)[0]

        # remove outliers
        removed_x = np.delete(input_,outlier_idx,axis=0)
        removed_y = np.delete(output_,outlier_idx,axis=0)
        return removed_x, removed_y

    def intensityStats(self,value):
        """Calculate the min and max values in a batch, updating global min/max."""
        vvalue[value==0] = np.min(value[value!=0])
        value = np.log(value)
        CMB = -40.2927
        value[value<=CMB] = CMB

        batch_min = np.min(value)
        batch_max = np.max(value)

        self.global_minI = min(self.global_minI, batch_min)
        self.global_maxI = max(self.global_maxI, batch_max)

    def veloStats(self,value):
        """Calculate the mean and std values in a batch, updating global."""
        batch_mean = np.mean(value)
        batch_std = np.std(value)

        # Update global mean using Welford's incremental method
        new_n = self.n + batch_n
        delta = batch_mean - self.global_mean
        self.global_mean += delta * batch_n / new_n

        # Update global variance
        batch_variance = np.var(value)
        m2_batch = batch_variance * batch_n
        self.global_variance += m2_batch + delta**2 * self.n * batch_n / new_n

        self.n = new_n
    def finalize_std(self):
        """Calculate the final standard deviation after all batches have been processed."""
        if self.n > 1:
            self.global_std = np.sqrt(self.global_variance / (self.n - 1))
        else:
            self.global_std = 0.0

    def process_stats(self, value, global_min_attr):
        """Calculate the min and update the global attribute."""
        value[value == 0] = np.min(value[value != 0])
        value = np.log(value)
        batch_min = np.min(value)

        setattr(self, global_min_attr, min(getattr(self, global_min_attr), batch_min))
        self.global_median_calculator.add_data_point(value)

    def tempStats(self, value):
        """Calculate min and median values for temperature data."""
        self.process_stats(value, 'global_minT')
        self.global_medianT = self.global_median_calculator.get_median()

    def coStats(self, value):
        """Calculate min and median values for CO data."""
        self.process_stats(value, 'global_minC')
        self.global_medianC = self.global_median_calculator.get_median()

    def calculator(self):

        for file in files:
            with h5.File(file, 'r') as h5f:
                input_   = np.array(h5f['input'],np.float64)
                output_ = np.array(h5f['output'],np.float64)

            # remove outlier
            input_data, output_data = self.outlierRemover(output_)

            num_samples = x.shape[0]
            for i in range(0,num_samples,self.batch_size):
                input_batch = input_data[i:i + self.batch_size]
                output_batch = output_data[i:i + self.batch_size]

                x_t = np.transpose(input_batch, (1, 0, 2, 3, 4))
                # velocity stats
                xv  = x_t[0]
                vbatch = xv[i:i + batch_size]
                self.veloStats(vbatch)
                
                # temperature and co stats
                temp  = x_t[1]
                tbatch = temp[i:i + batch_size]
                self.tempStats(tbatch)

                co  = x_t[1]
                tbatch = co[i:i + batch_size]
                self.coStats(tbatch)
                
                # intensity stats
                self.intensityStats(output_batch)

    def get_global_stats(self):
        return {
            "global_mean": self.global_mean,
            "global_std": self.global_std,
            'global_minT': self.global_minT,
            "global_medianT": self.global_medianT,
            'global_minC': self.global_minC,
            "global_medianC": self.global_medianC,
            "global_minI": self.global_minI,
            "global_maxI": self.global_maxI
        }

def main():
    # file_paths = [f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/Batches/batch_{rank}.hsdf5' for rank in range(2)]
    file_paths = [f'/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/data_augment/clean_rotate12000_{i}.hdf5' for i in range(5)]
    # file_paths += ['/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/faceon/new_test.hdf5',
    #                     '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/faceon/new_vali.hdf5']
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
    with open('/home/dc-su2/physical_informed/cnn/rotate/12000_statistics.pkl','wb') as file:
        pickle.dump(statistics,file)

if __name__ == '__main__':
    main()


  