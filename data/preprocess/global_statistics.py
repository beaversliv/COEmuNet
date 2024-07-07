### Find global mean, std, min and median for pre-processing####
import heapq
import json
import pickle
import h5py  as h5
import numpy as np
import time
import logging
def setup_logger(log_file):
    logger = logging.getLogger('GlobalStatsLogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
logger = setup_logger('/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/rotate/results/sql/logfile.log')
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'Function {func.__name__} took {end_time - start_time:.4f} seconds')
        return result
    return wrapper
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
        self.global_medianI = 0.0
        self.global_minT = float('inf')
        self.global_minC = float('inf')
        self.global_medianT = None
        self.global_medianC = None
        self.global_mean = 0.0
        self.global_variance = 0.0
        self.global_std = 0.0
        self.n = 0 
        self.global_median_calculator_T = OnlineMedianCalculator()
        self.global_median_calculator_C = OnlineMedianCalculator()
        self.global_median_calculator_I = OnlineMedianCalculator()
    def outlierRemover(self,input_,output_):
        y = output_.copy()
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
    def find_global_min(self,value,global_min_attr):
        """Calculate the min and update the global attribute."""
        value[value == 0] = np.min(value[value != 0])
        value = np.log(value)
        batch_min = np.min(value)

        setattr(self, global_min_attr, min(getattr(self, global_min_attr), batch_min))

    def find_global_median(self, value, global_min_attr, median_calculator):
        """Calculate the median and update the global attribute."""
        value[value == 0] = np.min(value[value != 0])
        value = np.log(value)
        min_value = value - getattr(self, global_min_attr)
        median_calculator.add_data_point(min_value)
        
    def process_stats(self, value, global_min_attr, median_calculator, is_first_pass):
        if is_first_pass:
            self.find_global_min(value, global_min_attr)
        else:
            self.find_global_median(value, global_min_attr, median_calculator)

    def veloStats(self,value):
        """Calculate the mean and std values in a batch, updating global."""
        batch_mean = np.mean(value)
        batch_std = np.std(value)
        batch_n = value.size 

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
    @timing_decorator
    def calculator(self):
        # First pass to determine global min values
        for file in self.files:
            logger.info(f'Find min {file}')
            with h5.File(file, 'r') as h5f:
                input_   = np.array(h5f['input'],np.float64)
                output_ = np.array(h5f['output'],np.float64)
            # remove outlier
            input_data, output_data = self.outlierRemover(input_,output_)
            num_samples = input_data.shape[0]
            for i in range(0,num_samples,self.batch_size):
                print(f'batch {i} - {i+self.batch_size}')
                input_batch = input_data[i:i + self.batch_size]
                output_batch = output_data[i:i + self.batch_size]

                x_t = np.transpose(input_batch, (1, 0, 2, 3, 4))
                # velocity stats
                xv  = x_t[0]
                self.veloStats(xv)
                temp = x_t[1]
                co = x_t[2]
                self.process_stats(temp, 'global_minT', self.global_median_calculator_T, is_first_pass=True)
                self.process_stats(co, 'global_minC', self.global_median_calculator_C, is_first_pass=True)
                self.process_stats(output_batch, 'global_minI', self.global_median_calculator_I, is_first_pass=True)

        self.finalize_std()
        # Second pass to determine global median values
        for file in self.files:
            print('find median',file)
            with h5.File(file, 'r') as h5f:
                input_ = np.array(h5f['input'], np.float64)
                output_ = np.array(h5f['output'], np.float64)

            input_data, output_data = self.outlierRemover(input_, output_)
            num_samples = input_data.shape[0]
            for i in range(0, num_samples, self.batch_size):
                print(f'batch {i} - {i+self.batch_size}')
                input_batch = input_data[i:i + self.batch_size]
                output_batch = output_data[i:i + self.batch_size]

                x_t = np.transpose(input_batch, (1, 0, 2, 3, 4))
                temp = x_t[1]
                co = x_t[2]
                self.process_stats(temp, 'global_minT', self.global_median_calculator_T, is_first_pass=False)
                self.process_stats(co, 'global_minC', self.global_median_calculator_C, is_first_pass=False)
                self.process_stats(output_batch, 'global_minI', self.global_median_calculator_I, is_first_pass=False)

        self.global_medianT = self.global_median_calculator_T.get_median()
        self.global_medianC = self.global_median_calculator_C.get_median()
        self.global_medianI = self.global_median_calculator_I.get_median()

    def get_global_stats(self):
        meta = {
            'vel':{'mean':self.global_mean,'std':self.global_std},
            'temp':{'min':self.global_minT,'median':self.global_medianT},
            'co':{'min':self.global_minC,'median':self.global_medianC},
            'y':{'min':self.global_minI,'median':self.global_medianI}
        }
        print('Global statistics:',meta)
        return meta
    def save_meta_hdf5(self,filename='meta.h5'):
        meta = self.get_global_stats()
        with h5.File(filename, 'w') as f:
            for key, subdict in meta.items():
                grp = f.create_group(key)
                for subkey, value in subdict.items():
                    grp.create_dataset(subkey, data=value)

def main(file_paths:list,batch_size):
    global_calculator = GlobalStatsCalculator(file_paths,batch_size)
    global_calculator.calculator()
    global_calculator.save_meta_hdf5('/Users/ss1421/Documents/physical_informed/data/preprocess/statistic/dummy.hdf5')
if __name__ == '__main__':
    file_paths = file_paths = ['/home/dc-su2/rds/rds-dirac-dr004/Magritte/dummy.hdf5',
                    '/home/dc-su2/rds/rds-dirac-dr004/Magritte/dummy1.hdf5']              
    main(file_paths,1000)


  