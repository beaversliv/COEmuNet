import numpy as np
import h5py as h5
import os
import json
from mpi4py import MPI

class PathFind:
    """
    A class to generate paths for models and corresponding rotated data files.

    Attributes:
    ----------
    num_rotations : int
        Number of rotations for each model.
    gen_path : str
        Path suffix to generate the rotation data path.
    
    Methods:
    -------
    model_find():
        Returns a list of absolute paths for all original models (xxxx.hdf5).
    data_path_files(model_file, rotation_idx):
        Generates the file path for the rotated data.
    file_paths_gen():
        Generates and returns a list of all file paths for the rotated data.
    """
    
    def __init__(self, num_rotations, gen_path):
        """
        Initializes the PathFind class with the number of rotations and the path for generating files.

        Parameters:
        ----------
        num_rotations : int
            Number of rotations for each model.
        gen_path : str
            Path suffix to generate the rotation data path.
        """
        self.num_rotations = num_rotations
        self.gen_path = gen_path

    def model_find(self):
        """
        Finds and returns absolute paths of all original Magritte models (xxxx.hdf5).

        Returns:
        -------
        model_files : list
            A list of absolute paths to the model files.
        """
        model_files = []
        base_path = "/home/dc-su2/rds/rds-dirac-dp012/dc-su2/AMRVAC_3D/"

        for model_dir in os.listdir(base_path):
            model_path = os.path.join(base_path, model_dir)
            for file_dir in os.listdir(model_path):
                current_path = os.path.join(model_path, file_dir)
                model_file = os.path.join(current_path, f'{file_dir}.hdf5')
                model_files.append(model_file)
        
        return model_files

    def data_path_files(self, model_file, rotation_idx):
        """
        Generates the file path for a specific rotation data.

        Parameters:
        ----------
        model_file : str
            The original model file path.
        rotation_idx : int
            The index of the rotation.

        Returns:
        -------
        rotation_file : str
            The generated file path for the rotated data.
        """
        listb = model_file.split('/')
        # Insert the gen_path into the appropriate position in the path
        listb.insert(6, self.gen_path)
        # Modify the file name to reflect the rotation index
        listb[-1] = f'{listb[-2]}_{rotation_idx}.hdf5'
        rotation_file = '/'.join(listb)
        
        return rotation_file

    def file_paths_gen(self):
        """
        Generates and returns a list of all file paths for the rotated data.

        Returns:
        -------
        rotation_files : list
            A list of file paths for the rotated data.
        """
        model_files = self.model_find()
        # model_files = model_files[:2]  # Process only the first 1000 files for now
        
        rotation_files = []
        for idx in range(len(model_files) * self.num_rotations):
            file_idx = idx // self.num_rotations
            rotation_idx = idx % self.num_rotations
            model_file = model_files[file_idx]
            rotation_file = self.data_path_files(model_file, rotation_idx)
            rotation_files.append(rotation_file)
        
        return rotation_files

def outlier_detection(rotation_file):
    with h5.File(rotation_file,'r') as f:
        y = np.array(f['I'][:,:,15:16],dtype=np.float32) # (64,64,1)
    y[y == 0.0] = np.min(y[y != 0.0])
    y = np.log(y)
    return np.min(y) <= -50

def main(dataset_name):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rotation_files_find = PathFind(num_rotations=100,gen_path=f'physical_forward/{dataset_name}/grid64')
    model_files = rotation_files_find.file_paths_gen()

    # Split the file list among the available processes
    files_per_proc = len(model_files) // size
    remainder = len(model_files) % size

    if rank < remainder:
        start_idx = rank * (files_per_proc + 1)
        end_idx = start_idx + files_per_proc + 1
    else:
        start_idx = rank * files_per_proc + remainder
        end_idx = start_idx + files_per_proc
    print(f'total num jobs: {len(model_files)}, process {rank} do jobs {start_idx} - {end_idx}')
    local_non_outlier_files = []
    for i in range(start_idx, end_idx):
        model_file = model_files[i]
        if not outlier_detection(model_file):
            local_non_outlier_files.append(model_file)

    # Gather all non-outlier files from all processes
    all_non_outlier_files = comm.gather(local_non_outlier_files, root=0)

    # Root process collects and flattens the list
    if rank == 0:
        non_outlier_files = [file for sublist in all_non_outlier_files for file in sublist]
        print(f"Total non-outlier files collected: {len(non_outlier_files)}")
        # Optionally, save the list of non-outliers to a file
        with open(f'data/preprocess/statistic/{dataset_name}_non_outliers.json', 'w') as f:
            json.dump(non_outlier_files, f)

if __name__ == "__main__":
    dataset_name = 'mulfreq'
    # dataset_name = 'random'
    main(dataset_name)