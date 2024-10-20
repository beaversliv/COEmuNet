from outlier import outlier_main
from stats import stats_main
from transform  import transform_main
from train_test_split import train_test_split_main
from utils  import Logging
import argparse
import json
def parse_args():
    parser = argparse.ArgumentParser(description="Select dataset and override config options")
    parser.add_argument('--dataset', choices=['random','mulfreq', 'faceon'], required=True,
                        help="Specify the dataset to use: 'random', 'mulfreq', or 'faceon'")
    parser.add_argument('--num_rotations',type=int,
                        help="number of rotations in dataset")
    return parser.parse_args()
def data_clean_pipeline():
    args = parse_args()
    dataset_name = args.dataset
    num_rotations = args.num_rotations
    logger = Logging(file_dir=f"aluxinary_files", file_name=f"extra_{dataset_name}.log")
    non_outlier_file_path = f'aluxinary_files/extra_{dataset_name}_non_outliers.json'
    clean_file_path       = f"aluxinary_files/extra_{dataset_name}_clean_files.json"
    stats_file            = f'data/preprocess/statistic/new_{dataset_name}.hdf5'
    
    outlier_main(dataset_name,num_rotations,logger,non_outlier_file_path)
    stats_main(dataset_name,logger,non_outlier_file_path,stats_file)
    transform_main(dataset_name,non_outlier_file_path,clean_file_path,stats_file,logger)
    train_test_split_main(dataset_name,clean_file_path,logger)

if __name__ == '__main__':
    data_clean_pipeline()

