from outlier import outlier_main
from stats import stats_main
from transform  import transform_main
from train_test_split import train_test_split_main
from utils  import Logging
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Select dataset and override config options")
    parser.add_argument('--dataset', choices=['random', 'dummy','mulfreq', 'faceon'], required=True,
                        help="Specify the dataset to use: 'random', 'mulfreq', or 'faceon'")
    return parser.parse_args()
def data_clean_pipeline():
    dataset_name = 'dummy'
    logger = Logging(file_dir=f"aluxinary_files", file_name=f"{dataset_name}.log")
    non_outlier_file_path = f'aluxinary_files/{dataset_name}_non_outliers.json'
    clean_file_path       = f"aluxinary_files/{dataset_name}_clean_files.json"
    outlier_main(dataset_name,logger,non_outlier_file_path)
    stats_main(dataset_name,logger,non_outlier_file_path)
    transform_main(dataset_name,logger)
    train_test_split_main(clean_file_path,logger)

if __name__ == '__main__':
    data_clean_pipeline()

