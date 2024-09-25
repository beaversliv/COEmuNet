from ruamel.yaml import YAML
import os

def check_file(file_path):
    return os.path.isfile(file_path)
def load_yaml(path):
    with open(path, 'rb') as f:
        yaml = YAML()
        dt = yaml.load(f)
        return dt
class Logging:
    def __init__(self, file_dir:str, file_name:str):
        try:
            # Attempt to create the directory if it does not exist
            os.makedirs(file_dir, exist_ok=True)  # exist_ok=True avoids errors if directory already exists
        except Exception as e:
            print(f"Warning: Failed to create directory {file_dir}. {e}")
        # if not os.path.exists(file_dir):
        #     os.mkdir(file_dir)
        self.log_file = os.path.join(file_dir, file_name)
        open(self.log_file, 'a').close()

    def info(self, message, gpu_rank=0, console=True):
        # only log rank 0 GPU if running with multiple GPUs/multiple nodes.
        if gpu_rank is None or gpu_rank == 0:
            if console:
                print(message)

            with open(self.log_file, 'a') as f:  # a for append to the end of the file.
                print(message, file=f)