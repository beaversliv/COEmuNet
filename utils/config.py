import argparse
import os
from collections import OrderedDict
import yaml
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/dc-su2/physical_informed/cnn/config/faceon_dataset.yaml', help='Path to the YAML configuration file')
    parser.add_argument('--grid',type=int,help='grid of hydro model:[32,64,128]')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--seed', type=int, help='Override seed')
    parser.add_argument('--df',type=float)

    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--alpha', type=float, help='weight for feature loss')
    parser.add_argument('--beta', type=float, help='weight for MSE')
    parser.add_argument('--weight_decay', type=float, help='regulerizaton')
    
    parser.add_argument('--save_path', type=str,help='path for history.png, history.pkl and model.pth')
    parser.add_argument('--log_file', type=str,help='training history')
    parser.add_argument('--model_file', type=str,help='saved model name')
    parser.add_argument('--pkl_file', type=str,help='saved target and pred in pkl')
    parser.add_argument('--history_img', type=str,help='train test value vs epoch, history.png')
    parser.add_argument('--img_dir', type=str,help='save target img vs pred img in which dir')
    
    args = parser.parse_args()
    return args
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def merge_config(args, config):
    if args.grid is not None:
        config['dataset']['grid'] = args.grid
    if args.batch_size is not None:
        config['dataset']['batch_size'] = args.batch_size
    if args.seed is not None:
        config['model']['seed'] = args.seed
    if args.epochs is not None:
        config['model']['epochs'] = args.epochs
    if args.alpha is not None:
        config['model']['alpha'] = args.alpha
    if args.beta is not None:
        config['model']['beta']  = args.beta
    if args.weight_decay is not None:
        config['optimizer']['weight_decay'] = args.weight_decay
    if args.save_path is not None:
        config['output']['save_path'] = args.save_path
    if args.log_file is not None:
        config['output']['log_file'] = args.log_file
    if args.model_file is not None:
        config['output']['model_file'] = args.model_file
    if args.pkl_file is not None:
        config['output']['pkl_file'] = args.pkl_file 
    if args.history_img is not None:
        config['output']['history_img'] = args.history_img
    if args.img_dir is not None:
        config['output']['img_dir'] = args.img_dir
     
    if 'optimizer' in config and config['optimizer']['type'] == 'adam':
        if args.lr:
            config['optimizer']['params']['lr'] = args.lr
    config = OrderedDict(config)
    return config
if __name__ == "__main__":
     args = parse_args()
     config = load_config('/Users/ss1421/Documents/physical_informed/cnn/config/faceon_dataset.yaml')
     config = merge_config(args, config)
     print(config)


