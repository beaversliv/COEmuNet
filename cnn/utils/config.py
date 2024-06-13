import argparse
import os
from collections import OrderedDict
import yaml

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path_dir', type = str, default = os.getcwd())
#     parser.add_argument('--seed',type = int, default = 1234)
#     parser.add_argument('--dataset', type = str, default = 'faceon',help='type of dataset:[faceon,random,freq]')
#     parser.add_argument('--model_grid',type=int,default= 64,help='grid of hydro model:[32,64,128]')
#     parser.add_argument('--save_path',type =str, default = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/results/')
#     parser.add_argument('--logfile',type = str, default = 'log_file.txt')
#     parser.add_argument('--model_name', type = str, default = 'model.pth')
#     parser.add_argument('--history', type = str, default = 'history.pkl')
#     parser.add_argument('--epochs', type = int, default = 1000)
#     parser.add_argument('--batch_size', type = int, default = 128,help='batch size per script')
#     parser.add_argument('--lr', type = float, default = 1e-3)
#     parser.add_argument('--lr_decay', type = float, default = 0.95)
#     parser.add_argument('--alpha', type = float, default = 0.8, help='weight for feature loss')
#     parser.add_argument('--beta', type = float, default = 0.2,help='weight for MSE') 


#     args = parser.parse_args()    
#     config = OrderedDict([
#             ('path_dir', args.path_dir),
#             ('seed', args.seed),
#             ('dataset', args.dataset),
#             ('model_grid', args.model_grid),
#             ('save_path',args.save_path),
#             ('logfile',args.logfile),
#             ('model_name', args.model_name),
#             ('history', args.history),
#             ('epochs', args.epochs),
#             ('batch_size', args.batch_size),
#             ('lr', args.lr),
#             ('lr_decay', args.lr_decay),
#             ('alpha', args.alpha),
#             ('beta', args.beta)
#             ])
    
#     return config
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/dc-su2/physical_informed/cnn/config/faceon_dataset.yaml', help='Path to the YAML configuration file')
    parser.add_argument('--grid',type=int,help='grid of hydro model:[32,64,128]')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--seed', type=int, help='Override seed')

    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--alpha', type=float, help='weight for feature loss')
    parser.add_argument('--beta', type=float, help='weight for MSE')
    
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--logfile', type=str,help='training history')
    parser.add_argument('--model_name', type=str,help='saved model name')
    parser.add_argument('--results', type=str,help='saved target and pred')
    
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
    if args.save_path is not None:
        config['output']['save_path'] = args.save_path
    if args.logfile is not None:
        config['output']['logfile'] = args.logfile
    if args.model_name is not None:
        config['output']['model_name'] = args.model_name
    if args.results is not None:
        config['output']['results'] = args.results  
    if 'optimizer' in config and config['optimizer']['type'] == 'adam':
        if args.lr:
            config['optimizer']['params']['lr'] = args.lr

    config = OrderedDict(config)
    return config
if __name__ == "__main__":
     args = parse_args()
     config = load_config(args.config)
     config = merge_config(args, config)
     print(config)


