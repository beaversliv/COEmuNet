import argparse
import os
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--seed',type = int, default = 1223)
    parser.add_argument('--dataset', type = str, default = 'p3droslo')
    parser.add_argument('--model_grid',type=int,default= 64,help='grid of hydro model:[32,64,128]')
    parser.add_argument('--save_path',type =str, default = '/home/dc-su2/rds/rds-dirac-dp225-5J9PXvIKVV8/3DResNet/grid64/original/results/')
    parser.add_argument('--logfile',type = str, default = 'log_file.txt')
    parser.add_argument('--model_name', type = str, default = 'model.pth')
    parser.add_argument('--history', type = str, default = 'history.pkl')
    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--lr_decay', type = float, default = 0.95)
    parser.add_argument('--alpha', type = float, default = 0.8)
    parser.add_argument('--beta', type = float, default = 0.2)


    args = parser.parse_args()    
    config = OrderedDict([
            ('path_dir', args.path_dir),
            ('seed', args.seed),
            ('dataset', args.dataset),
            ('model_grid', args.model_grid),
            ('save_path',args.save_path),
            ('logfile',args.logfile),
            ('model_name', args.model_name),
            ('history', args.history),
            ('epochs', args.epochs),
            ('batch_size', args.batch_size),
            ('lr', args.lr),
            ('lr_decay', args.lr_decay),
            ('alpha', args.alpha),
            ('beta', args.beta)
            ])
    
    return config

def faceon_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--exp_name',type=str,default='faceOn_grid64',help='name of the experiment')
    parser.add_argument('--model', type = str, default = '3dResNet',help='network name: [3dResNet, so3]')
    parser.add_argument('--model_grid',type=int,default= 64,help='grid of hydro model:[32,64,128]')
    parser.add_argument('--dataset', type = str, default = 'faceon',help='observe direction:[faceon,random]')
    parser.add_argument('--epochs', type = int, default = 1000,help='number of epochs')
    parser.add_argument('--batch_size', type = int, default = 64,help='number of batch size')
    parser.add_argument('--lr', type = float, default = 1e-3,help='learning rate')
    parser.add_argument('--lr_decay', type = float, default = 0.95)
    parser.add_argument('--seed',type = int, default=1234,help='random seed')
    parser.add_argument('--num_freqs',type = int, default=31,help='number of frequency')

    args = parser.parse_args()
    
    config = OrderedDict([
            ('path_dir', args.path_dir),
            ('exp_name', args.exp_name),
            ('model', args.model),
            ('model_grid', args.model_grid),
            ('dataset', args.dataset),
            ('epochs', args.epochs),
            ('batch_size', args.batch_size),
            ('lr', args.lr),
            ('lr_decay', args.lr_decay),
            ('seed', args.seed),
            ('num_freqs', args.num_freqs)
            ])
    return config
