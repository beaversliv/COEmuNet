import argparse
import os

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
