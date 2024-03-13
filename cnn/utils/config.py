import argparse
import os

def faceon_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--exp_name',type=str,default='faceOn_grid64',help='name of the experiment')
    parser.add_argument('--model', type = str, default = '3dResNet',help='network name: [3dResNet, so3]')
    parser.add_argument('--model_grid',type=int,default= 64,help='grid of hydro model:[32,64,128]')
    parser.add_argument('--dataset', type = str, default = 'faceon',help='observe direction:[faceon,random]')
    parser.add_argument('--epochs', type = int, default = 100,help='number of epochs')
    parser.add_argument('--batch_size', type = int, default = 64,help='number of batch size')
    parser.add_argument('--lr', type = float, default = 1e-3,help='learning rate')
    parser.add_argument('--lr_decay', type = float, default = 0.95)
    parser.add_argument('--seed',type = int, default=1234,help='random seed')
    return parser

def rotation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--model', type = str, default = '3dResNet')
    parser.add_argument('--dataset', type = str, default = '/home/dc-su2/rds/rds-dirac-dp147/vtu_oldmodels/Magritte-examples/physical_forward/cnn/grid64/rotation/clean_rotate12000_0.hdf5')
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--lr_decay', type = float, default = 0.95)
    parser.add_argumeny('--seed',type = int, default=1234)
    parser.add_argumeny('--histroy',type = str, default='/home/dc-su2/physical_informed/cnn/rotate/results/history.pkl')
    return parser

def equivariant_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--model_name', type = str, default = 'so3')
    parser.add_argument('--dataset', type = str, default = 'rotation')
    parser.add_argument('--epochs', type = int, default = 50)
    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--lr_decay', type = float, default = 0.95)
    parser.add_argumeny('--seed',type = int, default=1234)
    parser.add_argumeny('--histroy',type = str, default='/home/dc-su2/physical_informed/cnn/steerable/history.pkl')
    return parser
