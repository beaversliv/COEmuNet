import argparse
import os
from collections import OrderedDict
import yaml
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. Got {value}.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/mulfreq_dataset.yaml', help='Path to the YAML configuration file')
    parser.add_argument('--grid',type=int,choices=[32,64,128],help='grid of hydro model:[32,64,128]')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--seed', type=int, help='Override seed')

    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--alpha', type=float, help='weight for MSE ,then (1-alpha) is the weight for frequency loss')
    parser.add_argument('--stage',type=int,choices=[1,2],help='stage 1: start from pretrained faceon model; stage 2: load whole model')
    parser.add_argument('--resume_checkpoint',type=str,help='checkpoint path')

    parser.add_argument("--use_scheduler", type=str_to_bool, default=None, help="Enable or disable the scheduler")
    parser.add_argument("--scheduler_type", type=str, choices=["StepLR","CosineAnnealingLR"], default="StepLR", help="Override scheduler type (e.g., stepLR, cosineAnnealingLR)")
    parser.add_argument("--step_size", type=int, default=50, help="Step size for StepLR")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma value for StepLR")
    parser.add_argument("--T_max", type=int, default=20, help="Maximum number of iterations for CosineAnnealingLR")
    
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
    if args.stage is not None:
        config['model']['stage'] = args.stage
    if args.resume_checkpoint is not None:
        config['model']['resume_checkpoint'] = args.resume_checkpoint

    config["use_scheduler"] = args.use_scheduler if args.use_scheduler is not None else config.get("use_scheduler", True)

    # If scheduler is disabled, set it to None
    if not config["use_scheduler"]:
        config["scheduler"] = None
    else:
        # Dynamically select or override scheduler configuration
        selected_type = args.scheduler_type or config.get('scheduler_type', config["scheduler"][0]["type"])
        scheduler_config = None

        # Search for the matching scheduler type in the list
        for sched in config["scheduler"]:
            if sched["type"] == selected_type:
                scheduler_config = sched
                break

        if scheduler_config is None:
            raise ValueError(f"Scheduler type {selected_type} not found in config.")

        # Override parameters via command-line arguments
        if args.step_size and "step_size" in scheduler_config["params"]:
            scheduler_config["params"]["step_size"] = args.step_size
        if args.gamma and "gamma" in scheduler_config["params"]:
            scheduler_config["params"]["gamma"] = args.gamma
        if args.T_max and "T_max" in scheduler_config["params"]:
            scheduler_config["params"]["T_max"] = args.T_max

        # Replace config["scheduler"] with the selected scheduler
        config["scheduler"] = scheduler_config
    
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


