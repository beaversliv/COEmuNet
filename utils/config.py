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

    config["use_checkpoint"] = args.use_checkpoint if args.use_checkpoint is not None else config.get("use_checkpoint", True)
    if not config["use_checkpoint"]:
        config['resume_checkpoint'] = None
    else:
        config['resume_checkpoint'] = args.resume_checkpoint

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
    
    config["ddp_on"] = args.ddp_on if args.ddp_on is not None else config.get("ddp_on", True)
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
    # clean config with only selected scheduler + sampling
    clean_config = {
        "model": config.get("model", {}),
        "optimizer": config.get("optimizer", {}),
        "scheduler": config["scheduler"] if config.get("use_scheduler", True) else None,
        "dataset": config.get("dataset", {}),
        "output": config.get("output", {}),
        "ddp_on": config.get("ddp_on", True),
        "use_checkpoint":config.get("use_checkpoint", True),
        "resume_checkpoint":config["resume_checkpoint"] if config.get("use_checkpoint", True) else None
    }

    clean_config = OrderedDict(clean_config)
    return clean_config


