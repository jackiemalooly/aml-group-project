import os
import sys 
import yaml
import json
import torch
import shutil
import numpy as np 
from torch import nn
import torch.nn.functional as F 
from sklearn.metrics import f1_score
from torch.autograd import Variable
from tqdm.notebook import tqdm
import math
import random

# evaluate meters
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# print logger
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: 
            is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy module
    torch.manual_seed(seed_value)  # Sets the seed for generating random numbers for PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    print(f"Random seed set to: {seed_value}")

def create_hyperparameter_yaml(args, include_only=None):
    """
    Create a YAML file with hyperparameters based on provided arguments.
    
    Args:
        args: An argument namespace or dictionary containing experiment_name and other hyperparameters
        include_only: List of argument names to include in the YAML file (None means include all)
        
    Returns:
        str: Path to the created YAML file
    """
    if not isinstance(args, dict):
        args_dict = vars(args)
    else:
        args_dict = args
    
    experiment_name = args_dict.get('experiment_name', 'default')
    
    hyperparameters = {}
    for key, value in args_dict.items():
        # Include only --hyp keys if requested
        if include_only is not None:
            if key in include_only:
                hyperparameters[key] = value
        elif key != 'experiment_name' and not key.startswith('_'):
            hyperparameters[key] = value
    
    # Write the yaml file
    yaml_path = f"hyp.{experiment_name}.yaml"
    with open(yaml_path, 'w') as file:
        yaml.dump(hyperparameters, file, sort_keys=False)

    return yaml_path
