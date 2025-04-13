from tqdm.notebook import tqdm
from ultralytics import YOLO
import numpy as np
import comet_ml
import os
import requests
import glob

from utils import (
AverageMeter, 
Logger,
set_seed,
create_hyperparameter_yaml,
)
set_seed(42) # Set seed for reproducibility

from args import argument_parser
args = argument_parser().parse_args()

hyp_params = [
    'lr0'
    , 'lrf'
    , 'momentum'
    , 'weight_decay'
    , 'warmup_epochs'
    , 'warmup_momentum'
    , 'warmup_bias_lr'
    , 'box'
    , 'cls'
    , 'cls_pw'
    , 'obj'
    , 'obj_pw'
    , 'iou_t'
    , 'anchor_t'
    , 'fl_gamma'
    , 'hsv_h'
    , 'hsv_s'
    , 'hsv_v'
    , 'degrees'
    , 'scale'
    , 'shear'
    , 'perspective'
    , 'flipud'
    , 'fliplr'
    , 'mosaic'
    , 'mixup'
    , 'copy_paste'
    ]

# Initialize logger
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
log = Logger()
#log.open("logs/logfile.txt") # Not storing any logs for now. We'll likely just use comet_ml for logging.

# Initialize comet_ml
#COMET_ML_API_KEY = os.getenv("COMET_ML_API_KEY")
#if COMET_ML_API_KEY is None:
#    raise ValueError("COMET_ML_API_KEY is not set.")
#comet_ml.init(project_name="your-project-name", api_key="COMET_ML_API_KEY")

model = YOLO(args.model_name, 'detect')

def train(model=str, dataset_path=None, epochs=10, imgsz=640):
    data_yaml_files = glob.glob(os.path.join(dataset_path, '*.yaml'))
    if not data_yaml_files:
      raise FileNotFoundError(f"No YAML files found in {dataset_path}")
    data_yaml_file_path = data_yaml_files[0]
    print(f"For training data, using YAML file: {data_yaml_file_path}")
    hyp_yaml_file_path = yaml_path = create_hyperparameter_yaml(hyp_params)
    print(f'For hyperparameters, using YAML file: {hyp_yaml_file_path}')
    results = model.train(
      data=data_yaml_file_path, # Path to dataset config file
      epochs=epochs,
      hyp=hyp_yaml_file_path, # Path to hyperparameters config file
      imgsz=imgsz)
      # Also takes a device argument if needed, e.g. device="cpu"
    log.write(f"Results: {results}")
    return results

## TODO: refine test function
def test(model, dataset):
    results = model.test(data=dataset)
    log.write(f"Results: {results}")
    return results 

def main():
    # Main function to handle training or testing
    if args.model_mode == "train":
        train(model=model, dataset_path=args.dataset_location, epochs=args.epochs, imgsz=args.imgsz)
    elif args.model_mode == "test":
        test(args.model_name, args.dataset_location)
    else:
        raise ValueError("Model mode must be either train or test.")

if __name__ == "__main__":
    main()