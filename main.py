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
)
set_seed(42) # Set seed for reproducibility

from args import argument_parser
args = argument_parser().parse_args()

# Initialize logger
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
log = Logger()
log.open("logs/logfile.txt")

# Initialize comet_ml
#COMET_ML_API_KEY = os.getenv("COMET_ML_API_KEY")
#if COMET_ML_API_KEY is None:
#    raise ValueError("COMET_ML_API_KEY is not set.")
#comet_ml.init(project_name="your-project-name", api_key="COMET_ML_API_KEY")

model = YOLO(args.model_name)

def train(model=str, dataset_path=None, epochs=10, imgsz=640):
    yaml_files = glob.glob(os.path.join(dataset_path, '*.yaml'))
    if not yaml_files:
      raise FileNotFoundError(f"No YAML files found in {dataset_path}")
    yaml_file_path = yaml_files[0]
    print(f"Using YAML file: {yaml_file_path}")
    results = model.train(data=yaml_file_path, epochs=epochs, imgsz=imgsz)
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
        train(model_name=args.model_name, dataset_path=args.dataset_location, epochs=args.epochs, imgsz=args.imgsz)
    elif args.model_mode == "test":
        test(args.model_name, args.dataset_location)
    else:
        raise ValueError("Model mode must be either train or test.")

if __name__ == "__main__":
    main()