from tqdm.notebook import tqdm
from ultralytics import YOLO
from ultralytics.utils import yaml_load
import numpy as np
import comet_ml
import os
import requests
import glob
from ultralytics.utils.custom_detection_loss import CustomDetectionLoss 

from utils import (
AverageMeter, 
Logger,
set_seed,
)
set_seed(42) # Set seed for reproducibility

from args import argument_parser
args = argument_parser().parse_args()
print(args)
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

  # def train2(model=str, dataset_path=None, epochs=10, imgsz=640):
  #    data_yaml_files = glob.glob(os.path.join(dataset_path, '*.yaml'))
  #    if not data_yaml_files:
  #      raise FileNotFoundError(f"No YAML files found in {dataset_path}")
  #    yaml_file_path = data_yaml_files[0]
  #    print(f"Using YAML file: {yaml_file_path}")
#     ##helper function to build and load --hyp yaml
#     results = model.train(
#       data=yaml_file_path, # Path to dataset config file
#       epochs=epochs, # Number of training epochs
#       imgsz=imgsz), #Image size for training
#       # Also takes a device argument if needed, e.g. device="cpu"
#     log.write(f"Results: {results}")
#     return results
def train(model=str, dataset_path=None, epochs=10, imgsz=640, hyp=None):
    data_yaml_files = glob.glob(os.path.join(dataset_path, '*.yaml'))
    if not data_yaml_files:
      raise FileNotFoundError(f"No YAML files found in {dataset_path}")
    yaml_file_path = data_yaml_files[0]
    print(hyp)
    print(f"Using YAML file: {yaml_file_path}")
    #custom_loss = CustomDetectionLoss(model.model)
    #model.model.criterion = custom_loss
    print("\nVerifying loss function:")
    print(f"Current criterion type: {type(model.model.criterion).__name__}")
    #assert isinstance(model.model.criterion, CustomDetectionLoss), "Custom loss not properly set!"
    print("Custom loss function successfully integrated!")
    # Prepare training arguments
    train_args = {

        'data': yaml_file_path,  # Path to dataset config file

        'epochs': epochs,  # Number of training epochs

        'imgsz': imgsz,  # Image size for training        

    }    

    # Add hyp file if provided
    if hyp:
        custom_hyp= yaml_load(hyp)

        print(f"Using hyperparameter file: {hyp}")
        train_args.update(custom_hyp)

    # Train the model with all arguments
    results = model.train(**train_args)
    log.write(f"Results: {results}")
    return results

    
## TODO: refine test function
# def val(model=str, dataset_path=None, epochs=10, imgsz=640,batch=16,conf=0.25,iou=0.6):
#     data_yaml_files = glob.glob(os.path.join(dataset_path, '*.yaml'))
#       if not data_yaml_files:
#         raise FileNotFoundError(f"No YAML files found in {dataset_path}")
#       yaml_file_path = data_yaml_files[0]
#       print(f"Using YAML file: {yaml_file_path}")
#     ##helper function to build and load --hyp yaml
#      metrics = model.val(
#        data=yaml_file_path, # Path to dataset config file
#        epochs=epochs, # Number of training epochs
#        imgsz=imgsz,
#        batch=batch,
#        conf=conf,#Sets the minimum confidence threshold for detections. Lower values increase recall but may introduce more false positives. Used during validation to compute precision-recall curves.
#        iou=batch)#Sets the Intersection Over Union threshold for Non-Maximum Suppression. Controls duplicate detection elimination. 
# #       # Also takes a device argument if needed, e.g. device="cpu"
#     log.write(f"Results: {metrics}")
#      return metrics 

# def predict(model=str, dataset_path=None,imgsz=320,conf=0.5,save=True):
#     data_yaml_files = glob.glob(os.path.join(dataset_path, '*.yaml'))
#       if not data_yaml_files:
#         raise FileNotFoundError(f"No YAML files found in {dataset_path}")
#       yaml_file_path = data_yaml_files[0]
#       print(f"Using YAML file: {yaml_file_path}")
#     ##helper function to build and load --hyp yaml
#      predict = model.predict(
#        data=yaml_file_path, # Path to dataset config file
#         # Number of training epochs
#        imgsz=imgsz,
#        conf=conf,
#      save=save)
#      # Also takes a device argument if needed, e.g. device="cpu"
#     log.write(f"Results: {predict}")
#      return predict 

def main():
    # Main function to handle training or testing
    if args.model_mode == "train":
        train(model=model, dataset_path=args.dataset_location, epochs=args.epochs, imgsz=args.imgsz,hyp=args.hyp)
    elif args.model_mode == "test":
        test(args.model_name, args.dataset_location)
    else:
        raise ValueError("Model mode must be either train or test.")

if __name__ == "__main__":
    main()
