from ultralytics import YOLO
from ultralytics.utils import yaml_load
import numpy as np
import os
import glob
import comet_ml

from utils import (
    AverageMeter, 
    Logger,
    set_seed,
)
set_seed(42)  # Set seed for reproducibility

from args import argument_parser
args = argument_parser().parse_args()
print(args)
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
# Initialize logger
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
log = Logger()

# Load model for training only
if args.model_mode == "train":
    model = YOLO(args.model_name, 'detect')  # e.g. 'yolov8n.pt'

<<<<<<< Updated upstream
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
=======
# --------------------- TRAIN ---------------------
def train(model, dataset_path=None, epochs=10, imgsz=640, hyp=None):
    data_yaml_files = glob.glob(os.path.join(dataset_path, '*.yaml'))
    if not data_yaml_files:
        raise FileNotFoundError(f"No YAML files found in {dataset_path}")
    yaml_file_path = data_yaml_files[0]

    print(hyp)
    print(f"Using YAML file: {yaml_file_path}")

    train_args = {
        'data': yaml_file_path,
        'epochs': epochs,
        'imgsz': imgsz
    }

    if hyp:
        custom_hyp = yaml_load(hyp)
        print(f"Using hyperparameter file: {hyp}")
        train_args.update(custom_hyp)

>>>>>>> Stashed changes
    results = model.train(**train_args)
    log.write(f"Results: {results}")
    return results

<<<<<<< Updated upstream
def val(model=str, dataset_path=None, epochs=10, imgsz=640,batch=16,conf=0.25,iou=0.6):
    data_yaml_files = glob.glob(os.path.join(dataset_path, '*.yaml'))
      if not data_yaml_files:
        raise FileNotFoundError(f"No YAML files found in {dataset_path}")
      yaml_file_path = data_yaml_files[0]
      print(f"Using YAML file: {yaml_file_path}")
    ##helper function to build and load --hyp yaml
     metrics = model.val(
       data=yaml_file_path, # Path to dataset config file
       epochs=epochs, # Number of training epochs
       imgsz=imgsz,
       batch=batch,
       conf=conf,#Sets the minimum confidence threshold for detections. Lower values increase recall but may introduce more false positives. Used during validation to compute precision-recall curves.
       iou=batch)#Sets the Intersection Over Union threshold for Non-Maximum Suppression. Controls duplicate detection elimination. 
#       # Also takes a device argument if needed, e.g. device="cpu"
    log.write(f"Results: {results}")
     return metrics
    
## TODO: refine test function
def val(model=str, dataset_path=None, epochs=10, imgsz=640,batch=16,conf=0.25,iou=0.6):
    data_yaml_files = glob.glob(os.path.join(dataset_path, '*.yaml'))
      if not data_yaml_files:
        raise FileNotFoundError(f"No YAML files found in {dataset_path}")
      yaml_file_path = data_yaml_files[0]
      print(f"Using YAML file: {yaml_file_path}")
    ##helper function to build and load --hyp yaml
     metrics = model.val(
       data=yaml_file_path, # Path to dataset config file
       epochs=epochs, # Number of training epochs
       imgsz=imgsz,
       batch=batch,
       conf=conf,#Sets the minimum confidence threshold for detections. Lower values increase recall but may introduce more false positives. Used during validation to compute precision-recall curves.
       iou=batch)#Sets the Intersection Over Union threshold for Non-Maximum Suppression. Controls duplicate detection elimination. 
#       # Also takes a device argument if needed, e.g. device="cpu"
    log.write(f"Results: {metrics}")
     return metrics 

def predict(model=str, dataset_path=None,imgsz=320,conf=0.5,save=True):
    data_yaml_files = glob.glob(os.path.join(dataset_path, '*.yaml'))
      if not data_yaml_files:
        raise FileNotFoundError(f"No YAML files found in {dataset_path}")
      yaml_file_path = data_yaml_files[0]
      print(f"Using YAML file: {yaml_file_path}")
    ##helper function to build and load --hyp yaml
     predict = model.predict(
       data=yaml_file_path, # Path to dataset config file
        # Number of training epochs
       imgsz=imgsz,
       conf=conf,
     save=save)
     # Also takes a device argument if needed, e.g. device="cpu"
    log.write(f"Results: {predict}")
     return predict 
=======
# --------------------- VAL ---------------------
def val(model_path, dataset_path=None, epochs=10, imgsz=640,batch=16,conf=0.25, iou=0.6):
    model = YOLO(model_path)
    
    data_yaml_files = glob.glob(os.path.join(dataset_path, '*.yaml'))
    if not data_yaml_files:
        raise FileNotFoundError(f"No YAML files found in {dataset_path}")
    yaml_file_path = data_yaml_files[0]
    print(f"Using YAML file: {yaml_file_path}")

    metrics = model.val(
        data=yaml_file_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou
    )

    log.write(f"Results: {metrics}")
    return metrics

# --------------------- PREDICT ---------------------
def predict(model_path, dataset_path=None, imgsz=320, conf=0.5, save=True):
    model = YOLO(model_path)

    # Ensure test image directory or file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Test data path not found: {dataset_path}")
    print(f"Running prediction on: {dataset_path}")

    results = model.predict(
        source=dataset_path,  # 
        imgsz=imgsz,
        conf=conf,
        save=save
    )

    log.write(f"Results: {results}")
    return results

>>>>>>> Stashed changes

# --------------------- MAIN ---------------------
def main():
    if args.model_mode == "train":
<<<<<<< Updated upstream
        train(model=model, dataset_path=args.dataset_location, epochs=args.epochs, imgsz=args.imgsz,hyp=args.hyp)
=======
        train(
            model=model,
            dataset_path=args.dataset_location,
            epochs=args.epochs,
            imgsz=args.imgsz,
            hyp=args.hyp
        )
    elif args.model_mode == "val":
        val(
            model_path=args.model_name,
            dataset_path=args.dataset_location,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou
        )
>>>>>>> Stashed changes
    elif args.model_mode == "test":
        predict(
            model_path=args.model_name,
            dataset_path=args.dataset_location,
            imgsz=args.imgsz,
            conf=args.conf,
            save=True
        )
    else:
        raise ValueError("Model mode must be either train, val, or test.")

if __name__ == "__main__":
    main()
