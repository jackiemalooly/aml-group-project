import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Train military aircraft detection models.')
    parser.add_argument('--model_mode', type=str, default='train', help='Mode of the model: train or test')
    parser.add_argument('--model_name', type=str, default='yolo11n.pt', help='Name of the model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--hyp', type=str, required=True,default = None , help='Path to the hyperpameters')
    parser.add_argument('--freeze', type=list, default=None, help='First N layer to freeze for training. Or, itemized list of layers to freeze.')
    return parser