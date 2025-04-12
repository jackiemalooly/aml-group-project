import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Train military aircraft detection models.')
    parser.add_argument('--model_mode', type=str, default='train', help='Mode of the model: train or test')
    parser.add_argument('--model_name', type=str, default='yolo11n.pt', help='Name of the model')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset')
    # hyperparameters, defaults set to YOLOv5 baseline values
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    ##TODO: Add these arguments to main.py and ssh file.
    parser.add_argument('--seed', type=int, default=42, help='Sets the random seed for training, ensuring reproducibility.')
    parser.add_argument('--lr0', type=float, default=0.0002, help='Initial speed for weight updates in the model')
    parser.add_argument('--lrf', type=float, default=0.001, help='Learning rate used in the final stages of training')
    parser.add_argument('--momentum', default=0.937, type=float, help='Cumulative effect of previous gradient udpates.')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='Prevents model overfitting using L2 regularization to penalize large weights.')
    parser.add_argument('--box', type=float, default=0.05, help='Weight of the box loss componenet in the loss function.')
    parser.add_argument('--cls', type=float, default=0.3, help='Weight of the class loss in the total loss function.')
    parser.add_argument('--optimizer', type=str, default='auto', help='Choice of optimizer for training. SGD, Adam, AdamW, NAdam, RAdam, RMSProp available.')
    parser.add_argument('--close_mosaic', type=int, default=10, help='Disables mosaic data augmentation in the last N epochs to stabilize training before completion. Setting to 0 disables this feature.')
    parser.add_argument('--freeze', type=int|list, default=None, help='Can also accept a list. Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters.')
    return parser