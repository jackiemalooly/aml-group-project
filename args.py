import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Train military aircraft detection models.')
    parser.add_argument('--model_mode', type=str, default='train', help='Mode of the model: train or test')
    parser.add_argument('--model_name', type=str, default='yolo11n.pt', help='Name of the model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--model_task', type=str, default='detect', help='YOLO task specification, i.e. detect, segment, classify, pose, obb.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset')
<<<<<<< Updated upstream
    parser.add_argument('--hyp', type=str, required=True,default = None , help='Path to the hyperpameters')
    parser.add_argument('--iou', type=float,default = 0.6 , help='Iou of validation')
    parser.add_argument('--conf', type=float,default = 0.25 , help='confidence threshold')
    parser.add_argument('--save', type=str,default = True , help='save images')
    
    
=======
    parser.add_argument('--hyp', type=str, required=False,default = None , help='Path to the hyperpameters')
   # parser.add_argument('--iou', type=float,default = 0.6 , help='Iou of validation')
   # parser.add_argument('--conf', type=float,default = 0.25 , help='confidence threshold')
   # parser.add_argument('--save', type=str,required=False,default = True , help='save images')
   # parser.add_argument('--batch', type=int,required=False, default=16, help='Number of batch')
>>>>>>> Stashed changes
    return parser
    # # hyperparameters, defaults set to baseline values
     # parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    # parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    # ##TODO: Add these arguments to main.py and ssh file.
    # parser.add_argument('--seed', type=int, default=42, help='Sets the random seed for training, ensuring reproducibility.')
    # parser.add_argument('--optimizer', type=str, default='auto', help='Choice of optimizer for training. SGD, Adam, AdamW, NAdam, RAdam, RMSProp available.')
    # parser.add_argument('--close_mosaic', type=int, default=10, help='Disables mosaic data augmentation in the last N epochs to stabilize training before completion. Setting to 0 disables this feature.')
    # parser.add_argument('--freeze', type=int|list, default=None, help='Can also accept a list. Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters.')
    # --hyp yaml file args
    # parser.add_argument('--lr0', type=float, default=0.0002, help='Initial speed for learning rate (SGD=1E-2, Adam=1E-3)')
    # parser.add_argument('--lrf', type=float, default=0.001, help='Learning rate used in the final stages of training. Final OneCycleLR learning rate (lr0 * lrf)')
    # parser.add_argument('--momentum', type=float, default=0.937, help='Cumulative effect of previous gradient udpates. SGD momentum/Adam beta1')
    # parser.add_argument('--weight_decay', type=float, default=0.0005, help='Prevents model overfitting using L2 regularization to penalize large weights.')
    # parser.add_argument('--warmup_epochs', type=float, default=0.0, help='Warmup epochs (fractions ok)')
    # parser.add_argument('--warmup_momentum', type=float, default=0, help='Warmup initial momentum.')
    # parser.add_argument('--warmup_bias_lr', type=float, help='Warmup initial bias lr.')
    # parser.add_argument('--box', type=float, default=0.05, help='Weight of the box loss componenet in the loss function.')
    # parser.add_argument('--cls', type=float, default=0.3, help='Weight of the class loss in the total loss function.')
    # parser.add_argument('--cls_pw', type=float, default=1.0, help='cls BCELoss positive_weight')
    # parser.add_argument('--obj', type=float, default=0.7, help='obj loss gain (scale with pixels)')
    # parser.add_argument('--obj_pw', type=float, default=1.0, help='obj BCELoss positive_weight')
    # parser.add_argument('--iou_t', type=float, default=0.20, help='IoU training threshold')
    # parser.add_argument('--anchor_t', type=float, default=4.0, help='Anchor-multiple box alignment threshold')
    # ## augmentation args for --hyp yaml file
    # parser.add_argument('--fl_gamma', type=float, default=0.0, help='focal loss gamma (efficientDet default gamma=1.5)')
    # parser.add_argument('--hsv_h', type=float, default=0.0, help='image HSV-Hue augmentation (fraction)')
    # parser.add_argument('--hsv_s', type=float, default=0.0, help='image HSV-Saturation augmentation (fraction)')
    # parser.add_argument('--hsv_v', type=float, default=0.0, help='image HSV-Value augmentation (fraction)')
    # parser.add_argument('--degrees', type=float, default=0.0, help='image rotation (+/- deg)')
    # parser.add_argument('--scale', type=float, default=0.0, help='image scale (+/- gain)')
    # parser.add_argument('--shear', type=float, default=0.0, help='image shear (+/- deg)')
    # parser.add_argument('--perspective', type=float, default=0.0, help='image perspective (+/- fraction), range 0-0.001')
    # parser.add_argument('--flipud', type=float, default=0.0, help='image flip up-down (probability)')
    # parser.add_argument('--fliplr', type=float, default=0.0, help='image flip left-right (probability)')
    # parser.add_argument('--mosaic', type=float, default=0.0, help='image mosaic (probability)')
    # parser.add_argument('--mixup', type=float, default=0.0, help='image mixup (probability)')
    # parser.add_argument('--copy_paste', type=float, default=0.0, help='segment copy-paste (probability)')
    # parser.add_argument('--copy_paste', type=float, default=0.0, help='segment copy-paste (probability)')
  