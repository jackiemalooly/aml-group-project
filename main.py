import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm.notebook import tqdm
import numpy as np
import os

from utils import (
AverageMeter, 
Logger,
set_seed,
)

# set seed
set_seed(42) 

# Initialize logger
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
log = Logger()
# log.open("logs/%s_log_train.txt")

# load the STL10 dataset
image_transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Once we have the transformations defined, lets define the train and test sets
train_dataset = torchvision.datasets.STL10('dataset/',
                                           split='train',
                                           download=True,
                                           transform=image_transform)
test_dataset = torchvision.datasets.STL10('dataset/',
                                          split='test',
                                          download=True,
                                          transform=image_transform)

batch_size_train = 256 # We use smaller batch size here for training
batch_size_test = 1024 # We use bigger batch size for testing

# Once we have the datasets defined, lets define the data loaders as follows
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size_train,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size_test,
                                          shuffle=True)

# Define the model
class AlexNet1(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet1, self).__init__()
        # input channel 3, output channel 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        # relu non-linearity
        self.relu1 = nn.ReLU()
        # max pooling
        self.max_pool2d1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # input channel 64, output channel 192
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.max_pool2d2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # input channel 192, output channel 384
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        # input channel 384, output channel 256
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        # input channel 256, output channel 256
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.max_pool2d5 = nn.MaxPool2d(kernel_size=3, stride=2)
        # adaptive pooling
        self.adapt_pool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        #dropout layer
        self.dropout1 = nn.Dropout()
        # linear layer
        self.linear1 = nn.Linear(in_features=9216, out_features=4096, bias=True)
        self.relu6 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        self.linear2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool2d1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2d2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.max_pool2d5(x)
        x = self.adapt_pool(x)
        # Note how we are flattening the feature map, B x C x H x W -> B x C*H*W
        x = x.reshape(x.shape[0], -1)
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.relu6(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.relu7(x)
        x = self.linear3(x)
        return x

# Initialize the model
model = AlexNet1(10) # since STL-10 dataset has 10 classes, we set num_classes = 10
# device: cuda (gpu) or cpu
device = "cuda"
# map to device
model = model.to(device) # `model.cuda()` will also do the same job
# make the parameters trainable
for param in model.parameters():
    param.requires_grad = True

# Define Optimizer
learning_rate = 0.0001
weight_decay = 0.0005
# define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

##define train function
def train(model, device, train_loader, optimizer):
    # meter
    loss = AverageMeter()
    # switch to train mode
    model.train()
    tk0 = tqdm(train_loader, total=int(len(train_loader)))
    for batch_idx, (data, target) in enumerate(tk0):
        # after fetching the data transfer the model to the
        # required device, in this example the device is gpu
        # transfer to gpu can also be done by
        # data, target = data.cuda(), target.cuda()
        data, target = data.to(device), target.to(device)
        # compute the forward pass
        # it can also be achieved by model.forward(data)
        output = model(data)
        # compute the loss function
        loss_this = F.cross_entropy(output, target)
        # initialize the optimizer
        optimizer.zero_grad()
        # compute the backward pass
        loss_this.backward()
        # update the parameters
        optimizer.step()
        # update the loss meter
        loss.update(loss_this.item(), target.shape[0])
    log('Train: Average loss: {:.4f}\n'.format(loss.avg))
    return loss.avg

##define test function
def test(model, device, test_loader):
    # meters
    loss = AverageMeter()
    acc = AverageMeter()
    correct = 0
    # switch to test mode
    model.eval()
    for data, target in test_loader:
        # after fetching the data transfer the model to the
        # required device, in this example the device is gpu
        # transfer to gpu can also be done by
        # data, target = data.cuda(), target.cuda()
        data, target = data.to(device), target.to(device)  # data, target = data.cuda(), target.cuda()
        # since we dont need to backpropagate loss in testing,
        # we dont keep the gradient
        with torch.no_grad():
            # compute the forward pass
            # it can also be achieved by model.forward(data)
            output = model(data)
        # compute the loss function just for checking
        loss_this = F.cross_entropy(output, target) # sum up batch loss
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        # check which of the predictions are correct
        correct_this = pred.eq(target.view_as(pred)).sum().item()
        # accumulate the correct ones
        correct += correct_this
        # compute accuracy
        acc_this = correct_this/target.shape[0]*100.0
        # update the loss and accuracy meter
        acc.update(acc_this, target.shape[0])
        loss.update(loss_this.item(), target.shape[0])
    log('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss.avg, correct, len(test_loader.dataset), acc.avg))
    
## define training loop
def main():
    # create TensorBoard logger
    writer = SummaryWriter('runs/mnist_experiment_1')
    # number of epochs we decide to train
    num_epoch = 10
    for epoch in range(1, num_epoch + 1):
        epoch_loss = train(model, device, train_loader, optimizer)
        writer.add_scalar('training_loss', epoch_loss, global_step=epoch)
    test(model, device, test_loader)
    log({summary(model, (1, 28, 28))})

if __name__ == "__main__":
    main()