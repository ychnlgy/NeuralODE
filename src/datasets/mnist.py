import torch
import torchvision.datasets

import constants

from . import util

def get(download=0):
    
    download = int(download)
    
    CLASSES = 10
    CHANNELS = 1
    IMAGESIZE = (32, 32)
    
    train = torchvision.datasets.MNIST(root=constants.DATA_ROOT, train=True, download=download)
    train_X = train.train_data.view(-1, 1, 28, 28).float()/255.0
    train_X = util.convert_size(train_X, IMAGESIZE)
    train_Y = torch.LongTensor(train.train_labels)
    
    test = torchvision.datasets.MNIST(root=constants.DATA_ROOT, train=False, download=download)
    test_X = test.test_data.view(-1, 1, 28, 28).float()/255.0
    test_X = util.convert_size(test_X, IMAGESIZE)
    test_Y = torch.LongTensor(test.test_labels)

    return train_X, train_Y, test_X, test_Y, CLASSES, CHANNELS, IMAGESIZE


