import torch

import modules

IDENTITY = torch.nn.Sequential()
LEAKY_RELU = torch.nn.LeakyReLU()

class ResBlock(torch.nn.Module, modules.NormalInit):

    def __init__(self, conv, shortcut=IDENTITY, activation=LEAKY_RELU, output=True):
        super(ResBlock, self).__init__()
        self.cn = conv
        self.sc = shortcut
        self.ac = activation
        self.op = output
        
        self.init_weights(self.cn)
        self.init_weights(self.sc)
    
    def get_init_targets(self):
        return [torch.nn.Conv2d]
        
    def forward(self, X):
        return self.ac(self.forward_cnn(X) + self.sc(X)), self.op
    
    def forward_cnn(self, X):
        return self.cn(X)
