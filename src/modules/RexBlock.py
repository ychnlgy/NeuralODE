import torch

from .ResBlock import ResBlock

IDENTITY = torch.nn.Sequential()
LEAKY_RELU = torch.nn.LeakyReLU()

class RexBlock(ResBlock):

    def __init__(self, *args, **kwargs):
        super(RexBlock, self).__init__(*args, **kwargs)
        self.cn = torch.nn.ModuleList(self.cn)
        
        for cn in self.cn:
            self.init_weights(cn)
        
    def forward_cnn(self, X):
        out = []
        for column in self.cn:
            out.append(column(X))
        return sum(out)
