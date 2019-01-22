import torch

class Upsample(torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(Upsample, self).__init__()
        self.args = args
        self.kwargs = kwargs
    
    def forward(self, X):
        return torch.nn.functional.interpolate(X, *self.args, **self.kwargs)
