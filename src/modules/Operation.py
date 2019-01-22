import torch

class Operation(torch.nn.Module):

    def __init__(self, f):
        super(Operation, self).__init__()
        self.f = f
    
    def forward(self, X):
        return self.f(X)
