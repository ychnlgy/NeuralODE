import torch

class RandomChance(torch.nn.Module):

    def __init__(self, output_size):
        super(RandomChance, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1))
        self.output_size = output_size

    def forward(self, X, *args, **kwargs):
        return torch.autograd.Variable(
            torch.rand(len(X), self.output_size),
            requires_grad = True
        )
