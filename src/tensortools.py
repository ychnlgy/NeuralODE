import torch

def flip(t, axis):
    c = t.size(axis)
    i = torch.arange(-1, -c-1, -1).long().to(t.device)
    return t.transpose(0, axis)[i].transpose(0, axis)
