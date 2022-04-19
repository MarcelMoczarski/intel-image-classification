import torch.nn.functional as F


def cross_entropy(x, y):
    return F.cross_entropy(x, y)
    