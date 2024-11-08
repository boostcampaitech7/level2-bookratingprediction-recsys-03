import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss as MAELoss
from torch.nn import *
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6
    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss