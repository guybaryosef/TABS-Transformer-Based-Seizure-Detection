import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import numpy as np

WINDOW_SIZE=300

class EnsembleModel(nn.Module):
        def __init__(self):
                super(EnsembleModel, self).__init__()
                
        def forward(self, x):
            x = torch.zeros(x.shape[0], 2)
            x[:,1] = 1
            return x        
##########################################################################
##########################################################################
##########################################################################
