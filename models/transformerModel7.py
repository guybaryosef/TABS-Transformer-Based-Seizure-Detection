import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import numpy as np

WINDOW_SIZE=300

class CnnModel(nn.Module):
        def __init__(self, input_size, output_size):
                super(CnnModel, self).__init__()
                self.cnn1 = nn.Conv1d(20, output_size, 5, padding=12)
                self.bn1  = nn.BatchNorm1d(output_size)
                self.dropout = nn.Dropout(p=0.7)
                self.relu = nn.ReLU()

        def forward(self, x):
                x = self.dropout(self.relu(self.bn1(self.cnn1(x))))
                return x        
##########################################################################
class TransformerModel(nn.Module):
        def __init__(self, hidden_size=20, num_heads=16):
                super(TransformerModel, self).__init__()
                self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
                self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        def forward(self, x):
                x = x.transpose(1,0)
                x = self.transformer_encoder(x)
                x = x.transpose(0,1)
                return x        
##########################################################################
class FullyConnectedModel(nn.Module):
        def __init__(self, input_shape):
                super(FullyConnectedModel, self).__init__()
                self.linear = nn.Linear(input_shape, 128)
                self.bn1 = nn.BatchNorm1d(128)
                self.dropout = nn.Dropout(p=0.7)
                self.linear2 = nn.Linear(128, 2)
                self.relu = nn.ReLU()

        def forward(self, x):
                x = self.relu(self.dropout(self.bn1(self.linear(x).squeeze(1))))
                x = self.linear2(x)
                return x 
##########################################################################
class EnsembleModel(nn.Module):
        def __init__(self):
                super(EnsembleModel, self).__init__()
                self.cnn  = CnnModel(WINDOW_SIZE, 50)
                self.txmr = TransformerModel(320)
                self.fc   = FullyConnectedModel(340)

                self.cnn2 = nn.Conv1d(50, 1, 1, dilation=2, padding=10)
                self.bn1 = nn.BatchNorm1d(1)
                self.dropout = nn.Dropout(p=0.7)
                self.relu = nn.ReLU()

        def forward(self, x):
                x = self.cnn(x)
                x = self.txmr(x)
                x = self.dropout(self.relu(self.bn1(self.cnn2(x))))
                x = self.fc(x)
                return x        
##########################################################################
##########################################################################
##########################################################################
