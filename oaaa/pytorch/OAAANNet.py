import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class OAAA(nn.Module):
    def __init__(self, game, args):
        # game params
        self.input_size = 878  # Size of your 1D input
        self.action_size = game.getActionSize()
        self.args = args
        
        super(OAAA, self).__init__()
        
        # Define network layers
        self.fc1 = nn.Linear(self.input_size, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 128)
        self.fc_bn3 = nn.BatchNorm1d(128)
        
        # Policy head - outputs probability distribution over moves
        self.fc_policy = nn.Linear(128, self.action_size)
        
        # Value head - outputs state value estimation
        self.fc_value = nn.Linear(128, 1)

    def forward(self, s):
        # Input shape: (N, 878, 1) -> reshape to (N, 878)
        s = s.view(-1, self.input_size)
        
        # Apply layers with dropout
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn3(self.fc3(s))), p=self.args.dropout, training=self.training)
        
        # Policy head
        pi = self.fc_policy(s)                                                                   # batch_size x action_size
        pi = F.log_softmax(pi, dim=1)
        
        # Value head
        v = self.fc_value(s)                                                                     # batch_size x 1
        v = torch.tanh(v)
        
        return pi, v
