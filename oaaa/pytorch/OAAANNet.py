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
        # print(f"Input shape: {s.shape}")
        # batch_size = s.size(0)
        # s = s.view(batch_size, -1)
        s = s.view(-1, 1, 878, 1)                # batch_size x 1 x board_x x board_y
        # print(f"Reshaped shape: {s.shape}")
        s = s.view(-1, 878)
        # print(f"Reshaped shape 2: {s.shape}")

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        # print(f"After fc1: {s.shape}")
        
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)
        # print(f"After fc2: {s.shape}")
        
        s = F.dropout(F.relu(self.fc_bn3(self.fc3(s))), p=self.args.dropout, training=self.training)
        # print(f"After fc3: {s.shape}")
        
        pi = F.log_softmax(self.fc_policy(s), dim=1)
        v = torch.tanh(self.fc_value(s))
        return pi, v

        v = torch.tanh(self.fc_value(s))
        return pi, v
