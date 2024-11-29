import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        obs_action = torch.cat([x, a], 1)
        x = F.relu(self.fc1(obs_action))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x