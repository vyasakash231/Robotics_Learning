import torch
import torch.nn as nn
import torch.nn.functional as F

## Q-NETWORK
class SoftQNetwork(nn.Module):
    def __init__(self, obs_dims, n_actions):
        super(SoftQNetwork, self).__init__()
        self.state_dim = obs_dims
        self.action_dim = n_actions

        self.f1 = nn.Linear(self.state_dim + self.action_dim, 256)
        self.f2 = nn.Linear(256, 128)
        self.f3 = nn.Linear(128, 1)

        init_w = 1e-4
        torch.nn.init.uniform_(self.f1.weight, -init_w, init_w)
        torch.nn.init.uniform_(self.f2.weight, -init_w, init_w)
        torch.nn.init.uniform_(self.f3.weight, -init_w, init_w)

        torch.nn.init.uniform_(self.f1.bias, -init_w, init_w)
        torch.nn.init.uniform_(self.f2.bias, -init_w, init_w)
        torch.nn.init.uniform_(self.f3.bias, -init_w, init_w)

    def forward(self, state, action): 
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.f1(x)) 
        x = F.relu(self.f2(x)) 
        x = self.f3(x) 
        return x