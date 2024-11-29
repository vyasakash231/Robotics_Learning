import os
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions


class TanhTransform(distributions.transforms.Transform):
    domain = distributions.constraints.real
    codomain = distributions.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(distributions.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = distributions.Normal(loc, scale)
        transforms = TanhTransform()
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

##########################################################################################################################################

class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_dim, action_dim,  act_max, act_min):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

        self.action_max = act_max
        self.action_min = act_min

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mu = self.fc_mean(x)
        log_std = self.fc_logstd(x)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = -5, 2
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist
    
    def get_action(self, obs, differentiable=True):
        distribution = self.forward(obs)

        if differentiable:
            action = distribution.rsample() # rsample(), generate samples from the distribution and also supports differentiating through the sampler
            log_pi = distribution.log_prob(action).sum(-1, keepdim=True)
            return action, log_pi
        else:
            action = distribution.sample() # sample(), generate samples from the distribution and does not supports differentiating through the sampler
            action = action.clamp(self.action_min, self.action_max)
            return action
        
    def test_action(self, obs):
        distribution = self.forward(obs)
        action = distribution.mean.clamp(self.action_min, self.action_max)
        return action

    # Save model parameters
    def save_model(self,folder_name, global_step):
        folder_path = os.path.join(os.getcwd(), folder_name)  # Construct the full path to the folder
        
        # Check if the folder exists, If the folder does not exist, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_name}' created at {folder_path}")

        torch.save(self.state_dict(), os.path.join(os.getcwd(),folder_name+f"/policy_{global_step}"))
    
    # Load model parameters
    def load_model(self,folder_name, global_step):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(os.path.join(os.getcwd(),folder_name+f"/policy_{global_step}"), map_location=torch.device(device)))



