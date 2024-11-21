import os
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, normalization='layernorm', output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

#########################################################################################################################################

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
        transforms = [TanhTransform()]
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
    def __init__(self, obs_dim, action_dim,  act_max, act_min, hidden_dim, hidden_depth):
        super().__init__()

        self.trunk = mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth, normalization='batchnorm')

        self.outputs = dict()
        self.apply(weight_init)

        self.action_max = act_max
        self.action_min = act_min

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = -5, 2
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

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