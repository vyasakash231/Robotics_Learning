import numpy as np
import os
import math
from collections import OrderedDict
from numbers import Number

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal as TorchNormal
from torch.distributions import Independent as TorchIndependent
from torch.distributions import Distribution as TorchDistribution

################################################################################################################
def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

################################################################################################################
class Distribution(TorchDistribution):
    def sample_and_logprob(self):
        s = self.sample()
        log_p = self.log_prob(s)
        return s, log_p

    def rsample_and_logprob(self):
        s = self.rsample()
        log_p = self.log_prob(s)
        return s, log_p

    def mle_estimate(self):
        return self.mean

    def get_diagnostics(self):
        return {}

################################################################################################################
class TorchDistributionWrapper(Distribution):
    def __init__(self, distribution: TorchDistribution):
        self.distribution = distribution

    @property
    def batch_shape(self):
        return self.distribution.batch_shape

    @property
    def event_shape(self):
        return self.distribution.event_shape

    @property
    def arg_constraints(self):
        return self.distribution.arg_constraints

    @property
    def support(self):
        return self.distribution.support

    @property
    def mean(self):
        return self.distribution.mean

    @property
    def variance(self):
        return self.distribution.variance

    @property
    def stddev(self):
        return self.distribution.stddev

    def sample(self, sample_size=torch.Size()):
        return self.distribution.sample(sample_shape=sample_size)

    def rsample(self, sample_size=torch.Size()):
        return self.distribution.rsample(sample_shape=sample_size)

    def log_prob(self, value):
        return self.distribution.log_prob(value)

    def cdf(self, value):
        return self.distribution.cdf(value)

    def icdf(self, value):
        return self.distribution.icdf(value)

    def enumerate_support(self, expand=True):
        return self.distribution.enumerate_support(expand=expand)

    def entropy(self):
        return self.distribution.entropy()

    def perplexity(self):
        return self.distribution.perplexity()

    def __repr__(self):
        return 'Wrapped ' + self.distribution.__repr__()
    
################################################################################################################
class Independent(Distribution, TorchIndependent):
    def get_diagnostics(self):
        return self.base_dist.get_diagnostics()
    
################################################################################################################
def create_stats_ordered_dict(name, data, stat_prefix=None, always_show_all_stats=True, exclude_max_min=False):
    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict("{0}_{1}".format(name, number),d)
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([(name + ' Mean', np.mean(data)),(name + ' Std', np.std(data))])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats

#################################################################################################################
class MultivariateDiagonalNormal(TorchDistributionWrapper):
    from torch.distributions import constraints
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}

    def __init__(self, loc, scale_diag, reinterpreted_batch_ndims=1):
        dist = Independent(TorchNormal(loc, scale_diag),reinterpreted_batch_ndims=reinterpreted_batch_ndims)
        super().__init__(dist)

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(create_stats_ordered_dict('mean',get_numpy(self.mean),))
        stats.update(create_stats_ordered_dict('std',get_numpy(self.distribution.stddev)))
        return stats

    def __repr__(self):
        return self.distribution.base_dist.__repr__()


## POLICY NETWORK  ##############################################################################################
LOG_STD_MAX = 2
LOG_STD_MIN = -20
class Actor(nn.Module):
    def __init__(self, obs_dims, n_actions, action_max, action_min):
        super(Actor, self).__init__()
        self.state_dim = obs_dims
        self.action_dim = n_actions

        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc_mean = nn.Linear(128, self.action_dim)
        self.fc_logstd = nn.Linear(128, self.action_dim)
        
        init_w = 1e-4
        torch.nn.init.uniform_(self.fc1.weight, -init_w, init_w)
        torch.nn.init.uniform_(self.fc2.weight, -init_w, init_w)
        torch.nn.init.uniform_(self.fc1.bias, -init_w, init_w)
        torch.nn.init.uniform_(self.fc2.bias, -init_w, init_w)

        init_w = 1e-3
        torch.nn.init.uniform_(self.fc_mean.weight, -init_w, init_w)
        torch.nn.init.uniform_(self.fc_logstd.weight, -init_w, init_w)
        torch.nn.init.uniform_(self.fc_mean.bias, -init_w, init_w)
        torch.nn.init.uniform_(self.fc_logstd.bias, -init_w, init_w)

        # action rescaling
        self.action_scale = torch.tensor((action_max - action_min) / 2.0, dtype=torch.float32).to('cuda')
        self.action_bias = torch.tensor((action_max + action_min) / 2.0, dtype=torch.float32).to('cuda')

    def forward(self, state):        
        x = F.relu(self.fc1(state))  
        x = F.relu(self.fc2(x))  
        
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action(self, state, deterministic=False):
        # we ouput log_std instead of std because SGD is not good at constrained optimisation
        # we want std to be positive i.e. >0. So, what we do is ouput log_std and then take exponential
        # so NN output is negative also i.e. log_std can be negative but std will always be positive
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean) * self.action_scale + self.action_bias  # Only used for evaluating policy at test time.
            log_prob = None
            return action
        else:
            normal = TorchNormal(mean, std)  # MultivariateDiagonalNormal(mean, std) 
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            x_t.requires_grad_()
            y_t = torch.tanh(x_t)  # In case of stochastic a = tanh(x), x~N(mean, std)

            if y_t is None:
                y_t = torch.clamp(y_t, -0.999999, 0.999999)
                x_t = (1/2) * (torch.log(1 + y_t) / torch.log(1 - y_t))

            action = y_t * self.action_scale + self.action_bias

            # Enforcing Action Bound
            log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-8) #.sum(dim=1)
            log_prob = log_prob.sum(1, keepdim=True) 
            return action, log_prob, mean
    
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
        self.load_state_dict(torch.load(os.path.join(os.getcwd(),folder_name+f"/policy_{global_step}")))