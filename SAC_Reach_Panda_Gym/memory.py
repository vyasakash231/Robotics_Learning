import numpy as np
import csv
import os
import math
import random
import collections


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## MEMORY 
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.memory_count = 0

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.memory_count += 1

    def sample_buffer(self, batch_size, device):
        mini_batch = random.sample(self.buffer, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        obs = torch.from_numpy(np.array(s_lst)).type('torch.FloatTensor').to(device)
        actions = torch.from_numpy(np.array(a_lst)).type('torch.FloatTensor').to(device)
        rewards = torch.from_numpy(np.array(r_lst)).type('torch.FloatTensor').to(device)
        next_obs = torch.from_numpy(np.array(s_prime_lst)).type('torch.FloatTensor').to(device)
        terminals = torch.from_numpy(np.array(done_mask_lst)).type('torch.FloatTensor').to(device)

        return obs, actions, rewards, next_obs, terminals
