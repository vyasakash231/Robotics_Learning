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

# ## MEMORY 
# class ReplayBuffer():
#     def __init__(self, buffer_limit):
#         self.buffer = collections.deque(maxlen=buffer_limit)
#         self.memory_count = 0

#     def store_transition(self, transition):
#         self.buffer.append(transition)
#         self.memory_count += 1

#     def sample_buffer(self, batch_size, device):
#         mini_batch = random.sample(self.buffer, batch_size)
#         s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

#         for transition in mini_batch:
#             s, a, r, s_prime, done_mask = transition
#             s_lst.append(s)
#             a_lst.append(a)
#             r_lst.append(r)
#             s_prime_lst.append(s_prime)
#             done_mask_lst.append(done_mask)

#         obs = torch.from_numpy(np.array(s_lst)).type('torch.FloatTensor').to(device)
#         actions = torch.from_numpy(np.array(a_lst)).type('torch.FloatTensor').to(device)
#         rewards = torch.from_numpy(np.array(r_lst)).type('torch.FloatTensor').to(device)
#         next_obs = torch.from_numpy(np.array(s_prime_lst)).type('torch.FloatTensor').to(device)
#         terminals = torch.from_numpy(np.array(done_mask_lst)).type('torch.FloatTensor').to(device)

#         return obs, actions, rewards, next_obs, terminals


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity):
        self.capacity = capacity

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        # obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        obs_dtype = np.float32
        self.obses = np.empty((capacity, obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def store_transition(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_buffer(self, batch_size, device):
        idxs = np.random.randint(0,self.capacity if self.full else self.idx,size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=device).float()
        actions = torch.as_tensor(self.actions[idxs], device=device)
        rewards = torch.as_tensor(self.rewards[idxs], device=device)
        next_obses = torch.as_tensor(self.next_obses[idxs],device=device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=device)
        
        return obses, actions, rewards, next_obses, not_dones
