import numpy as np
import time
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
import panda_gym
from policy_network import Actor
from observation_conversion import obs_convert

# create environment instance
env = gym.make("PandaPush-v3", reward_type="dense", render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

observation_shape = len(obs_convert(env.observation_space.sample()))
action_shape = env.action_space.shape[0]
action_max = torch.from_numpy(env.action_space.high).to(device)
action_min = torch.from_numpy(env.action_space.low).to(device)

actor = Actor(observation_shape, action_shape, 512, 2).to(device)
actor.load_model("Push_20240601_Seed_0",1800000)  ## Write Folder name and policy no
actor.train(False)

# TRY NOT TO MODIFY: start the game
for _ in range(10):
    obs = obs_convert(env.reset()[0])
    while True:
        obs_t = torch.from_numpy(np.array([obs])).type('torch.FloatTensor').to(device)
        distribution = actor(obs_t)
        act = distribution.mean
        act = act.clamp(action_min, action_max)
        actions = act[0].cpu().detach().numpy()
        
        next_obs, rewards, terminations, truncations, _ = env.step(actions)

        obs = obs_convert(next_obs)

        time.sleep(0.1)
        if terminations or truncations:
            break
env.close()