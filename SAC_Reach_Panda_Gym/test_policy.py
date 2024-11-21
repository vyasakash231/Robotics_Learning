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
env = gym.make("PandaReach-v3", reward_type="dense", render_mode="human")

observation_shape = len(obs_convert(env.observation_space.sample()))
action_shape = env.action_space.shape[0]
action_max = env.action_space.high
action_min = env.action_space.low

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

actor = Actor(observation_shape, action_shape, action_max, action_min).to(device)
actor.load_model("Reach_20240528_Seed_0",13000)  ## Write Folder name and policy no
actor.train(False)

# TRY NOT TO MODIFY: start the game
for _ in range(10):
    obs = obs_convert(env.reset()[0])
    while True:
        obs_t = torch.from_numpy(np.array([obs])).type('torch.FloatTensor').to(device)
        act = actor.get_action(obs_t, deterministic=True)
        actions = act[0].cpu().detach().numpy()
        
        next_obs, rewards, terminations, truncations, _ = env.step(actions)

        obs = obs_convert(next_obs)

        time.sleep(0.1)
        if terminations or truncations:
            break
env.close()