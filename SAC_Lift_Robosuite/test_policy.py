import numpy as np
import time
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.utils.transform_utils import *

from policy_network import Actor
from normalized_box_env import NormalizedBoxEnv

# import sys
# print("Python executable:", sys.executable)
# print("Python version:", sys.version)

max_episode_length = 300

## create environment instance
env_suite = suite.make(env_name="Lift",    # try with other tasks like "Stack" and "Door"
    robots="UR5e",       # try with other robots like "Sawyer","Jaco","Panda","Kinova Gen3","Baxter"
    controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),  # controller option --> OSC_POSE, OSC_POSITION, JOINT_POSITION, JOINT_VELOCITY, and JOINT_TORQUE.
    has_renderer=True,
    use_object_obs=True,   # provide object observations to agent
    use_camera_obs=False,  # Take camera image as observation data from the environment (For CNN)
    render_camera="frontview",  # to view robot from front view
    has_offscreen_renderer=False,  
    horizon=max_episode_length,    # these episode does not have end time, so we are setting it ourself
    hard_reset=True,
    reward_shaping=True,   # allow reward shaping --> dense reward
    reward_scale=1,
    control_freq=20)  

# # Make sure we only pass in the proprio and object obs (no images)
# # env = NormalizedBoxEnv(GymWrapper(env_suite))
env = GymWrapper(env_suite)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# Environment dimensions
observation_shape = env.observation_space.shape[0]  # 47
action_shape = env.action_space.shape[0]  # 7
action_max = torch.from_numpy(env.action_space.high).to(device)
action_min = torch.from_numpy(env.action_space.low).to(device)

actor = Actor(observation_shape, action_shape, action_max, action_min, 256, 2).to(device)
actor.load_model("Lift_20240805_"+"173618",4700000)  ## Write Folder name and policy no
actor.train(False)

# TRY NOT TO MODIFY: start the game
for _ in range(5):
    obs = env.reset()
    while True:
        env.render()
        obs_t = torch.from_numpy(np.array([obs])).type('torch.FloatTensor').to(device)
        act = actor.test_action(obs_t)
        action = act[0].cpu().detach().numpy()
        
        obs, reward, done, info = env.step(action)
        
        if done:
            break
env.close()


