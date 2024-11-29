import numpy as np
import os
import math
import time
import random
import json

from datetime import datetime
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from robosuite.wrappers import GymWrapper
import robosuite as suite
from gym_robotic_custom import RoboGymObservationWrapper

from policy_network import Actor
from critic_network import SoftQNetwork
from memory import ReplayBuffer


class Agent:
    def __init__(self, device=None, q_lr=0, policy_lr=0, gamma=0, tau=0, batch_size=0, folder_name=None, target_frequency=1, 
                 learning_starts=0, obs_shape=0, act_shape=0, act_max=0, act_min=0, alpha=0.2, autotune=False):
        
        self.learning_starts = learning_starts  # start learning after storing these many images in the buffer_memory
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size   
        self.target_network_frequency = target_frequency
        self.writer = SummaryWriter(folder_name)
        self.device = device

        self.qf1 = SoftQNetwork(obs_shape, act_shape).to(device)
        self.qf2 = SoftQNetwork(obs_shape, act_shape).to(device)
        self.actor = Actor(obs_shape, act_shape, act_max, act_min).to(device)
        self.qf1_target = SoftQNetwork(obs_shape, act_shape).to(device)
        self.qf2_target = SoftQNetwork(obs_shape, act_shape).to(device)

        self.qf1_target.load_state_dict(self.qf1.state_dict()) # initial hard update target network weights
        self.qf2_target.load_state_dict(self.qf2.state_dict()) # initial hard update target network weights

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=policy_lr)

        self.autotune = autotune

        # Automatic entropy tuning
        if self.autotune:
            self.target_entropy = -torch.prod(torch.Tensor([act_shape]).to(device)).item()
            self.log_alpha = torch.tensor(np.log(0.1)).to(device) #torch.zeros(1, requires_grad=True, device=device)
            self.log_alpha.requires_grad = True
            self.alpha = self.log_alpha.exp()  # item() method is used to extract a Python scalar from a tensor that contains a single value
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr)
        else:
            self.alpha = alpha

    def train(self,state_vec, action_vec, reward_vec, next_state_vec, done_vector, global_step):
        self.actor.train(True)
        self.qf1.train(True)
        self.qf2.train(True)

        state_vec = torch.FloatTensor(state_vec).to(self.device)
        action_vec = torch.FloatTensor(action_vec).to(self.device)
        reward_vec = torch.FloatTensor(reward_vec).to(self.device)
        next_state_vec = torch.FloatTensor(next_state_vec).to(self.device)
        done_vector = torch.FloatTensor(done_vector).to(self.device)

        '''ALGO LOGIC: training.'''
        # Train Critic
        with torch.no_grad():
            next_pi, next_log_pi = self.actor.get_action(next_state_vec)  # rsample(), generate samples from the distribution and also supports differentiating through the sampler

            qf1_next_target = self.qf1_target(next_state_vec, next_pi)
            qf2_next_target = self.qf2_target(next_state_vec, next_pi)
            
            if self.autotune:
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha.detach() * next_log_pi
            else:
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_pi

            next_q_value = reward_vec + (1 - done_vector) * self.gamma * min_qf_next_target.view(-1) # without view(-1) shape --> (batch_size, 1) and after (batch_size)
        
        self.qf1_a_values = self.qf1(state_vec, action_vec).view(-1)  # without view(-1) shape --> (batch_size, 1) and after (batch_size)
        self.qf2_a_values = self.qf2(state_vec, action_vec).view(-1)
        self.qf1_loss = F.mse_loss(self.qf1_a_values, next_q_value.detach()) 
        self.qf2_loss = F.mse_loss(self.qf2_a_values, next_q_value.detach()) 
        self.qf_loss = (self.qf1_loss + self.qf2_loss)/2
        
        # optimize the model
        self.q_optimizer.zero_grad()
        self.qf_loss.backward()
        self.q_optimizer.step()

        # Train Actor
        pi, log_pi = self.actor.get_action(state_vec)  # rsample(), generate samples from the distribution and also supports differentiating through the sampler
        
        qf1_pi = self.qf1(state_vec, pi)   # without view(-1) shape --> (batch_size, 1)
        qf2_pi = self.qf2(state_vec, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        if self.autotune:
            self.actor_loss = (self.alpha.detach() * log_pi - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        else:
            self.actor_loss = (self.alpha * log_pi - min_qf_pi).mean()
        
        self.actor_optimizer.zero_grad()
        self.actor_loss.backward()
        self.actor_optimizer.step()

        if self.autotune: 
            self.a_optimizer.zero_grad()
            self.alpha_loss = -(self.alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_loss.backward()
            self.a_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # update the target networks
        if global_step % self.target_network_frequency == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.actor.train(False)
        self.qf1.train(False)
        self.qf2.train(False)

    def store_data(self, global_step):
        #self.writer.add_scalar("losses/qf1_loss", self.qf1_loss.item(), global_step)
        #self.writer.add_scalar("losses/qf2_loss", self.qf2_loss.item(), global_step)
        self.writer.add_scalar("losses/qf_loss", self.qf_loss.item(), global_step)
        self.writer.add_scalar("losses/actor_loss", self.actor_loss.item(), global_step)
        if self.autotune:
            self.writer.add_scalar("losses/alpha", self.alpha.item(), global_step)
            self.writer.add_scalar("losses/alpha_loss", self.alpha_loss.item(), global_step)

    def store_reward(self, episode, episodic_reward, avg_episodic_reward):
        self.writer.add_scalar("Reward/episodic_reward",episodic_reward, episode)
        self.writer.add_scalar("Reward/average_reward", avg_episodic_reward, episode)

def save_hyperparameters_to_json(folder_name, file_name, hyperparameters):
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, f"{file_name}.json")
    with open(file_path, 'w') as json_file:
        json.dump(hyperparameters, json_file, indent=4)

#==========================================================================================================================================#

if __name__ == '__main__': 
    seed: int = 0
    cuda: bool = True
    n_episodes = 5000
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.015
    batch_size: int = 256
    learning_starts: int = 2500
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    target_network_frequency: int = 1 
    alpha: float = 0.2
    autotune: bool = True
    save_model:int = 125000
    max_episode_length = 500

    controller_type = "JOINT_VELOCITY"
    # controller_type = "OSC_POSE"
    control_freq = 25

    # create environment instance
    env_suite = suite.make(env_name="PickPlaceMilk",    # try with other tasks like "Stack" and "Door"
        robots="UR5e",       # try with other robots like "Sawyer","Jaco","Panda","Kinova Gen3","Baxter"
        controller_configs=suite.load_controller_config(default_controller=controller_type),  # controller option --> OSC_POSE, OSC_POSITION, JOINT_POSITION, JOINT_VELOCITY, and JOINT_TORQUE.
        has_renderer=True,
        use_object_obs=True,   # provide object observations to agent
        use_camera_obs=False,  # Take camera image as observation data from the environment (For CNN)
        render_camera="frontview",  # to view robot from front view
        has_offscreen_renderer=False,  
        horizon=max_episode_length,    #  Every episode lasts for exactly @horizon timesteps.
        # hard_reset=True,
        # ignore_done=True,    #  True if never terminating the environment (ignore @horizon)
        reward_shaping=True,   # allow reward shaping --> dense reward
        #reward_scale=None, # if none, it will not normalize the reward
        control_freq=control_freq)

    # Make sure we only pass in the proprio and object obs (no images)
    env = GymWrapper(env_suite)
    env = RoboGymObservationWrapper(env)

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get the current date and time in YYYYMMDD_HHMMSS format
    folder_name = f"PickPlaceMilk_{current_datetime}"

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    print(f"device: {device}")

    """ Robot Arm State: {'robot0_joint_pos_cos':6, 'robot0_joint_pos_sin':6, 'robot0_joint_vel':6, 'robot0_eef_pos':3, 'robot0_eef_quat':4, 'robot0_gripper_qpos':6, 'robot0_gripper_qvel':6}"""
    """ Environment State: {'Milk_pos':3, 'Milk_quat':4, 'Milk_to_robot0_eef_pos':3, 'Milk_to_robot0_eef_quat':4, 
                            'Bread_pos':3, 'Bread_quat':4, 'Bread_to_robot0_eef_pos':3, 'Bread_to_robot0_eef_quat':4,
                            'Cereal_pos':3, 'Cereal_quat':4, 'Cereal_to_robot0_eef_pos':3, 'Cereal_to_robot0_eef_quat':4, 
                            'Can_pos':3, 'Can_quat':4, 'Can_to_robot0_eef_pos':3, 'Can_to_robot0_eef_quat':4, 
                            'robot0_proprio-state':37, 'object-state':56}"""

    # Environment dimensions
    state = env.reset()
    observation_shape = len(state)  # 54
    action_shape = env.action_space.shape[0]  # 7
    action_max = torch.from_numpy(env.action_space.high).to(device)
    action_min = torch.from_numpy(env.action_space.low).to(device)

    global_step = 1
    reward_history = []

    SAC_agent = Agent(device=device, q_lr=q_lr, policy_lr=policy_lr, gamma=gamma, learning_starts=learning_starts, tau=tau, 
                      target_frequency=target_network_frequency, folder_name=folder_name, obs_shape=observation_shape, 
                      act_shape=action_shape, act_max=action_max, act_min=action_min, autotune=autotune, alpha=alpha)
    
    # store hyperparameters in folder
    hyperparameters = {'seed':seed,
                       'no_of_episodes':n_episodes,
                       'buffer_memory_size':buffer_size,
                       'gamma':gamma,
                       'tau':tau,
                       'mini_batch_size':batch_size,
                       'steps_before_learning_starts':learning_starts,
                       'policy_learning_rate':policy_lr,
                       'Q_learning_rate':q_lr,
                       'policy_optimizer':'Adam',
                       'Q_optimizer':'Adam',
                       'weight_decay':0.0,
                       'target_network_update_frequency':target_network_frequency,
                       'constant_alpha':alpha,
                       'autotune_used':autotune,
                       'save_model_after_these_many_steps':save_model,
                       'episode_length:':max_episode_length,
                       'controller_type':controller_type,
                       'control_freq':control_freq}
    
    save_hyperparameters_to_json(folder_name,"hyperparameters",hyperparameters)

    buffer_memory = ReplayBuffer(buffer_size, observation_shape, action_shape)
    buffer_memory.load_from_csv(filename="demo/expert_demonstration.npz")

    time.sleep(2)

    # TRY NOT TO MODIFY: start the game
    for i in tqdm(range(1, n_episodes+1)):
        score = 0
        obs = env.reset()
        while True:
            env.render()
            # ALGO LOGIC: put action logic here
            if global_step < learning_starts:
                action = env.action_space.sample()
            else:
                obs_t = torch.from_numpy(np.array([obs])).type('torch.FloatTensor').to(device)
                SAC_agent.actor.train(False)
                act = SAC_agent.actor.get_action(obs_t, differentiable=False) # sample(), generate samples from the distribution and does not supports differentiating through the sampler
                action = act.detach().cpu()[0].numpy()
            
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = env.step(action)  # done, if number of elapsed timesteps is greater than horizon

            score += reward
            
            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            buffer_memory.store_transition(obs, action, reward, next_obs, done)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > batch_size:
                # after every 250 episode reduce the amount of expert_demo and increase robot experience samples for training
                if i % 250 == 0:
                    buffer_memory.expert_data_ratio -= 0.25

                state_vec, action_vec, reward_vec, next_state_vec, done_vector = buffer_memory.sample_buffer(batch_size)
                
                # Perform Learning Step
                SAC_agent.train(state_vec, action_vec, reward_vec, next_state_vec, done_vector, global_step)
                      
                if global_step % 100 == 0:
                    SAC_agent.store_data(global_step)  # Store other data
                
            if global_step % save_model == 0:
                SAC_agent.actor.save_model(folder_name, global_step) # Store Policy 

            global_step += 1

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if done or global_step % max_episode_length == 0:
                # print(done, global_step)
                reward_history.append(score)   
                avg_reward = np.mean(reward_history[-100:])
                SAC_agent.store_reward(i, score, avg_reward)  # store reward
                break

    SAC_agent.writer.close()
    env.close()