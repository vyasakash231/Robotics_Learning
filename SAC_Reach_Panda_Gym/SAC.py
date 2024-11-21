import numpy as np
import os
import math
import random
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import panda_gym

from policy_network import Actor
from critic_network import SoftQNetwork
from memory import ReplayBuffer
from observation_conversion import obs_convert

class Agent:
    def __init__(self, device=None, q_lr=0, policy_lr=0, gamma=0, tau=0, batch_size=0, folder_name=None, policy_frequency=1,
                target_frequency=1, learning_starts=0, obs_shape=0, act_shape=0, act_max=0, act_min=0, alpha=0.2, autotune=False):
        
        self.learning_starts = learning_starts  # start learning after storing these many images in the buffer_memory
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size   
        self.actor_update_freq = policy_frequency
        self.target_network_frequency = target_frequency
        self.writer = SummaryWriter(folder_name)

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
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()  # item() method is used to extract a Python scalar from a tensor that contains a single value
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr)
        else:
            self.alpha = alpha

    def learn(self,state_vec, action_vec, reward_vec, next_state_vec, done_vector, global_step):
        self.actor.train(True)
        self.qf1.train(True)
        self.qf2.train(True)

        # ALGO LOGIC: training.
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(next_state_vec)
            qf1_next_target = self.qf1_target(next_state_vec, next_state_actions)
            qf2_next_target = self.qf2_target(next_state_vec, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi #.unsqueeze(-1)
            next_q_value = reward_vec.flatten() + (1 - done_vector.flatten()) * self.gamma * (min_qf_next_target).view(-1)

        self.qf1_a_values = self.qf1(state_vec, action_vec).view(-1)
        self.qf2_a_values = self.qf2(state_vec, action_vec).view(-1)
        self.qf1_loss = F.mse_loss(self.qf1_a_values, next_q_value) 
        self.qf2_loss = F.mse_loss(self.qf2_a_values, next_q_value) 
        self.qf_loss = (self.qf1_loss + self.qf2_loss)/2
        
        # optimize the model
        self.q_optimizer.zero_grad()
        self.qf_loss.backward()
        self.q_optimizer.step()

        if global_step % self.actor_update_freq == 0:  # TD 3 Delayed update support
            for _ in range(self.actor_update_freq):  # compensate for the delay by doing 'actor_update_interval' instead of 1                
                pi, log_pi, _ = self.actor.get_action(state_vec) 
                
                qf1_pi = self.qf1(state_vec, pi)
                qf2_pi = self.qf2(state_vec, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                self.actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

                self.actor_optimizer.zero_grad()
                self.actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune: 
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(state_vec)
                    self.alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy)).mean()
                    
                    self.a_optimizer.zero_grad()
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
        self.writer.add_scalar("losses/qf1_loss", self.qf1_loss.item(), global_step)
        self.writer.add_scalar("losses/qf2_loss", self.qf2_loss.item(), global_step)
        self.writer.add_scalar("losses/qf_loss", self.qf_loss.item(), global_step)
        self.writer.add_scalar("losses/actor_loss", self.actor_loss.item(), global_step)
        self.writer.add_scalar("losses/alpha", self.alpha, global_step)
        if self.autotune:
            self.writer.add_scalar("losses/alpha_loss", self.alpha_loss.item(), global_step)

    def store_reward(self, episode, episodic_reward, avg_episodic_reward):
        self.writer.add_scalar("Reward/episodic_reward",episodic_reward, episode)
        self.writer.add_scalar("Reward/average_reward", avg_episodic_reward, episode)

#==========================================================================================================================================#

if __name__ == '__main__': 
    seed: int = 0
    cuda: bool = True
    n_episodes = 1500
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 1000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 1
    target_network_frequency: int = 1 
    alpha: float = 0.2
    autotune: bool = True
    save_model:int = 1000

    '''The interaction frequency is 25 Hz. An episode is made of 50 interactions, so the duration of an episode is 2 seconds'''
    env = gym.make("PandaReach-v3", reward_type="dense")

    current_date = datetime.now().strftime("%Y%m%d")  # Get the current date in YYYYMMDD format
    folder_name = f"Reach_{current_date}_Seed_{seed}"

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Environment dimensions
    observation_shape = len(obs_convert(env.observation_space.sample()))
    action_shape = env.action_space.shape[0]
    action_max = env.action_space.high
    action_min = env.action_space.low

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    print(f"device: {device}")

    global_step = 1
    reward_history = []

    buffer_memory = ReplayBuffer(buffer_size)

    SAC_agent = Agent(device=device, q_lr=q_lr, policy_lr=policy_lr, gamma=gamma, policy_frequency=policy_frequency, 
                      learning_starts=learning_starts, tau=tau, target_frequency=target_network_frequency, folder_name=folder_name,
                      obs_shape=observation_shape, act_shape=action_shape, act_max=action_max, act_min=action_min,
                      autotune=autotune, alpha=alpha)

    # TRY NOT TO MODIFY: start the game
    for i in tqdm(range(1, n_episodes+1)):
        score = 0
        obs = obs_convert(env.reset()[0])  # convert dictonary to list
        while True:
            # ALGO LOGIC: put action logic here
            if global_step < learning_starts:
                actions = env.action_space.sample()
            else:
                obs_t = torch.from_numpy(np.array([obs])).type('torch.FloatTensor').to(device)
                SAC_agent.actor.train(False)
                act, _, _ = SAC_agent.actor.get_action(obs_t)
                actions = act.detach().cpu()[0].numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            next_obs = obs_convert(next_obs)

            score += rewards

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            buffer_memory.store_transition((obs, actions, rewards, next_obs, terminations or truncations))

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > batch_size:
                state_vec, action_vec, reward_vec, next_state_vec, done_vector = buffer_memory.sample_buffer(batch_size, device)
                
                # Perform Learning Step
                SAC_agent.learn(state_vec, action_vec, reward_vec, next_state_vec, done_vector, global_step)
                      
                if global_step % 100 == 0:
                    SAC_agent.store_data(global_step)  # Store other data

            if global_step % save_model == 0:
                SAC_agent.actor.save_model(folder_name, global_step) # Store Policy 

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if terminations or truncations:
                reward_history.append(score)   
                avg_reward = np.mean(reward_history[-100:])
                SAC_agent.store_reward(i, score, avg_reward)  # store reward
                break

            global_step += 1

    env.close()
