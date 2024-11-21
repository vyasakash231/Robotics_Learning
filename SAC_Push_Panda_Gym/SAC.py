import numpy as np
import os
import math
import time
import random
from datetime import datetime
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import panda_gym

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

from policy_network import Actor
from critic_network import SoftQNetwork
from memory import ReplayBuffer
from observation_conversion import obs_convert

class Agent:
    def __init__(self, device=None, q_lr=0, policy_lr=0, gamma=0, tau=0, batch_size=0, folder_name=None, no_of_neurons=256,
                target_frequency=1, learning_starts=0, obs_shape=0, act_shape=0, act_max=0, act_min=0, alpha=0.2, autotune=False):
        
        self.learning_starts = learning_starts  # start learning after storing these many images in the buffer_memory
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size   
        self.target_network_frequency = target_frequency
        self.writer = SummaryWriter(folder_name)

        self.qf1 = SoftQNetwork(obs_shape, act_shape, no_of_neurons, 2).to(device)
        self.qf2 = SoftQNetwork(obs_shape, act_shape, no_of_neurons, 2).to(device)
        self.actor = Actor(obs_shape, act_shape, no_of_neurons, 2).to(device)
        self.qf1_target = SoftQNetwork(obs_shape, act_shape, no_of_neurons, 2).to(device)
        self.qf2_target = SoftQNetwork(obs_shape, act_shape, no_of_neurons, 2).to(device)

        self.qf1_target.load_state_dict(self.qf1.state_dict()) # initial hard update target network weights
        self.qf2_target.load_state_dict(self.qf2.state_dict()) # initial hard update target network weights

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr, betas=[0.9,0.999])
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=policy_lr, betas=[0.9,0.999])

        self.autotune = autotune

        # Automatic entropy tuning
        if self.autotune:
            self.target_entropy = -torch.prod(torch.Tensor([act_shape]).to(device)).item()
            self.log_alpha = torch.tensor(np.log(0.1)).to(device) #torch.zeros(1, requires_grad=True, device=device)
            self.log_alpha.requires_grad = True
            self.alpha = self.log_alpha.exp()  # item() method is used to extract a Python scalar from a tensor that contains a single value
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr, betas=[0.9,0.999])
        else:
            self.alpha = alpha

    def learn(self,state_vec, action_vec, reward_vec, next_state_vec, done_vector, global_step):
        self.actor.train(True)
        self.qf1.train(True)
        self.qf2.train(True)

        '''ALGO LOGIC: training.'''
        # Train Critic
        with torch.no_grad():
            next_distribution = self.actor(next_state_vec)
            next_pi = next_distribution.rsample()
            next_log_pi = next_distribution.log_prob(next_pi).sum(-1, keepdim=True)

            qf1_next_target = self.qf1_target(next_state_vec, next_pi)
            qf2_next_target = self.qf2_target(next_state_vec, next_pi)
            
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha.detach() * next_log_pi
            next_q_value = reward_vec + (1 - done_vector) * self.gamma * (min_qf_next_target)
        
        self.qf1_a_values = self.qf1(state_vec, action_vec)
        self.qf2_a_values = self.qf2(state_vec, action_vec)
        self.qf1_loss = F.mse_loss(self.qf1_a_values, next_q_value.detach()) 
        self.qf2_loss = F.mse_loss(self.qf2_a_values, next_q_value.detach()) 
        self.qf_loss = (self.qf1_loss + self.qf2_loss)/2
        
        # optimize the model
        self.q_optimizer.zero_grad()
        self.qf_loss.backward()
        self.q_optimizer.step()

        # Train Actor
        distribution = self.actor(state_vec)
        pi = distribution.rsample()
        log_pi = distribution.log_prob(pi).sum(-1, keepdim=True)
        
        qf1_pi = self.qf1(state_vec, pi)
        qf2_pi = self.qf2(state_vec, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        self.actor_loss = (self.alpha.detach() * log_pi - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

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
        self.writer.add_scalar("losses/qf1_loss", self.qf1_loss.item(), global_step)
        self.writer.add_scalar("losses/qf2_loss", self.qf2_loss.item(), global_step)
        self.writer.add_scalar("losses/qf_loss", self.qf_loss.item(), global_step)
        self.writer.add_scalar("losses/actor_loss", self.actor_loss.item(), global_step)
        self.writer.add_scalar("losses/alpha", self.alpha.item(), global_step)
        if self.autotune:
            self.writer.add_scalar("losses/alpha_loss", self.alpha_loss.item(), global_step)

    def store_reward(self, episode, episodic_reward, avg_episodic_reward):
        self.writer.add_scalar("Reward/episodic_reward",episodic_reward, episode)
        self.writer.add_scalar("Reward/average_reward", avg_episodic_reward, episode)

#==========================================================================================================================================#

if __name__ == '__main__': 
    seed: int = 0
    cuda: bool = True
    n_episodes = 60000
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.01
    batch_size: int = 512
    learning_starts: int = 2500
    policy_lr: float = 8e-4
    q_lr: float = 1e-4
    target_network_frequency: int = 1 
    alpha: float = 0.2
    autotune: bool = True
    save_model:int = 100000
    max_episode_length = 100
    
    no_of_neurons = 512
    no_of_layers = 2

    """
    @ achieved_goal: achieved object/cube position (3,), desired_goal: desired object/cube position (3,)
    for Push task observation contains, [robot observation, task observation], the gripper is blocked closed, (So, no observation of gripper)
    * Robot Observation contains: position and speed of the gripper (6,)
    * Task Observation contains: object_position (3,), object_rotation (3,), object_velocity (3,), object_angular_velocity (3,)
    """

    '''The interaction frequency is thus 25 Hz. An episode is made of 50 interactions, so the duration of an episode is 2 seconds'''
    env = gym.make("PandaPushDense-v3", reward_type="dense") #, render_mode='human')

    current_date = datetime.now().strftime("%Y%m%d")  # Get the current date in YYYYMMDD format
    folder_name = f"Push_{current_date}_Seed_{seed}"

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    print(f"device: {device}")

    ## Environment dimensions
    observation_shape = len(obs_convert(env.observation_space.sample()))
    action_shape = env.action_space.shape[0]
    action_max = torch.from_numpy(env.action_space.high).to(device)
    action_min = torch.from_numpy(env.action_space.low).to(device)

    global_step = 1
    reward_history = []

    buffer_memory = ReplayBuffer(observation_shape, action_shape, buffer_size)

    SAC_agent = Agent(device=device, q_lr=q_lr, policy_lr=policy_lr, gamma=gamma, no_of_neurons = no_of_neurons,
                      learning_starts=learning_starts, tau=tau, target_frequency=target_network_frequency, folder_name=folder_name,
                      obs_shape=observation_shape, act_shape=action_shape, act_max=action_max, act_min=action_min,
                      autotune=autotune, alpha=alpha)

    # TRY NOT TO MODIFY: start the game
    for i in tqdm(range(1, n_episodes+1)):
        score, step = 0, 0
        obs = obs_convert(env.reset()[0])  # convert dictonary to list
        while True:
            # ALGO LOGIC: put action logic here
            if global_step < learning_starts:
                actions = env.action_space.sample()
            else:
                obs_t = torch.from_numpy(np.array([obs])).type('torch.FloatTensor').to(device)
                # SAC_agent.actor.train(False)
                distribution = SAC_agent.actor(obs_t)
                act = distribution.sample()
                act = act.clamp(action_min, action_max)
                actions = act.detach().cpu()[0].numpy()
                        
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, _, _, info = env.step(actions)

            # done is only true, when episode ended due to task completion
            if info["is_success"] == True:
                done = 1
            else:
                done = 0

            next_obs = obs_convert(next_obs)

            score += rewards

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            buffer_memory.store_transition(obs, actions, rewards, next_obs, done)

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
            if bool(done) or step==max_episode_length:
                reward_history.append(score)   
                avg_reward = np.mean(reward_history[-100:])
                SAC_agent.store_reward(i, score, avg_reward)  # store reward
                break

            global_step += 1
            step += 1

    env.close()

#==========================================================================================================================================#

# def train_sac(config, checkpoint_dir=None):
#     seed = config.get("seed", 0)
#     cuda = config.get("cuda", True)
#     n_episodes = config.get("n_episodes", 5000)
#     max_episode_length = config.get("max_episode_length", 100)
#     buffer_size = config.get("buffer_size", int(1e6))
    
#     # TRY NOT TO MODIFY: seeding
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    
#     '''The interaction frequency is thus 25 Hz. An episode is made of 50 interactions, so the duration of an episode is 2 seconds'''
#     entry_point = "panda_gym.envs:PandaPushEnv"

#     # Registering the environment
#     gym.register(id="PandaPushDense-v3", entry_point=entry_point)
#     env = gym.make("PandaPushDense-v3", reward_type="dense", control_type="ee")
    
#     current_date = datetime.now().strftime("%Y%m%d")
#     folder_name = f"Push_{current_date}_Seed_{seed}"
    
#     observation_shape = len(obs_convert(env.observation_space.sample()))
#     action_shape = env.action_space.shape[0]
#     action_max = torch.from_numpy(env.action_space.high).to(device)
#     action_min = torch.from_numpy(env.action_space.low).to(device)

#     global_step = 1
    
#     # Define a deque to store the last 100 episode rewards
#     reward_history = deque(maxlen=100)

#     buffer_memory = ReplayBuffer(observation_shape, action_shape, buffer_size)

#     SAC_agent = Agent(
#         device=device,
#         q_lr=config["q_lr"],
#         policy_lr=config["policy_lr"],
#         gamma=config["gamma"],
#         learning_starts=config["learning_starts"],
#         tau=config["tau"],
#         batch_size=config["batch_size"],
#         no_of_neurons=config["no_of_neurons"],
#         target_frequency=config["target_network_frequency"],
#         folder_name=folder_name,
#         obs_shape=observation_shape,
#         act_shape=action_shape,
#         act_max=action_max,
#         act_min=action_min,
#         autotune=True,
#         alpha=0.2
#     )

#     for i in range(1, n_episodes + 1):
#         score, step = 0, 0
#         obs = obs_convert(env.reset()[0])  # convert dictionary to list
#         while True:
#             if global_step < config["learning_starts"]:
#                 actions = env.action_space.sample()
#             else:
#                 obs_t = torch.from_numpy(np.array([obs])).type(torch.FloatTensor).to(device)
#                 distribution = SAC_agent.actor(obs_t)
#                 act = distribution.sample()
#                 act = act.clamp(action_min, action_max)
#                 actions = act.detach().cpu()[0].numpy()

#             next_obs, rewards, _, _, info = env.step(actions)

#             done = 1 if info["is_success"] else 0
#             next_obs = obs_convert(next_obs)
#             score += rewards

#             buffer_memory.store_transition(obs, actions, rewards, next_obs, done)

#             obs = next_obs

#             if global_step > config["batch_size"]:
#                 state_vec, action_vec, reward_vec, next_state_vec, done_vector = buffer_memory.sample_buffer(config["batch_size"], device)
#                 SAC_agent.learn(state_vec, action_vec, reward_vec, next_state_vec, done_vector, global_step)

#                 if global_step % 100 == 0:
#                     SAC_agent.store_data(global_step)

#             if global_step % config["save_model"] == 0:
#                 SAC_agent.actor.save_model(folder_name, global_step)

#             if bool(done) or step == max_episode_length:
#                 reward_history.append(score)
#                 avg_reward = np.mean(reward_history)
#                 SAC_agent.store_reward(i, score, avg_reward)
#                 ray.train.report(dict(episode_reward=score, avg_reward=avg_reward))
#                 break

#             global_step += 1
#             step += 1

#     env.close()

# # Register the trainable function
# tune.register_trainable("train_sac", train_sac)

# search_space = {
#     "q_lr": tune.loguniform(1e-4, 1e-3),
#     "policy_lr": tune.loguniform(1e-4, 1e-3),
#     "gamma": tune.uniform(0.98, 0.99),
#     "tau": tune.uniform(0.005, 0.05),
#     "batch_size": tune.choice([256, 512, 1024]),
#     "learning_starts": tune.choice([1000, 2500, 5000]),
#     "no_of_neurons": tune.choice([256, 512, 768, 1024]),
#     "target_network_frequency": 1,
#     "n_episodes": 5000,
#     "max_episode_length": 100,
#     "save_model": 5000
# }

# '''Sets up the early stopping strategy to terminate underperforming trials early'''
# '''The ASHAScheduler is a state-of-the-art hyperparameter optimization algorithm designed for efficient and scalable tuning'''
# scheduler = ASHAScheduler(metric="avg_reward",   # performance metric you want to optimize
#                           mode="max",    # Since higher rewards are better in most RL tasks, we set this to "max"
#                           max_t=5000 * 100,   # the maximum number of training iterations (or time steps) for a single trial
#                           grace_period=(5000 * 100)/2,    # the minimum number of training iterations that a trial is guaranteed to run before it can be considered for early stopping
#                           reduction_factor=2)

# '''metrics to be displayed during the tuning process'''
# reporter = CLIReporter(metric_columns=["episode_reward", "avg_reward", "training_iteration"])

# analysis = tune.run(train_sac,
#     resources_per_trial={"cpu": 4, "gpu": 1},
#     config=search_space,  # saerch space of hype-parameters
#     num_samples=50,  # 50 different combinations of hyperparameters will be evaluated.
#     scheduler=scheduler, progress_reporter=reporter)

# # Visualizing results
# df = analysis.results_df
# df.to_csv("ray_results.csv")  # Save the results to a CSV file

# # Example: Plot average reward over trials
# plt.figure(figsize=(10, 6))
# plt.plot(df['avg_reward'], label='Average Reward')
# plt.xlabel('Trial')
# plt.ylabel('Average Reward')
# plt.legend()
# plt.title('Average Reward over Trials')
# plt.grid(True)
# plt.savefig('average_reward_over_trials.png')
# plt.show()