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

from policy_network import Actor
from critic_network import SoftQNetwork
from memory import ReplayBuffer


#def get_gpu_usage():
#    output = subprocess.run(
#        ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used", "--format=csv,nounits,noheader"],
#        capture_output=True, text=True
#    )
#    gpu_utilization, memory_utilization, memory_total, memory_free, memory_used = map(float, output.stdout.strip().split(", "))
#    return gpu_utilization, memory_utilization, memory_total, memory_free, memory_used


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
        self.actor = Actor(obs_shape, act_shape, act_max, act_min, no_of_neurons, 2).to(device)
        self.qf1_target = SoftQNetwork(obs_shape, act_shape, no_of_neurons, 2).to(device)
        self.qf2_target = SoftQNetwork(obs_shape, act_shape, no_of_neurons, 2).to(device)

        self.qf1_target.load_state_dict(self.qf1.state_dict()) # initial hard update target network weights
        self.qf2_target.load_state_dict(self.qf2.state_dict()) # initial hard update target network weights

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr, betas=[0.9,0.999], weight_decay=0.0005)
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

        # Log GPU usage
        #gpu_utilization, memory_utilization, memory_total, memory_free, memory_used = get_gpu_usage()
        #self.writer.add_scalar('GPU/Utilization', gpu_utilization, global_step)
        #self.writer.add_scalar('GPU/Memory Utilization', memory_utilization, global_step)
        #self.writer.add_scalar('GPU/Memory Total', memory_total, global_step)
        #self.writer.add_scalar('GPU/Memory Free', memory_free, global_step)
        #self.writer.add_scalar('GPU/Memory Used', memory_used, global_step)

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
    n_episodes = 10000
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.015
    batch_size: int = 512
    learning_starts: int = 5000
    policy_lr: float = 3e-4
    q_lr: float = 1e-4
    target_network_frequency: int = 1 
    alpha: float = 0.2
    autotune: bool = True
    save_model:int = 50000
    max_episode_length = 500
    
    no_of_neurons = 256
    no_of_layers = 2

    #controller_type = "JOINT_VELOCITY"
    controller_type = "OSC_POSE"
    control_freq = 25


    """
    The expected action space of the OSC_POSE controller (without a gripper) is (dx, dy, dz, droll, dpitch, dyaw)

    OSC_POSE: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis, z-axis, relative to the global coordinate frame
    OSC_POSITION: Gripper moves sequentially and linearly in x, y, z direction, relative to the global coordinate frame
    IK_POSE: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis, z-axis, relative to the local robot end effector frame
    JOINT_POSITION: Robot Joints move sequentially in a controlled fashion
    JOINT_VELOCITY: Robot Joints move sequentially in a controlled fashion
    JOINT_TORQUE: Unlike other controllers, joint torque controller is expected to act rather lethargic, as the "controller" is really just a wrapper for direct torque control of the mujoco actuators. 
    Therefore, a "neutral" value of 0 torque will not guarantee a stable robot when it has non-zero velocity!
    """

    # create environment instance
    env_suite = suite.make(env_name="Lift",    # try with other tasks like "Stack" and "Door"
        robots="UR5e",       # try with other robots like "Sawyer","Jaco","Panda","Kinova Gen3","Baxter"
        controller_configs=suite.load_controller_config(default_controller=controller_type),  # controller option --> OSC_POSE, OSC_POSITION, JOINT_POSITION, JOINT_VELOCITY, and JOINT_TORQUE.
        has_renderer=False,
        use_object_obs=True,   # provide object observations to agent
        use_camera_obs=False,  # Take camera image as observation data from the environment (For CNN)
        render_camera="frontview",  # to view robot from front view
        has_offscreen_renderer=False,  
        horizon=max_episode_length,    #  Every episode lasts for exactly @horizon timesteps.
        hard_reset=True,
        # ignore_done=True,    #  True if never terminating the environment (ignore @horizon)
        reward_shaping=True,   # allow reward shaping --> dense reward
        #reward_scale=None, # if none, it will not normalize the reward
        control_freq=control_freq)

    # Make sure we only pass in the proprio and object obs (no images)
    # env = NormalizedBoxEnv(GymWrapper(env_suite))
    env = GymWrapper(env_suite)
    
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get the current date and time in YYYYMMDD_HHMMSS format
    folder_name = f"Lift_{current_datetime}"

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    print(f"device: {device}")

    """
    1) Robotic arm --> (37,)
    # robot_joint_pos_cos, robot_joint_pos_sin --> robotic arm joint positions encoded in cos (6,) and sine (6,)
    # Robot arm joint velocities: robot_joint_vel --> Robot arm joint velocities (6,)
    # Robot arm end-effector position: robot_eef_pos --> Robot arm end-effector position (3,)
    # end-effector orientation in quaternion: robot_eef_quat --> Robot arm end-effector orientation in quaternion (4,)
    # Gripper finger Joint Position: robot_gripper_qpos --> Gripper finger Position (6,)
    # Gripper finger Joint velocities: robot_gripper_qvel --> Gripper finger velocities (6,)

    2) environment --> (10,)
    # Cube position: cube_pos --> Cube position (3,)
    # Cube orientation in quaternion: cube_quat --> Cube orientation in quaternion (4,)
    # gripper to cube position: gripper_to_cube_pos --> gripper to cube position (3,)
    """

    # Environment dimensions
    observation_shape = env.observation_space.shape[0]  # 47
    action_shape = env.action_space.shape[0]  # 7
    action_max = torch.from_numpy(env.action_space.high).to(device)
    action_min = torch.from_numpy(env.action_space.low).to(device)

    global_step = 1
    reward_history = []

    buffer_memory = ReplayBuffer(buffer_size)

    SAC_agent = Agent(device=device, q_lr=q_lr, policy_lr=policy_lr, gamma=gamma, no_of_neurons = no_of_neurons, 
                      learning_starts=learning_starts, tau=tau, target_frequency=target_network_frequency, folder_name=folder_name,
                      obs_shape=observation_shape, act_shape=action_shape, act_max=action_max, act_min=action_min,
                      autotune=autotune, alpha=alpha)
    
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
                       'weight_decay':0.0005,
                       'target_network_update_frequency':target_network_frequency,
                       'constant_alpha':alpha,
                       'autotune_used':autotune,
                       'save_model_after_these_many_steps':save_model,
                       'episode_length:':max_episode_length,
                       'no_of_neurons':no_of_neurons,
                       'no_of_layers':no_of_layers,
                       'controller_type':controller_type,
                       'control_freq':control_freq}
    
    save_hyperparameters_to_json(folder_name,"hyperparameters",hyperparameters)

    # TRY NOT TO MODIFY: start the game
    for i in tqdm(range(1, n_episodes+1)):
        score = 0
        obs = env.reset()
        while True:
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
                        
            """
            This part is added in: 
            1) robosuite/environments/manipulation/lift.py
            2) robosuite/wrappers/gym_wrapper.py
            """
            # task_completed is only true, when episode ended due to task completion
            if info["is_success"] == True: 
                task_completed = 1
            else:
                task_completed = 0

            score += reward
            
            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            buffer_memory.store_transition((obs, action, reward, next_obs, task_completed))

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
            if bool(task_completed) or done:
                reward_history.append(score)   
                avg_reward = np.mean(reward_history[-100:])
                SAC_agent.store_reward(i, score, avg_reward)  # store reward
                break

            global_step += 1

    SAC_agent.writer.close()
    env.close()