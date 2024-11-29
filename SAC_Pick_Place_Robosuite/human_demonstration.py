import sys
import numpy as np
import pygame
import torch
import atexit

from robosuite.wrappers import GymWrapper
import robosuite as suite
from joystick_control import Controller
from gym_robotic_custom import RoboGymObservationWrapper
from robosuite_multiview_custom import MultiViewWrapper

from memory import ReplayBuffer

def cleanup():
    """Cleanup function to properly close environment and pygame"""
    if 'env' in globals():
        env.close()
    pygame.quit()

# Register cleanup function
atexit.register(cleanup)

# Environment settings
n_episodes = 10000
cuda = True
buffer_size = int(1e6)
max_episode_length = 500
controller_type = "JOINT_VELOCITY"
control_freq = 25

device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

# Initialize joystick controller
joystick = Controller(control_mode=controller_type)

# Create environment with all available cameras
env_name="PickPlaceMilk" 
env = suite.make(
    env_name=env_name,
    robots="UR5e",
    controller_configs=suite.load_controller_config(default_controller=controller_type),
    # object_type="milk",
    has_renderer=True,
    use_object_obs=True,
    use_camera_obs=False,
    reward_shaping=True,
    horizon=max_episode_length,
    camera_names=["agentview", "birdview", "frontview", "robot0_robotview"],
    has_offscreen_renderer=False,
    camera_heights=256,
    camera_widths=256,
    control_freq=control_freq,
    # single_object_mode=1,
    )

""" Robot Arm State: {'robot0_joint_pos_cos':6, 'robot0_joint_pos_sin':6, 'robot0_joint_vel':6, 'robot0_eef_pos':3, 'robot0_eef_quat':4, 'robot0_gripper_qpos':6, 'robot0_gripper_qvel':6}"""
""" Environment State: {'Milk_pos':3, 'Milk_quat':4, 'Milk_to_robot0_eef_pos':3, 'Milk_to_robot0_eef_quat':4, 
                        'Bread_pos':3, 'Bread_quat':4, 'Bread_to_robot0_eef_pos':3, 'Bread_to_robot0_eef_quat':4,
                        'Cereal_pos':3, 'Cereal_quat':4, 'Cereal_to_robot0_eef_pos':3, 'Cereal_to_robot0_eef_quat':4, 
                        'Can_pos':3, 'Can_quat':4, 'Can_to_robot0_eef_pos':3, 'Can_to_robot0_eef_quat':4, 
                        'robot0_proprio-state':37, 'object-state':56}"""


# Add custom wrapper and gym wrapper
env = MultiViewWrapper(env, env_name)
env = GymWrapper(env)
env = RoboGymObservationWrapper(env)

state = env.reset()

buffer_memory = ReplayBuffer(buffer_size, len(state), env.action_space.shape[0], device)
buffer_memory.load_from_csv(filename="demo/expert_demonstration.npz")
starting_memory_count = buffer_memory.memory_count

print(f"Start memory count {starting_memory_count}")

# Enable visualization settings
for setting in env.env.get_visualization_settings():
    env.env.set_visualization_setting(setting, True)

running = True

try:
    while running:
        steps = 0
        obs = env.reset()
        done = False
        
        while not done and steps < max_episode_length:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                    
            if not running:
                break

            # Get and apply actions
            action = joystick.get_actions()
            if action is not None:
                next_obs, reward, done, info = env.step(action)
    
                buffer_memory.store_transition(obs, action, reward, next_obs, done)
                print(f"Step: {steps} Reward: {reward} Successfully added {buffer_memory.memory_count - starting_memory_count} steps to memory. Total: {buffer_memory.memory_count}")
        
                obs = next_obs
                steps += 1

            env.render()
            pygame.time.wait(1)

        print("episode ends!!")

        if not running:
            break
            
        buffer_memory.save_to_csv(filename="demo/expert_demonstration.npz")

finally:
    cleanup()