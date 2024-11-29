import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper

class RoboGymObservationWrapper(ObservationWrapper):
    def __init__(self, env:gym.Env):
        super(RoboGymObservationWrapper, self).__init__(env)
        self.goal=None

    def set_goal(self, goal):
        self.goal = goal

    def reset(self):
        observation = self.env.reset()
        observation = self.process_observation(observation)
        return observation
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.process_observation(observation)
        """
        This part is added in: 
        1) robosuite/environments/manipulation/pick_place.py
        """
        done = self.env.check_success()
        return observation, reward, done, info
    
    def process_observation(self, observation):
        robot_obs = observation[:37]   # observation["robot0_proprio-state"]
        object_obs = observation[37:]  # observation["object-state"]
        object_des_pose = self.env.target_bin_placements[self.env.object_id,:]

        obs_concatenated = np.concatenate((robot_obs, object_obs, object_des_pose))
        return obs_concatenated




