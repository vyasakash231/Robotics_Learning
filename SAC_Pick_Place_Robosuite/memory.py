import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size, obs_dims, n_actions, augment_data=False, augment_noise_ratio=0.1, expert_data_ratio=0.5, device=None):
        self.max_memory_size = max_size   # max capacity of memory
        self.memory_count = 0   # will keep track of first unsaved memory(use it to add new memory to replay buffer)
        self.augment_data = augment_data
        self.augment_noise_ratio = augment_noise_ratio   # btw 0 and 1
        self.expert_data_ratio = expert_data_ratio   # btw 0 and 1
        self.expert_data_cutoff = 0
        self.device = device

        self.state_memory = np.zeros((self.max_memory_size, obs_dims), dtype=np.float32)
        self.action_memory = np.zeros((self.max_memory_size, n_actions), dtype=np.int32)
        self.reward_memory = np.zeros(self.max_memory_size, dtype=np.float32)
        self.next_state_memory = np.zeros((self.max_memory_size, obs_dims), dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_memory_size, dtype=np.int32)

    # add transition to the memory buffer
    def store_transition(self, state, action, reward, new_state, done): 
        index = self.memory_count % self.max_memory_size 
        
        self.state_memory[index,:] = state
        self.action_memory[index,:] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index,:] = new_state
        self.terminal_memory[index] = int(done)
        self.memory_count += 1

    def sample_buffer(self, batch_size):
        size_of_data_available = min(self.memory_count, self.max_memory_size)

        if self.expert_data_ratio > 0:
            expert_data_size = int(batch_size * self.expert_data_ratio)
            random_batch = np.random.choice(size_of_data_available, batch_size-expert_data_size)
            expert_batch = np.random.choice(self.expert_data_cutoff, expert_data_size)
            batch = np.concatenate((random_batch, expert_batch))
        else:
            batch = np.random.choice(size_of_data_available, batch_size, replace=False) # replace=False, means once a memory is sampled we won't sample it again

        states = self.state_memory[batch,:]
        actions = self.action_memory[batch,:]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch,:]
        terminal = self.terminal_memory[batch]

        if self.augment_data:  # add noise it data
            states_noise_std = self.augment_noise_ratio * np.mean(np.abs(states))
            action_noise_std = self.augment_noise_ratio * np.mean(np.abs(actions))

            states = states + np.random.normal(0, states_noise_std, states.shape)
            actions = actions + np.random.normal(0, action_noise_std, actions.shape)

        states = torch.from_numpy(np.array(states)).type('torch.FloatTensor').to(self.device)
        actions = torch.from_numpy(np.array(actions)).type('torch.FloatTensor').to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).type('torch.FloatTensor').to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).type('torch.FloatTensor').to(self.device)
        terminal = torch.from_numpy(np.array(terminal)).type('torch.FloatTensor').to(self.device)

        return states, actions, rewards, next_states, terminal   
    
    def save_to_csv(self, filename):
        np.savez(filename, state=self.state_memory[:self.memory_count],
                        action=self.action_memory[:self.memory_count],
                        reward=self.reward_memory[:self.memory_count],
                        next_state=self.next_state_memory[:self.memory_count],
                        terminal=self.terminal_memory[:self.memory_count])
        
    def load_from_csv(self, filename, expert_data=True):
        try:
            data = np.load(filename)
            self.memory_count = len(data["state"])
            self.state_memory[:self.memory_count] = data["state"]
            self.action_memory[:self.memory_count] = data["action"]
            self.reward_memory[:self.memory_count] = data["reward"]
            self.next_state_memory[:self.memory_count] = data["next_state"]
            self.terminal_memory[:self.memory_count] = data["terminal"]
            print(f"successfully loaded {filename} into memory.")
            print(f"{self.memory_count} memories loaded")

            if expert_data:
                self.expert_data_cutoff = self.memory_count
        except:
            print(f"unable to load data from {filename}")