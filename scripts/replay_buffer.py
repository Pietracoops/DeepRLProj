import numpy as np
import torch

from utils import to_device

class ReplayBuffer():
    
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.current_size = 0
        
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.terminals = []
        
    def store(self, state, action, next_state, reward, terminal):
        if self.current_size < self.size:
            self.states.append(state)
            self.actions.append(action)
            self.next_states.append(next_state)
            self.rewards.append(reward)
            self.terminals.append(terminal)
            
            self.current_size += 1
            self.index += 1
        else:
            self.index = self.index % self.size
            
            self.states[self.index] = state
            self.actions[self.index] = action
            self.next_states[self.index] = next_state
            self.rewards[self.index] = reward
            self.terminals[self.index] = terminal
            
            self.index += 1
    
    def sample(self, batch_size):
        indices = np.random.randint(0, self.current_size, size=batch_size)

        states      = to_device(torch.cat([self.states[i] for i in indices], dim=0))
        actions     = to_device(torch.tensor([self.actions[i] for i in indices], dtype=torch.long))
        next_states = to_device(torch.cat([self.next_states[i] for i in indices], dim=0))
        rewards     = to_device(torch.tensor([self.rewards[i] for i in indices], dtype=torch.float32))
        terminals   = to_device(torch.tensor([self.terminals[i] for i in indices], dtype=torch.float32))
        return states, actions, next_states, rewards, terminals