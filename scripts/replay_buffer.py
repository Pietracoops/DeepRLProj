import numpy as np
import torch

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
        return self.states[indices], self.actions[indices], self.next_states[indices], self.rewards[indices], self.terminal[indices]