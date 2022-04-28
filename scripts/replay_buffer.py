import numpy as np
import torch

from utils import to_device

class ReplayBuffer():
    
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.current_size = 0
        
        self.states = []
        self.arm_states = []
        self.actions = []
        self.next_states = []
        self.next_arm_states = []
        self.rewards = []
        self.terminals = []
        
    def store(self, state, action, next_state, reward, terminal):
        state      = [state[0], state[1]]
        next_state = [next_state[0], next_state[1]]

        state[0]      = np.float16(state[0].cpu().detach().numpy())
        next_state[0] = np.float16(next_state[0].cpu().detach().numpy())

        if state[1] is not None:
            state[1] = state[1].cpu().detach().numpy()
        if next_state[1] is not None:
            next_state[1] = next_state[1].cpu().detach().numpy()

        if self.current_size < self.size:
            self.states.append(state[0])
            self.arm_states.append(state[1])
            self.actions.append(action)
            self.next_states.append(next_state[0])
            self.next_arm_states.append(next_state[1])
            self.rewards.append(reward)
            self.terminals.append(terminal)
            
            self.current_size += 1
            self.index += 1
        else:
            self.index = self.index % self.size
            
            self.states[self.index] = state[0]
            self.arm_states[self.index] = state[1]
            self.actions[self.index] = action
            self.next_states[self.index] = next_state[0]
            self.next_arm_states[self.index] = next_state[1]
            self.rewards[self.index] = reward
            self.terminals[self.index] = terminal
            
            self.index += 1
    
    def sample_dqn(self, batch_size):
        indices = np.random.randint(0, self.current_size, size=batch_size)

        states      = to_device(torch.cat([torch.from_numpy(np.float32(self.states[i])) for i in indices], dim=0))
        actions     = to_device(torch.tensor([self.actions[i] for i in indices], dtype=torch.long))
        next_states = to_device(torch.cat([torch.from_numpy(self.next_states[i]) for i in indices], dim=0))
        rewards     = to_device(torch.tensor([self.rewards[i] for i in indices], dtype=torch.float32))
        terminals   = to_device(torch.tensor([self.terminals[i] for i in indices], dtype=torch.float32))
        return states, actions, next_states, rewards, terminals

    def sample_ddpg(self, batch_size):
        indices = np.random.randint(0, self.current_size, size=batch_size)

        states          = to_device(torch.cat([torch.from_numpy(np.float32(self.states[i])) for i in indices], dim=0))
        arm_states      = to_device(torch.cat([torch.from_numpy(self.arm_states[i]) for i in indices], dim=0))
        actions         = to_device(torch.tensor([self.actions[i] for i in indices], dtype=torch.float32))
        next_states     = to_device(torch.cat([torch.from_numpy(np.float32(self.next_states[i])) for i in indices], dim=0))
        next_arm_states = to_device(torch.cat([torch.from_numpy(self.next_arm_states[i]) for i in indices], dim=0))
        rewards         = to_device(torch.tensor([self.rewards[i] or i in indices], dtype=torch.float32))
        terminals       = to_device(torch.tensor([self.terminals[i] for i in indices], dtype=torch.float32))
        return states, arm_states, actions, next_states, next_arm_states, rewards, terminals