import copy
import numpy as np
import torch

from q_net import QNetwork
from replay_buffer import ReplayBuffer

class DQNAgent():
    
    def __init__(self, n_iter, agent_params):
        
        self.q_net_push = QNetwork(n_iter, agent_params["q_net"])
        self.q_net_grasp = copy.copy(self.q_net_push)

        self.n_actions = self.q_net_push.output_size
        print("Output image size: {}".format(np.sqrt(self.n_actions)))
        print("Size of action space: {}".format(self.n_actions))

        self.push_replay_buffer = ReplayBuffer(agent_params["size"])
        self.grasp_replay_buffer = ReplayBuffer(agent_params["size"])
        
        self.epsilon = agent_params["epsilon"]
        
        self.learning_starts = agent_params["learning_starts"]
        self.learning_freq = agent_params["learning_freq"]
        self.target_update_freq = agent_params["target_update_freq"]
        self.batch_size = agent_params["batch_size"]
        
        self.t = 0
        self.num_param_updates = 0
    
    def get_action(self, state):
        random = torch.rand(1)[0]
        if random < self.epsilon:
            return torch.randint(0, 1, size=(1,)).item(), torch.randint(0, self.n_actions, size=(1,)).item()
        
        push_q_values = self.q_net_push.forward(state[0]).squeeze(0)
        push_argmax = torch.argmax(push_q_values)
        push_max = torch.max(push_q_values)

        grasp_q_values = self.q_net_grasp.forward(state[0]).squeeze(0)
        grasp_argmax = torch.argmax(grasp_q_values)
        grasp_max = torch.max(grasp_q_values)

        if push_max > grasp_max:
            return 0, push_argmax.item()
        return 1, grasp_argmax.item()
    
    def store(self, state, action, next_state, reward, terminal):
        if action[0] == 0:
            self.push_replay_buffer.store(state, action[1], next_state, reward, terminal)
        else:
            self.grasp_replay_buffer.store(state, action[1], next_state, reward, terminal)
    
    def sample_push_buffer(self):
        states, actions, next_states, rewards, terminals = self.push_replay_buffer.sample_dqn(self.batch_size)
        return states, actions, next_states, rewards, terminals

    def sample_grasp_buffer(self):
        states, actions, next_states, rewards, terminals = self.grasp_replay_buffer.sample_dqn(self.batch_size)
        return states, actions, next_states, rewards, terminals
    
    def update(self):
        logs = { }
        if (self.t > self.learning_starts and 
            self.t % self.learning_freq == 0 and
            self.push_replay_buffer.current_size > self.batch_size and
            self.grasp_replay_buffer.current_size > self.batch_size
            ):

            states, actions, next_states, rewards, terminals = self.sample_push_buffer()
            logs["push_network"] = self.q_net_push.update(states, actions, next_states, rewards, terminals)

            states, actions, next_states, rewards, terminals = self.sample_grasp_buffer()
            logs["grasp_network"] = self.q_net_grasp.update(states, actions, next_states, rewards, terminals)

            del states
            del actions
            del next_states
            del rewards
            del terminals

            self.num_param_updates += 1
            if self.num_param_updates % self.target_update_freq == 0:
                self.q_net_push.update_target_network()
                self.q_net_grasp.update_target_network()
        
        self.t += 1
        return logs