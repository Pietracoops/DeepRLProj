import torch

from q_net import QNetwork
from replay_buffer import ReplayBuffer

class DQNAgent():
    
    def __init__(self, n_iter, agent_params):
        
        self.q_net_push = QNetwork(n_iter, agent_params["q_net"])
        self.q_net_grasp = QNetwork(n_iter, agent_params["q_net"])
        self.replay_buffer = ReplayBuffer(agent_params["size"])
        
        self.epsilon = agent_params["epsilon"]
        
        self.learning_starts = agent_params["learning_starts"]
        self.learning_freq = agent_params["learning_freq"]
        self.target_update_freq = agent_params["target_update_freq"]
        
        self.n_actions = agent_params["q_net"]["output_size"] * agent_params["q_net"]["output_size"]
        
        self.t = 0
    
    def get_action(self, state):
        random = torch.rand(1)[0]
        if random < self.epsilon:
            return torch.randint(0, self.n_actions * 2, size=(1,))
        
        push_q_values = self.q_net_push.forward(state).squeeze(0)
        push_argmax = torch.argmax(push_q_values)
        push_max = torch.max(push_q_values)

        grasp_q_values = self.q_net_push.forward(state).squeeze(0)
        grasp_argmax = torch.argmax(push_q_values)
        grasp_max = torch.max(push_q_values)

        if push_max > grasp_max:
            return (0, push_argmax.item())
        return (1, grasp_argmax.item())
    
    def store(self, state, action, next_state, reward, terminal):
        self.replay_buffer.store(state, action, next_state, terminal)
    
    def sample(self):
        return self.replay_buffer.sample(self.batch_size)
    
    def update(self):
        
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0):

            #states, actions, next_states, rewards, terminals = self.sample()
            #loss = self.q_net.update(states, actions, next_states, rewards, terminals)

            if self.num_param_updates % self.target_update_freq == 0:
                self.q_net_push.update_target_network()
                self.q_net_grasp.update_target_network()
        
        self.t += 1
        
        return loss