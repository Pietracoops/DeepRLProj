import torch

from q_net import QNetwork

class DQNAgent():
    
    def __init__(self, agent_params):
        
        self.q_net = QNetwork(agent_params)
        self.replay_buffer = []
        
        self.epsilon = agent_params["epsilon"]
        self.batch_size = agent_params["train_batch_size"]
        
        self.learning_starts = agent_params["learning_starts"]
        self.learning_freq = agent_params["learning_freq"]
        self.target_update_freq = agent_params["target_update_freq"]
        
        self.n_actions = agent_params["n_actions"]
        
        self.t = 0
    
    def get_action(self, state):
    
        random = torch.rand(1)[0]
        if random < self.epsilon:
            return torch.randint(0, self.n_actions)
        return torch.argmax(self.q_net.forward(state))
    
    def store(self, state, action, next_state, reward, terminal):
        pass
    
    def sample(self):
        #return states, actions, next_states, rewards, terminal
        pass
    
    def update(self):
        
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0):

            #states, actions, next_states, rewards, terminal = sample()
            #loss = self.q_net.update(states, actions, next_states, rewards, terminal)

            if self.num_param_updates % self.target_update_freq == 0:
                self.q_net.update_target_network()
        
        self.t += 1