import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import utils

from utils import to_device
from utils import robot_optimizer

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus()
}

def get_size(size, kernel_size, padding, stride):
    return ((size - kernel_size + padding * 2) // stride) + 1

class QNetwork():
    
    def __init__(self, n_iter, net_params):
        self.n_layers = net_params["n_layers"]
        self.input_size = net_params["input_size"]
        self.n_input_channels = net_params["n_input_channels"]
        self.n_channels = net_params["n_channels"]
        self.kernel_size = net_params["kernel_size"]
        self.stride = net_params["stride"]
        self.padding = net_params["padding"]
        
        self.activation = net_params["activation"]
        
        self.gamma = net_params["gamma"]
        
        self.q_net = to_device(self.build_nn())
        self.q_net_target = to_device(self.build_nn())
        
        self.grad_norm_clipping = net_params["grad_norm_clipping"]
        self.optimizer_spec = robot_optimizer(n_iter)
        self.optimizer = optim.Adam(self.q_net.parameters())
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.optimizer_spec.learning_rate_schedule)
        self.loss = nn.SmoothL1Loss()  # AKA Huber loss

        self.alg = net_params["alg"]
        
    def build_nn(self):
        if isinstance(self.activation, str):
            activation = _str_to_activation[self.activation]

        layers = []
        size = self.input_size
        in_n_channels = self.n_input_channels
        out_n_channels = self.n_channels
        for _ in range(self.n_layers):
            layers.append(nn.Conv2d(in_channels=in_n_channels, 
                                    out_channels=out_n_channels, 
                                    kernel_size=self.kernel_size,  
                                    stride=self.stride, 
                                    padding=self.padding))
            
            size = get_size(size, self.kernel_size, self.padding, self.stride)
            
            layers.append(activation)
            
            in_n_channels = self.n_channels
            out_n_channels = self.n_channels
            
        layers.append(nn.Conv2d(in_channels=out_n_channels, out_channels=1, kernel_size=1))
        
        self.output_size = size * size
        return nn.Sequential(*layers)
    
    def forward(self, state):
        return torch.flatten(self.q_net(state).detach())
    
    def update(self, states, actions, next_states, rewards, terminals):
        q_values = torch.flatten(self.q_net(states), start_dim=1)
        q_values = torch.gather(q_values, 1, actions.unsqueeze(1)).squeeze(1)
        
        # DQN
        if self.alg == "dqn":
            next_q_values, _ = torch.flatten(self.q_net_target(next_states), start_dim=1).max(dim=1)

        # Double DQN
        if self.alg == "ddqn":
            next_q_values = torch.flatten(self.q_net(next_states), start_dim=1)
            next_actions = torch.argmax(next_q_values, 1).unsqueeze(-1)
            next_q_values = torch.gather(torch.flatten(self.q_net_target(next_states), start_dim=1), 1, next_actions).squeeze(1)

        next_q_values = next_q_values.detach()
        
        target = rewards + self.gamma * next_q_values * (1.0 - terminals)
        target = target.detach()
        
        assert q_values.shape == target.shape
        loss = self.loss(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.learning_rate_scheduler.step()
        return { "loss": loss.item(), "q_values": torch.mean(q_values).item(), "target_q_values": torch.mean(next_q_values).item() }
        
    def update_target_network(self):
        for target_param, param in zip(self.q_net_target.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data)