import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import utils

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

def getHeightAndWidth(height, width, kernel_size, padding, stride):
    return ((height - kernel_size + padding * 2) // stride) + 1, ((width - kernel_size + padding * 2) // stride) + 1

class QNetwork():
    
    def __init__(self, net_params):
        super().__init__()
        
        self.n_layers = net_params["n_layers"]
        self.input_size = net_params["input_size"]
        self.n_input_channels = net_params["n_input_channels"]
        self.n_channels = net_params["n_channels"]
        self.stride = net_params["stride"]
        self.kernel_size = net_params["kernel_size"]
        self.maxpool_kernel_size = net_params["maxpool_kernel_size"]
        
        self.activation = net_params["activation"]
        
        self.gamma = net_params["gamma"]
        
        self.grad_norm_clipping = 10
        self.optimizer = optim.Adam(self.q_net.parameters())
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.optimizer_spec.learning_rate_schedule)
        self.loss = nn.SmoothL1Loss()  # AKA Huber loss
        
        if isinstance(self.activation, str):
            activation = _str_to_activation[self.activation]
        
        layers = []
        size = self.input_size
        n_channels = self.n_input_channels
        for _ in range(self.n_layers):
        
            padding = (size- 1) // 2
            layers.append(nn.Conv2d(in_channels=n_channels, 
                                   out_channels=self.n_channels, 
                                   kernel_size=self.kernel_size,  
                                   stride=self.stride, 
                                   padding=padding))
            
            feature_map_height, feature_map_width = getHeightAndWidth(size, 
                                                                      size,
                                                                      self.kernel_size,
                                                                      padding, 
                                                                      self.stride)
            
            layers.append(nn.MaxPool2d(kernel_size=self.maxpool_kernel_size))
            layers.append(activation)
            
            maxpool_map_height = ((feature_map_height - self.maxpool_kernel_size) // self.maxpool_kernel_size) + 1
            maxpool_map_width = ((feature_map_width - self.maxpool_kernel_size) // self.maxpool_kernel_size) + 1
            
            size = maxpool_map_height
            n_channels = self.n_channels
            
        layers.append(nn.Conv2d(in_channels=n_channels, 
                               out_channels=1, 
                               kernel_size=self.kernel_size,  
                               stride=self.stride, 
                               padding=padding))

        feature_map_height, feature_map_width = getHeightAndWidth(size, 
                                                                  size,
                                                                  self.kernel_size,
                                                                  padding, 
                                                                  self.stride)
        
        layers.append(nn.Linear(feature_map_height * feature_map_width, feature_map_height * feature_map_width))
        self.q_net = nn.Sequential(*layers)
        self.target_q_net = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.q_net(state)
    
    def update(self, states, actions, next_states, rewards, terminal):
        q_values = self.q_net(states)
        q_values = torch.gather(q_values, 1, actions.unsqueeze(1)).squeeze(1)
        
        next_q_values = self.target_q_net(next_states).max(dim=1)
        
        target = rewards + self.gamma * next_q_values * (1.0 - terminal)
        target = target.detach()
        
        assert q_values.shape == target.shape
        loss = self.loss(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.learning_rate_scheduler.step()
        return loss
        
    def update_target_network(self):
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data)