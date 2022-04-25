import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import utils

from utils import to_device

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

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class DDPGActor():
    
    def __init__(self, net_params):
        self.n_conv_layers = net_params["n_conv_layers"]
        self.input_size = net_params["input_size"]
        self.n_input_channels = net_params["n_input_channels"]
        self.n_channels = net_params["n_channels"]
        self.kernel_size = net_params["kernel_size"]
        self.stride = net_params["stride"]
        self.padding = net_params["padding"]
        
        self.activation = net_params["activation"]

        self.size = net_params["size"]
        self.n_linear_layers = net_params["n_linear_layers"]
        
        self.gamma = net_params["gamma"]
        
        self.actor_net = to_device(self.build_nn())
        self.actor_net_target = to_device(self.build_nn())
        
        self.lr = net_params["lr"]
        self.grad_norm_clipping = net_params["grad_norm_clipping"]
        self.optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr)
        
    def build_nn(self):
        if isinstance(self.activation, str):
            activation = _str_to_activation[self.activation]

        layers = []
        size = self.input_size
        in_n_channels = self.n_input_channels
        out_n_channels = self.n_channels
        for _ in range(self.n_conv_layers):
            layers.append(nn.Conv2d(in_channels=in_n_channels, 
                                    out_channels=out_n_channels, 
                                    kernel_size=self.kernel_size,  
                                    stride=self.stride, 
                                    padding=self.padding))
            
            size = get_size(size, self.kernel_size, self.padding, self.stride)
            
            layers.append(activation)
            
            in_n_channels = self.n_channels
            out_n_channels = self.n_channels

        layers.append(Flatten())
        layers.append(nn.Linear(size * size * out_n_channels, self.size))
        layers.append(activation)

        for _ in range(self.n_linear_layers):
            layers.append(nn.Linear(self.size, self.size))
            layers.append(activation)

        layers.append(nn.Linear(self.size, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, state):
        return self.actor_net(state).detach()
    
    def update(self, states, critic):
        actions = self.actor_net(states)
        loss = torch.neg(critic.q_net.forward(states, actions)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.actor_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        return { "loss": loss.item(), "actions": torch.mean(actions).item() }
        
    def update_target_network(self):
        for target_param, param in zip(self.actor_net_target.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(param.data)