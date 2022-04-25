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

class CriticNetwork():
    def __init__(self, net_params):
        if isinstance(net_params["activation"], str):
            activation = _str_to_activation[net_params["activation"]]

        conv_layers = []
        size = net_params["input_size"]
        in_n_channels = net_params["n_input_channels"]
        out_n_channels = net_params["n_channels"]
        for _ in range(net_params["n_conv_layers"]):
            conv_layers.append(to_device(nn.Conv2d(in_channels=in_n_channels, 
                                    out_channels=out_n_channels, 
                                    kernel_size=net_params["kernel_size"],  
                                    stride=net_params["stride"], 
                                    padding=net_params["padding"])))
            
            size = get_size(size, net_params["kernel_size"], net_params["padding"], net_params["stride"])
            
            conv_layers.append(to_device(activation))
            
            in_n_channels = net_params["n_channels"]
            out_n_channels = net_params["n_channels"]

        self.fcn = to_device(nn.Sequential(*conv_layers))

        linear_layers = []
        linear_layers.append(to_device(nn.Linear(size * size * out_n_channels + 1, net_params["size"])))
        linear_layers.append(to_device(activation))

        for _ in range(net_params["n_linear_layers"]):
            linear_layers.append(to_device(nn.Linear(net_params["size"], net_params["size"])))
            linear_layers.append(to_device(activation))

        linear_layers.append(to_device(nn.Linear(net_params["size"], 1)))

        self.mlp = to_device(nn.Sequential(*linear_layers))

    def forward(self, x, a):
        x = self.fcn(x)

        x = torch.flatten(x, 1)
        x = torch.cat((x, a.view(-1, 1)), dim=1)

        x = self.mlp(x)
        return x

class DDPGCritic():
    def __init__(self, net_params):        
        self.gamma = net_params["gamma"]
        
        self.q_net = CriticNetwork(net_params)
        self.q_net_target = CriticNetwork(net_params)
        
        self.grad_norm_clipping = net_params["grad_norm_clipping"]
        self.optimizer = optim.Adam([{'params': self.q_net.fcn.parameters()}, {'params': self.q_net.mlp.parameters()}])
        self.loss = nn.SmoothL1Loss()  # AKA Huber loss
    
    def forward(self, state, action):
        return self.q_net(state, action).detach()
    
    def update(self, states, actions, next_states, rewards, terminals, actor):
        q_values = self.q_net.forward(states, actions).squeeze()

        next_q_values = self.q_net_target.forward(next_states, actor.actor_net_target(next_states)).squeeze()
        next_q_values = next_q_values.detach()

        target = rewards + self.gamma * next_q_values * (1.0 - terminals)
        target = target.detach()

        assert q_values.shape == target.shape
        loss = self.loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.fcn.parameters(), self.grad_norm_clipping)
        utils.clip_grad_value_(self.q_net.mlp.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        return { "loss": loss.item(), "q_values": torch.mean(q_values).item(), "target_q_values": torch.mean(next_q_values).item() }
        
    def update_target_network(self):
        for target_param, param in zip(self.q_net_target.fcn.parameters(), self.q_net.fcn.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.q_net_target.mlp.parameters(), self.q_net.mlp.parameters()):
            target_param.data.copy_(param.data)