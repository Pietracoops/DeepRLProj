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

class ActorNetwork():
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
        # Add Joint angles (5) + status of gripper (1)
        linear_layers.append(to_device(nn.Linear(size * size * out_n_channels + 5 + 1, net_params["size"])))
        linear_layers.append(to_device(activation))

        for _ in range(net_params["n_linear_layers"]):
            linear_layers.append(to_device(nn.Linear(net_params["size"], net_params["size"])))
            linear_layers.append(to_device(activation))

        linear_layers.append(to_device(nn.Linear(net_params["size"], 1)))

        self.mlp = to_device(nn.Sequential(*linear_layers))

    def forward(self, x0, x1):
        x = self.fcn(x0)

        x = torch.flatten(x, 1)
        x = torch.cat((x, x1), dim=1)

        x = self.mlp(x)
        return x

class DDPGActor():
    
    def __init__(self, net_params):
        self.actor_net = ActorNetwork(net_params)
        self.actor_net_target = ActorNetwork(net_params)
        
        self.lr = net_params["lr"]
        self.grad_norm_clipping = net_params["grad_norm_clipping"]
        self.optimizer = optim.Adam([{'params': self.actor_net.fcn.parameters()}, {'params': self.actor_net.mlp.parameters()}], lr=self.lr)
    
    def forward(self, state, arm_state):
        return self.actor_net.forward(state, arm_state).detach()
    
    def update(self, states, arm_states, critic):
        actions = self.actor_net.forward(states, arm_states)
        loss = torch.neg(critic.q_net.forward(states, arm_states, actions)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.actor_net.fcn.parameters(), self.grad_norm_clipping)
        utils.clip_grad_value_(self.actor_net.mlp.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        return { "loss": loss.item(), "actions": torch.mean(actions).item() }
        
    def update_target_network(self):
        for target_param, param in zip(self.actor_net_target.fcn.parameters(), self.actor_net.fcn.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.actor_net_target.mlp.parameters(), self.actor_net.mlp.parameters()):
            target_param.data.copy_(param.data)