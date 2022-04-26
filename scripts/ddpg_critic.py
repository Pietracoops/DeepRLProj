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
        # Add Joint angles (5) + status of gripper (1) + action (1)
        linear_layers.append(to_device(nn.Linear(size * size * out_n_channels + 5 + 1 + 1, net_params["size"])))
        linear_layers.append(to_device(activation))

        for _ in range(net_params["n_linear_layers"]):
            linear_layers.append(to_device(nn.Linear(net_params["size"], net_params["size"])))
            linear_layers.append(to_device(activation))

        linear_layers.append(to_device(nn.Linear(net_params["size"], 1)))

        self.mlp = to_device(nn.Sequential(*linear_layers))

    def forward(self, x0, x1, a):
        x = self.fcn(x0)

        x = torch.flatten(x, 1)
        x = torch.cat((x, x1, a.view(-1, 1)), dim=1)

        x = self.mlp(x)
        return x

class DDPGCritic():
    def __init__(self, net_params, load):       
        self.gamma = net_params["gamma"]
        self.polyak_avg = net_params["polyak_avg"]

        self.fcn_path = "../model/critic/fcn.pt"
        self.mlp_path = "../model/critic/mlp.pt"
        
        self.q_net = CriticNetwork(net_params)
        self.q_net_target = CriticNetwork(net_params)
        
        self.lr = net_params["lr"]
        self.grad_norm_clipping = net_params["grad_norm_clipping"]
        self.optimizer = optim.Adam([{'params': self.q_net.fcn.parameters()}, {'params': self.q_net.mlp.parameters()}], lr=self.lr)
        self.loss = nn.SmoothL1Loss()  # AKA Huber loss

        if load:
            self.load()
    
    def forward(self, state, arm_state, action):
        return self.q_net.forward(state, arm_state, action).detach()
    
    def update(self, states, arm_states, actions, next_states, next_arm_state, rewards, terminals, actor):
        q_values = self.q_net.forward(states, arm_states, actions).squeeze()

        next_q_values = self.q_net_target.forward(next_states, next_arm_state, actor.actor_net_target.forward(next_states, next_arm_state)).squeeze()
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
            target_param.data.copy_(self.polyak_avg * param + (1.0 - self.polyak_avg) * target_param)

        for target_param, param in zip(self.q_net_target.mlp.parameters(), self.q_net.mlp.parameters()):
            target_param.data.copy_(self.polyak_avg * param + (1.0 - self.polyak_avg) * target_param)

    def save(self):
        torch.save(self.q_net.fcn.state_dict(), self.fcn_path)
        torch.save(self.q_net.mlp.state_dict(), self.mlp_path)

    def load(self):
        self.q_net.fcn.load_state_dict(torch.load(self.fcn_path))
        self.q_net.mlp.load_state_dict(torch.load(self.mlp_path))
