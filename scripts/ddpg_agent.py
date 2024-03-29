import numpy as np
import torch

from ddpg_actor import DDPGActor
from ddpg_critic import DDPGCritic
from replay_buffer import ReplayBuffer

class DDPGAgent():
    
    def __init__(self, n_iter, agent_params):
        self.critic = DDPGCritic(agent_params["q_net"], agent_params["load"])
        self.actor = DDPGActor(agent_params["actor_net"], agent_params["load"])

        self.n_actions = 5.0 + 2.0 # Number of Joints + Open and Close Grippers
        print("Size of action space: {}".format(self.n_actions))
        print(self.critic.q_net.fcn)
        print(self.critic.q_net.mlp)
        print(self.actor.actor_net.fcn)
        print(self.actor.actor_net.mlp)

        self.replay_buffer = ReplayBuffer(agent_params["size"])
        
        self.noise = agent_params["noise"]
        
        self.learning_starts = agent_params["learning_starts"]
        self.learning_freq = agent_params["learning_freq"]
        self.target_update_freq = agent_params["target_update_freq"]
        self.batch_size = agent_params["batch_size"]

        self.save_freq = agent_params["save_freq"]
        
        self.t = 0
        self.num_param_updates = 0
    
    def get_action(self, state):
        perform_random_action = np.random.normal(0.0, self.noise)
        action = self.actor.forward(state[0], state[1]) + perform_random_action
        return action.item()
    
    def store(self, state, action, next_state, reward, terminal):
        self.replay_buffer.store(state, action, next_state, reward, terminal)
    
    def sample(self):
        states, arm_states, actions, next_states, next_arm_states, rewards, terminals = self.replay_buffer.sample_ddpg(self.batch_size)
        return states, arm_states, actions, next_states, next_arm_states, rewards, terminals

    def update(self):
        logs = { }
        if (self.t > self.learning_starts and 
            self.t % self.learning_freq == 0 and
            self.replay_buffer.current_size > self.batch_size
            ):

            states, arm_states, actions, next_states, next_arm_states, rewards, terminals = self.sample()
            logs["critic"] = self.critic.update(states, arm_states, actions, next_states, next_arm_states, rewards, terminals, self.actor)
            logs["actor"] = self.actor.update(states, next_arm_states, self.critic)

            del states
            del arm_states
            del actions
            del next_states
            del next_arm_states
            del rewards
            del terminals
            torch.cuda.empty_cache()

            self.num_param_updates += 1
            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.update_target_network()
                self.actor.update_target_network() 

            if self.t % self.save_freq == 0:
                self.actor.save()
                self.critic.save()

        self.t += 1 
        return logs