import numpy as np
import torch

from robot_arm import RobotArm

def Environment():
    
    def __init__(self):
        self.arm = RobotArm()
    
    def update(self, action):
        pass
        self.arm.do_action(action)
        #next_state = self.get_state()
        #reward = self.get_reward(self, state, next_state)
        #terminal = self.get_reward(self, next_state)
        #return next_state, reward, terminal
        
    def get_reward(self, state, next_state):
        reward = 0.0
        #check gripper sensors, if object between grippers
        #reward = 1.0
        #else if sum(next_state - states) > threshold:
        #reward = 0.5
        return reward
    
    def get_terminal(self, next_state):
        terminal = 0.0
        #terminal = 1.0 if the the depth of of every image = default depth value
        return terminal
    
    def get_state(self):
        state = None
        #Set state to torch.from_numpy(camera.get_picture()) #returns depth and rgb image
        return state