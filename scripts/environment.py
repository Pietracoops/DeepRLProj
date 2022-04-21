import numpy as np
import torch

from robot_arm import RobotArm

class Environment():
    
    def __init__(self, config):
        self.arm = RobotArm()

        self.threshold = config["threshold"]

        self.t = 0
        self.max_timesteps = config["max_timesteps"]
        self.reset()
    
    def update(self, action):
        self.t += 1
        self.arm.do_action(action)
        next_state = self.get_state()
        #reward = self.get_reward(self, state, next_state)
        #terminal = self.get_terminal(self, next_state)
        #return next_state, reward, terminal
        
    def get_reward(self, state, next_state):
        reward = 0.0
        #check gripper sensors, if object between grippers
        #make arm put object in bin
        #reward = 1.0
        #else if sum(next_state - states) > threshold:
        #reward = 0.5
        return reward
    
    def get_terminal(self, next_state):
        terminal = 0.0
        if self.t > self.max_timesteps:
            terminal = 1.0
        return terminal
    
    def get_state(self):
        #Set state to torch.from_numpy(camera.get_picture()) #returns depth and rgb image
        rgb, depth = self.arm.bot.camera.get_rgb_depth()

        x, y = depth.shape
        size = np.minimum(x, y)
        rgb = rgb[:size, :size, :]
        depth = depth[:size, :size, np.newaxis]
        print("rgb: {} depth: {}".format(type(rgb), type(depth)))
        print("rgb: {} depth: {}".format(rgb.shape, depth.shape))

        state = torch.from_numpy(np.concatenate((rgb, depth), axis=2))
        H, W, C = state.shape
        state = state.reshape(C, H, W).unsqueeze(0)
        print("state: {}".format(type(state)))
        print("state: {}".format(state.shape))
        return state
    
    def reset(self):
        self.t = 0
        #move the arm back to default position
        #We place blocks back in the workspace
        #input("Continue?")