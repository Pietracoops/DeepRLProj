import numpy as np
import torch

from robot_arm import RobotArm
from pyrobot.utils.util import try_cv2_import
cv2 = try_cv2_import()

from utils import to_device

class Environment():
    
    def __init__(self, config):
        self.arm = RobotArm()

        self.agent = config["alg"]["agent"]
        self.threshold = config["env"]["threshold"]

        self.t = 0
        self.max_timesteps = config["env"]["max_timesteps"]
        self.current_state = None
        self.reset()
    
    def update(self, action):
        result = self.arm.do_action(action)
        next_state = self.get_state()
        reward = self.get_reward(result)
        terminal = self.is_terminal(next_state)

        self.current_state = next_state
        self.t += 1
        return next_state, reward, terminal
        
    def get_reward(self, result):
        reward = 0.0
        if not result:
            reward = -1.0
        if self.arm.bot.gripper.get_gripper_state() == 2:
            #make arm put object in bin
            reward = 1.0
        return reward
    
    def is_terminal(self, next_state):
        terminal = 0.0
        if self.t > self.max_timesteps:
            terminal = 1.0
        return terminal
    
    def get_state(self):
        rgb, depth = self.arm.bot.camera.get_rgb_depth()

        x, y = depth.shape
        size = np.minimum(x, y)
        rgb = rgb[:size, :size, :]
        depth = depth[:size, :size, np.newaxis] * 1000.0

        state = to_device(torch.from_numpy(np.concatenate((rgb, depth), axis=2)))
        H, W, C = state.shape
        state = state.reshape(C, H, W).unsqueeze(0)

        if self.agent == "dqn":
            return state, None

        arm_state = self.arm.bot.arm.get_joint_angles()     
        gripper_state = self.arm.bot.gripper.get_gripper_state()
        arm_state = to_device(torch.from_numpy(np.append(arm_state, gripper_state)))
        arm_state = arm_state.float().unsqueeze(0)
        return state, arm_state
    
    def reset(self):
        self.t = 0
        self.current_state = self.get_state()