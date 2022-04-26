import numpy as np
import time
import torch

from robot_arm import RobotArm
from pyrobot.utils.util import try_cv2_import
cv2 = try_cv2_import()

import rospy
from std_srvs.srv import Empty

from utils import to_device
from collision_utils import Collision

class Environment():
    
    def __init__(self, config):
        self.arm = RobotArm(config)

        self.agent = config["alg"]["agent"]
        self.threshold = config["env"]["threshold"]

        self.t = 0
        self.max_timesteps = config["env"]["max_timesteps"]
        self.current_state = None
        self.collision_listener = Collision(config, self.arm.bot)
        self.reset()

        self.collisions = 0
        self.graps = 0

    def update(self, action):
        if self.agent == "dqn":
            return self.update_dqn(action)
        elif self.agent == "ddpg":
            return self.update_ddpg(action)
        else:
            raise NotImplementedError

    def update_dqn(self, action):
        #self.arm.do_action(action)
        next_state = self.get_state()
        reward = self.get_reward_dqn(next_state)
        terminal = self.is_terminal(next_state)

        self.current_state = next_state
        self.t += 1
        return next_state, reward, terminal
    
    def update_ddpg(self, action):
        result, use_gripper = self.arm.do_action(action)
        next_state = self.get_state()
        reward = self.get_reward_ddpg(result, use_gripper, next_state)
        terminal = self.is_terminal(next_state)

        self.current_state = next_state
        self.t += 1
        return next_state, reward, terminal

    def get_reward_dqn(self, next_state):
        reward = 0.0
        #check gripper sensors, if object between grippers
        #make arm put object in bin
        #reward = 1.0
        #else if sum(next_state - self.current_state) > threshold:
        return reward
        
    def get_reward_ddpg(self, result, use_gripper, next_state):
        reward = 0.0
        if self.collision_listener.search_for_collision() == True:
            self.collisions += 1
            reward = 0.5

        if torch.abs(torch.sum(self.current_state[1] - next_state[1])).item() <= self.threshold and not use_gripper:
            reward = -0.1
        if not result:
            reward = -1.0

        if self.arm.bot.gripper.get_gripper_state() == 2:
            self.graps += 1
            self.arm.reset()
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
        arm_state = torch.from_numpy(np.append(arm_state, gripper_state))
        arm_state = arm_state.float().unsqueeze(0)
        arm_state = to_device(arm_state)
        return state, arm_state
    
    def reset(self):
        self.arm.reset()

        rospy.wait_for_service('/gazebo/reset_world')
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_world()

        time.sleep(1)

        self.collision_listener.get_gazebo_models_init()
        self.t = 0
        self.current_state = self.get_state()