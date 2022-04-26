import math
import numpy as np
import time
import torch

from pyrobot import Robot

JOINT_LIMIT_MAX = np.pi * 0.5
JOINT_LIMIT_MIN = -np.pi * 0.5
ACTION_UPPER_LIMIT = 3.5
ACTION_LOWER_LIMIT = -3.5
NUM_ACTIONS = 7
ACTION_INCREMENTS = ((ACTION_UPPER_LIMIT - ACTION_LOWER_LIMIT) / NUM_ACTIONS)
ACTION_UPPER_JOINT_LIMIT = ACTION_UPPER_LIMIT - ACTION_INCREMENTS
ACTION_LOWER_JOINT_LIMIT = ACTION_LOWER_LIMIT + ACTION_INCREMENTS

class RobotArm():
    def __init__(self):
        self.bot = Robot('locobot')
        self.bot.camera.reset()
        self.bot.camera.set_tilt(0.6, wait=True)

        self.bot.arm.go_home()
        self.bot.gripper.open(wait=True)
        time.sleep(1)
    
    def do_action(self, action):
        result = False
        use_joints = False

        joint = self.bot.arm.get_joint_angles() 
        action = np.clip(action.item(), ACTION_LOWER_LIMIT, ACTION_UPPER_LIMIT)
        if action >= ACTION_LOWER_LIMIT and action <= ACTION_LOWER_LIMIT + ACTION_INCREMENTS:
            if self.bot.gripper.get_gripper_state() != 0:
                result = True

            result = self.bot.gripper.open(wait=False)
        elif action > ACTION_UPPER_LIMIT - ACTION_INCREMENTS and action <= ACTION_UPPER_LIMIT:
            if self.bot.gripper.get_gripper_state() != 0:
                result = True

            result = self.bot.gripper.close(wait=False)
        else:
            use_joints = True
            joint_num = int(math.floor(action - ACTION_LOWER_JOINT_LIMIT))
            current_lower = (joint_num * ACTION_INCREMENTS) + ACTION_LOWER_JOINT_LIMIT
            current_upper = current_lower + ACTION_INCREMENTS
            joint[joint_num] = JOINT_LIMIT_MIN + (JOINT_LIMIT_MAX - JOINT_LIMIT_MIN) * ((action - current_lower) / (current_upper - current_lower))

        if use_joints:
            result = True
            self.bot.arm.set_joint_positions(joint, plan=False, wait=False)
            time.sleep(0.5)
        return result

    def reset(self):
        self.bot.arm.go_home()
        time.sleep(1)