import numpy as np
import time
import torch

from pyrobot import Robot

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

        action = np.clip(action.cpu().numpy(), 0.0, 7.0)
        joint = [0.0, 0.0, 0.0, 0.0, 0.0]
        if action >= 0.0 and action <= 1.0:
            result = self.bot.gripper.open(wait=True)
        elif action > 1.0 and action <= 2.0:
            joint[0] = 2.0 * np.pi * (action - 1.0)
            use_joints = True
        elif action > 2.0 and action <= 3.0:
            joint[1] = 2.0 * np.pi * (action - 2.0)
            use_joints = True
        elif action > 3.0 and action <= 4.0:
            joint[2] = 2.0 * np.pi * (action - 3.0)
            use_joints = True
        elif action > 4.0 and action <= 5.0:
            joint[3] = 2.0 * np.pi * (action - 4.0)
            use_joints = True
        elif action > 5.0 and action <= 6.0:
            joint[4] = 2.0 * np.pi * (action - 5.0)
            use_joints = True
        elif action > 6.0 and action <= 7.0:
            result = self.bot.gripper.close(wait=True)

        if use_joints:
            result = self.bot.arm.set_joint_positions(joint, plan=False, wait=True)
        return result

    def reset(self):
        self.bot.arm.go_home()
        time.sleep(1)