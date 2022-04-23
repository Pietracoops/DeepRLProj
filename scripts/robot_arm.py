import numpy as np
import time
import torch

from pyrobot import Robot

class RobotArm():
    
    def __init__(self):
        self.bot = Robot('locobot')
        self.bot.camera.reset()
        self.bot.camera.set_tilt(0.5, wait=True)

        time.sleep(1)
    
    def do_action(self, action):
        pass
        #robot moves arm in grid according to "action"