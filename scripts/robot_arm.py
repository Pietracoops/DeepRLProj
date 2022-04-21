import numpy as np
import torch

from pyrobot import Robot

class RobotArm():
    
    def __init__(self):
        self.bot = Robot('locobot')
        self.bot.camera.reset()
    
    def do_action(self, action):
        pass
        #robot moves arm in grid according to "action"