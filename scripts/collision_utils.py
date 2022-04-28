
import math
import numpy as np
import pose_utils
from gazebo_msgs.srv import GetModelState
import rospy
import numpy as np

class Block:
    def __init__(self, name):
        self._name = name
        self._pose = pose_utils.Pose()

class Collision():
    def __init__(self, config, bot):
        # World specific blocks
        self._blockListDict = {
        'block_a': Block('mass_cube_green'),
        'block_b': Block('mass_cube_green_clone'),
        'block_c': Block('mass_cube_pink'),
        'block_e': Block('mass_cube_blue'),
        'block_f': Block('mass_cube_red'),
        'block_g': Block('mass_cube_red_clone'),
        'block_h': Block('mass_cube_dark_blue')
        }

        self.eps = 0.001
        self.threshold = config["env"]["ee_collision_threshold"]
        self.bot = bot

    def get_gazebo_models_init(self, logging=False):
        try:
            model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            for block in self._blockListDict.itervalues():
                blockName = str(block._name)
                resp_coordinates = model_coordinates(blockName, "")
                block._pose.x = float(resp_coordinates.pose.position.x)
                block._pose.y = float(resp_coordinates.pose.position.y)
                block._pose.z = float(resp_coordinates.pose.position.z)
                if logging is True:
                    print '\n'
                    print 'Status.success = ', resp_coordinates.success
                    print(blockName)
                    print("Cube " + str(block._name))
                    print("Value of X : " + str(resp_coordinates.pose.position.x))
                    print("Value of Y : " + str(resp_coordinates.pose.position.y))
                    print("Value of Z : " + str(resp_coordinates.pose.position.z))
        except rospy.ServiceException as e:
            rospy.loginfo("Get Model State service call failed:  {0}".format(e))

    def search_for_collision(self):
        ret_val = False
        try:
            pose_obj = pose_utils.Pose()
            pose_obj.UpdateRobotPose(self.bot)
            model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            for block in self._blockListDict.itervalues():
                blockName = str(block._name)
                resp_coordinates = model_coordinates(blockName, "")
                x = float(resp_coordinates.pose.position.x)
                y = float(resp_coordinates.pose.position.y)
                z = float(resp_coordinates.pose.position.z)

                distance = pose_obj.GetEuclidianDistance(x, y, z)
                if (np.abs(block._pose.x - x) > self.eps):
                    block._pose.x = x
                    if distance < self.threshold:
                        ret_val = True
                if (np.abs(block._pose.y - y) > self.eps):
                    block._pose.y = y
                    if distance < self.threshold:
                        ret_val = True
                if (np.abs(block._pose.z - z) > self.eps):
                    block._pose.z = z
                    if distance < self.threshold:
                        ret_val = True
        except rospy.ServiceException as e:
            rospy.loginfo("Get Model State service call failed:  {0}".format(e))
        return ret_val