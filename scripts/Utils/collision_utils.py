
import math
import pose_utils
from gazebo_msgs.srv import GetModelState
import rospy

class Block:
    def __init__(self, name):
        self._name = name
        self._pose = pose_utils.Pose()

class Collision():
    def __init__(self):
        # World specific blocks
        self._blockListDict = {
        'block_a': Block('mass_cube_green'),
        'block_b': Block('mass_cube_green_clone'),
        'block_c': Block('mass_cube_pink'),
        'block_d': Block('mass_cube_pink_clone'),
        'block_e': Block('mass_cube_blue'),
        'block_f': Block('mass_cube_red'),
        'block_g': Block('mass_cube_red_clone'),
        'block_h': Block('mass_cube_dark_blue')
        }

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

    def SearchForCollision(self):
        ret_val = False
        try:
            model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            for block in self._blockListDict.itervalues():
                blockName = str(block._name)
                resp_coordinates = model_coordinates(blockName, "")
            if block._pose.x != float(resp_coordinates.pose.position.x):
                block._pose.x = float(resp_coordinates.pose.position.x)
                ret_val = True
            if block._pose.y != float(resp_coordinates.pose.position.y):
                block._pose.y = float(resp_coordinates.pose.position.y)
                ret_val = True
            if block._pose.z != float(resp_coordinates.pose.position.z):
                block._pose.z = float(resp_coordinates.pose.position.z)
                ret_val = True
        except rospy.ServiceException as e:
            rospy.loginfo("Get Model State service call failed:  {0}".format(e))
        return ret_val