
import math


class Pose():

    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0, roll=0.0):
        self.x = x
        self.y = y
        self.z = z

        # Need to convert these
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

    def GetEuclidianDistance(self, x, y, z):
        return math.sqrt(((self.x - x) ** 2) +
                         ((self.y - y) ** 2) +
                         ((self.z - z) ** 2))

    def UpdateRobotPose(self, bot):
        if bot is None:
            return
        coord_tuple = bot.arm.pose_ee
        cartesian_coordinates = coord_tuple[0]
        euler_angles = coord_tuple[1]
        self.x = cartesian_coordinates[0]
        self.y = cartesian_coordinates[1]
        self.z = cartesian_coordinates[2]
        self.roll = euler_angles[0]
        self.pitch = euler_angles[1]
        self.yaw = euler_angles[2]

    def GetClosestDistance(self, pose_array):
        distances = []
        for pose in pose_array:
            distance = self.GetEuclidianDistance(pose.x, pose.y, pose.z)
            distances.append(distance)

        return min(distances), distances


    def PrintPose(self):
        print("X = {}".format(self.x))
        print("Y = {}".format(self.y))
        print("Z = {}".format(self.z))
        print("Yaw = {}".format(self.yaw))
        print("Pitch = {}".format(self.pitch))
        print("Roll = {}".format(self.roll))

    def DetectCollision(self, pose_array):
        min_distance, distances = self.GetClosestDistance(pose_array)
        if min_distance < 0.03:
            return True
        else:
            return False

    def DetectCollisionV2(self, x, y, z):
        distance = self.GetEuclidianDistance(x, y, z)
        print("Minimum Distance = {}".format(distance))
        if distance < 0.03:
            return True
        else:
            return False



