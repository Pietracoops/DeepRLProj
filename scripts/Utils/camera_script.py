import cv2
import imageio
import numpy as np
import pyrealsense2 as rs
import os
import shutil

import math
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import time
import colorsys
from interbotix_common_modules import angle_manipulation as ang
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_perception_modules.armtag import InterbotixArmTagInterface
from interbotix_perception_modules.pointcloud import InterbotixPointCloudInterface


# Robot Limits

#Bottom Right
#[ 0.15551279  0.75576575  0.63610841  0.20206359]
#[-0.15220841  0.65458527 -0.74050707 -0.19777008]
#[-0.97603708  0.01833727  0.21683028  0.09542407]
#[ 0.          0.          0.          1.        ]


# Top Right
#[ 0.49375534  0.14331209 -0.8577105  -0.09975484]
#[-0.10532042  0.98892121  0.10460621  0.02127819]
#[ 0.86319944  0.03868456  0.50337882  0.70540003]
#[ 0.          0.          0.          1.        ]


# Top Left
#[-0.02225062 -0.69723524 -0.71649698 -0.27271747]
#[-0.016344    0.71683232 -0.69705401 -0.20032232]
#[ 0.99961882 -0.00379945 -0.02734558  0.65403481]
#[ 0.          0.          0.          1.        ]


# Bottom Right
#[ 0.42146348 -0.87544314  0.23657525  0.07543361]
#[ 0.87029423  0.46378844  0.16579578  0.15576541]
#[-0.25486565  0.13601321  0.957363    0.52791624]
#[ 0.          0.          0.          1.        ]

bottom_right_xyz = [0.19869789, -0.2275085, 0.18390031]
bottom_right_rpy = [0.00613592332229018, 1.1857671737670898, -0.8528933525085449]
top_right_xyz = [-0.58738417, 0.12623425, 0.36434747]
top_right_rpy = [-3.126252845051237, -0.7807962020211897, 2.9299033006005963]
bottom_left_xyz = [0.05933635, 0.22099422, 0.50608895]
bottom_left_rpy = [0.012271846644580364, 0.2393010854721069, 1.3084856271743774]
top_left_xyz = [-0.35688018, -0.32545723, 0.54629755]
top_left_rpy = [-3.1293208069452128, -1.2149128039651593, -2.402213903265544]
top_middle_xyz = [-5.81443681e-01, -9.72188069e-02, 3.49150857e-01]
top_middle_rpy = [-3.1400586727592206, -0.9142524917894088, -2.975922720628329]
bottom_middle_xyz = [0.0739942, 0.01697402, -0.02869087]
bottom_middle_rpy = [3.141592653589793, 1.0231650789552413, -2.9160974641614636]


def make_clean_folder(path_folder):
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    else:
        user_input = input("%s not empty. Overwrite? (y/n) : " % path_folder)
        if user_input.lower() == "y":
            shutil.rmtree(path_folder)
            os.makedirs(path_folder)
        else:
            exit()


def record_rgbd(store):
    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(
        rs.option.visual_preset, 3
    )  # Set high accuracy for depth sensor
    depth_scale = depth_sensor.get_depth_scale()

    clipping_distance_in_meters = 1
    clipping_distance = clipping_distance_in_meters / depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            raise RuntimeError("Could not acquire depth or color frames.")

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        grey_color = 153
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image)
        )  # Depth image is 1 channel, color is 3 channels
        bg_removed = np.where(
            (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
            grey_color,
            color_image,
        )

        color_image = color_image[..., ::-1]

        if store:
            make_clean_folder("data/realsense/")
            imageio.imwrite("data/realsense/depth.png", depth_image)
            imageio.imwrite("data/realsense/rgb.png", color_image)

    finally:
        pipeline.stop()

    return color_image, depth_image


def get_position():
    # Initialization
    bot = InterbotixManipulatorXS("vx250", "arm", "gripper", moving_time=5.0, accel_time=0.75, gripper_pressure=1.0)
    pcl = InterbotixPointCloudInterface()
    armtag = InterbotixArmTagInterface()
    while True:
        print(bot.arm.get_ee_pose_command())
        time.sleep(1)
        print("================================================================================")

def robo_magic():
    # Initialization
    bot = InterbotixManipulatorXS("vx250", "arm", "gripper", moving_time=5.0, accel_time=0.75, gripper_pressure=1.0)
    pcl = InterbotixPointCloudInterface()
    armtag = InterbotixArmTagInterface()

    print("Init complete")
    bot.arm.go_to_home_pose()
    time.sleep(0.5)

    bot.gripper.set_pressure(0.2)

    bot.arm.go_to_sleep_pose()

    # Z limit = 0.50    

    #bot.arm.set_ee_pose_components(x=0.3, y=0.0, z=0.55, roll=0.0, pitch=0.0,yaw=0.0)

    # print("ROLLING")
    # bot.arm.set_ee_pose_components(x=0.3, y=0.0, z=0.55, roll=0.3, pitch=0.0,yaw=0.0)
    # bot.arm.set_ee_pose_components(x=0.3, y=0.0, z=0.55, roll=0.0, pitch=0.0,yaw=0.0)

    # print("PTICHING")
    # bot.arm.set_ee_pose_components(pitch=0.5)
    # bot.arm.set_ee_pose_components(pitch=0.0)

    # print("YAWING")
    # bot.arm.set_ee_pose_components(x=0.3, y=0.0, z=0.55, roll=0.0, pitch=0.0,yaw=0.3)
    # bot.arm.set_ee_pose_components(x=0.3, y=0.0, z=0.55, roll=0.0, pitch=0.0,yaw=0.0)


    bot.arm.set_ee_cartesian_trajectory(x=0.3)

    #bot.arm.set_ee_pose_components(x=0.2, y=0.0, z=0.55, roll=0.0, pitch=0.0,yaw=0.00)
    
    bot.gripper.close()

    time.sleep(3)
    bot.gripper.open()



    # Set initial arm and gripper pose
    # bot.arm.set_ee_pose_components(x=0.3, y=0.2, z=0.55, roll=0.1, pitch=0.03,yaw=0.05)
    # bot.arm.set_ee_pose_components(x=bottom_right_xyz[0], y=bottom_right_xyz[1], z=bottom_right_xyz[2], roll=bottom_right_rpy[0],pitch=bottom_right_rpy[1],yaw=bottom_right_rpy[2])
    # time.sleep(0.5)
    # bot.arm.go_to_home_pose()
    # time.sleep(0.5)
    # bot.arm.set_ee_pose_components(x=top_right_xyz[0], y=top_right_xyz[1], z=top_right_xyz[2], roll=top_right_rpy[0],pitch=top_right_rpy[1],yaw=top_right_rpy[2])
    # time.sleep(0.5)
    # bot.arm.go_to_home_pose()
    # time.sleep(0.5)
    # bot.arm.set_ee_pose_components(x=bottom_left_xyz[0], y=bottom_left_xyz[1], z=bottom_left_xyz[2], roll=bottom_left_rpy[0],pitch=bottom_left_rpy[1],yaw=bottom_left_rpy[2])
    # time.sleep(0.5)
    # bot.arm.go_to_home_pose()
    # time.sleep(0.5)
    # bot.arm.set_ee_pose_components(x=top_left_xyz[0], y=top_left_xyz[1], z=top_left_xyz[2], roll=top_left_rpy[0],pitch=top_left_rpy[1],yaw=top_left_rpy[2])
    # time.sleep(0.5)
    # bot.arm.go_to_home_pose()
    # time.sleep(0.5)
    # bot.arm.set_ee_pose_components(x=top_middle_xyz[0], y=top_middle_xyz[1], z=top_middle_xyz[2], roll=top_middle_rpy[0],pitch=top_middle_rpy[1],yaw=top_middle_rpy[2])
    # time.sleep(0.5)
    # bot.arm.go_to_home_pose()
    # time.sleep(0.5)
    # bot.arm.set_ee_pose_components(x=bottom_middle_xyz[0], y=bottom_middle_xyz[1], z=bottom_middle_xyz[2], roll=bottom_middle_rpy[0],pitch=bottom_middle_rpy[1],yaw=bottom_middle_rpy[2])
    # time.sleep(0.5)
    # bot.arm.go_to_home_pose()
    # bot.gripper.open()

    # Get the ArmTag pose
    # armtag.find_ref_to_arm_base_transform()
  

    #bot.arm.set_ee_pose_components(x=0.3, z=0.55)
    #bot.arm.set_single_joint_position("waist", np.pi/2.0)
    #bot.gripper.open()
    #bot.arm.set_ee_cartesian_trajectory(x=0.1, z=-0.16)
    #bot.gripper.close()
    #bot.arm.set_ee_cartesian_trajectory(x=-0.1, z=0.16)
    #bot.arm.set_single_joint_position("waist", -np.pi/2.0)
    #bot.arm.set_ee_cartesian_trajectory(pitch=1.5)
    #bot.arm.set_ee_cartesian_trajectory(pitch=-1.5)
    #bot.arm.set_single_joint_position("waist", np.pi/2.0)
    #bot.arm.set_ee_cartesian_trajectory(x=0.1, z=-0.16)
    #bot.gripper.open()
    #bot.arm.set_ee_cartesian_trajectory(x=-0.1, z=0.16)
    #bot.arm.go_to_home_pose()
    #bot.arm.go_to_sleep_pose()

def get_rpy():
    bot = InterbotixManipulatorXS("vx250", "arm", "gripper", moving_time=3.0, accel_time=0.75, gripper_pressure=1.0)
    pos = bot.arm.get_ee_pose_command()
    rpy = ang.rotationMatrixToEulerAngles(pos[:3, :3])

    print("Positon Matrix: {}".format(pos))
    print("==========================================")
    print("RPY: {}".format(rpy))

if __name__ == "__main__":

    # Function to get RGB and Depth image
    #color_image, depth_image = record_rgbd(store=False)
    
    robo_magic()
    #get_position()
    #get_rpy()
