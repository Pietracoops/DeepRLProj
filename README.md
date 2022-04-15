# DeepRLProj

## Install WSL2 on Windows 10:
## Installing WSL2:
1. Open **Start**
2. Search for **Command Prompt**, right-click and select **Run as administrator**  
3. Type the following command to install the WSL on Windows 10
```
wsl --install
```
4. Restart your computer to finish the WSL installation

## Update WSL kernel
1. Open **Start**  
2. Search for **Command Prompt**, right-click and select **Run as administrator**  
3. Type the following command to update the WSL kernel
```
wsl --update
```
If the update command doesn’t work, open **Settings > Update & Security > Windows Update > Advanced options**, and turn on the **“Receive updates for other Microsoft products when you update Windows”** toggle switch.

## Enable Windows Subsystem for Linux
1. Open **Start**  
2. Search for **Turn Windows features on or off**  
3. Check the **Windows Subsystem for Linux** option  
4. Restart your computer if needed

## Enable Virtual Machine Platform
1. Open **Start**
2. Search for **PowerShell**, right-click and select **Run as administrator**
3. Type the following command to enable the Virtual Machine Platform feature
```
Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
```

## Enable Windows Subsystem for Linux 2
1. [Download this WSL 2 kernel update](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)
2. Run **wsl_update_x64.msi**
3. Open **Start**
4. Search for **PowerShell**, right-click and select **Run as administrator**
5. Type the following command to set **Windows Subsystem for Linux 2** your default architecture
```
wsl --set-default-version 2
```

## Confirm distro platform
1. Open **Start**
2. Search for **PowerShell**, right-click and select **Run as administrator**
3. Type the following command to  verify the version of the distro
```
wsl -l -v
```
4. Confirm the distro version is **2**

# Install ROS Noetic

1. Navigate to your home directory
```
cd
```
2. Setup your sources.list
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```
3. Setup your keys
```
sudo apt install curl
```
```
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
```
4. Update Debian package index
```
sudo apt update
```
5. Install ROS Desktop Full
```
sudo apt install ros-noetic-desktop-full
```
6. Environment setup
```
source /opt/ros/noetic/setup.bash
```
7. Automatic Environment setup
```
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
```
```
source ~/.bashrc
```
8. Dependencies for building packages
```
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```
9. Initialize rosdep
```
sudo apt install python3-rosdep
```
```
sudo rosdep init
```
```
rosdep update
```

# Setting up LoCoBot in Gazebo
1. Clone the Interbotix Repo
```
git clone https://github.com/Interbotix/interbotix_ros_rovers.git
```
2. Navigate to install directory
```
cd interbotix_ros_rovers/interbotix_ros_xslocobots/install/amd64
```
3. Run install script
```
chmod +x xslocobot_amd64_install.sh
```
```
./xslocobot_amd64_install.sh
```
# Setup XLaunch
1. Download and Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/)
2. Run XLaunch
3. Select Multiple Windows
4. Display number: 0
5. Click Next
6. Select Start no client
7. Click Next
8. **Disable** Native opengl
9. **Enable** Disable access control
10. Click Next
11. Finish

In your WSL2 environment, run the following Commands
```
export GAZEBO_IP=127.0.0.1
```
```
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0 
```
```
export LIBGL_ALWAYS_INDIRECT=0
```

# Catkin Make
Navigate to the folder /home/user/interbotix_ws and run the following command

```
catkin_make
```

# Run ROS/Gazebo
Launch ROS/Gazebo
```
roslaunch interbotix_xslocobot_gazebo xslocobot_gazebo.launch robot_model:=locobot_px100 show_lidar:=true use_position_controllers:=true dof:=4
```
**OPEN NEW TERMINAL**  
Unpause physics engine
```
rosservice call /gazebo/unpause_physics
```

# Removing ROS  
Uninstalling ROS if you wish to restart
```
sudo apt-get clean
sudo apt-get autoremove
sudo apt-get remove ros-*
sudo apt-get update
```

# Massimo's Ubuntu installation for the actual robot manipulation
First you will need to run the following commands to install the robotics libaries

```
sudo apt install curl
curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh
chmod +x xsarm_amd64_install.sh
./xsarm_amd64_install.sh
```
This script will create 3 folders in your home directory. There will be a cmake error in the realsense package, which will force us to make some modifications. The error in the cmake has to do with the c++ function finds_if. You will need to open up the realsense_ws folder in your home and search for this function and add a "std::" namespace to the instances that do not have it. after that you will need to purge the other two folders and modify the xsarm_amd64_install.sh script with the following:

```
  # Step 2B: Install realsense2 ROS Wrapper
 REALSENSE_WS=~/realsense_ws
#  if [ ! -d "$REALSENSE_WS/src" ]; then
  if [ -d "$REALSENSE_WS/src" ]; then
    echo "Installing RealSense ROS Wrapper..."
    # mkdir -p $REALSENSE_WS/src
    cd $REALSENSE_WS/src
    # git clone https://github.com/IntelRealSense/realsense-ros.git
    cd realsense-ros/
    # git checkout 2.3.1
    cd $REALSENSE_WS
    catkin_make clean
    catkin_make -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release
    catkin_make install
    echo "source $REALSENSE_WS/devel/setup.bash" >> ~/.bashrc
  else
    echo "RealSense ROS Wrapper already installed!"
  fi
  source $REALSENSE_WS/devel/setup.bash
```

We will also remove the last couple of lines that were inserted into our bashrc by doing the following:

```
gedit ~/.bashrc
```
After you connect the arm you can run the following command to enable or disable the actuators.
```
rosservice call /vx250/torque_enable "{cmd_type: 'group', name: 'all', enable: true}"
```

Then you can move the arm into a home position by doing the following
```
rostopic pub -1 /vx250/commands/joint_group interbotix_xs_msgs/JointGroupCommand "arm" "[0,0,0,0,0]"
```

## Python API For moving the arm
You can run the following script to try manipulating the viperx 250 arm

```
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np

# This script makes the end-effector perform pick, pour, and place tasks
#
# To get started, open a terminal and type 'roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250'
# Then change to this directory and type 'python bartender.py'

def main():
    bot = InterbotixManipulatorXS("vx250", "arm", "gripper")
    bot.arm.set_ee_pose_components(x=0.3, z=0.2)
    bot.arm.set_single_joint_position("waist", np.pi/2.0)
    bot.gripper.open()
    bot.arm.set_ee_cartesian_trajectory(x=0.1, z=-0.16)
    bot.gripper.close()
    bot.arm.set_ee_cartesian_trajectory(x=-0.1, z=0.16)
    bot.arm.set_single_joint_position("waist", -np.pi/2.0)
    bot.arm.set_ee_cartesian_trajectory(pitch=1.5)
    bot.arm.set_ee_cartesian_trajectory(pitch=-1.5)
    bot.arm.set_single_joint_position("waist", np.pi/2.0)
    bot.arm.set_ee_cartesian_trajectory(x=0.1, z=-0.16)
    bot.gripper.open()
    bot.arm.set_ee_cartesian_trajectory(x=-0.1, z=0.16)
    bot.arm.go_to_home_pose()
    bot.arm.go_to_sleep_pose()

if __name__=='__main__':
    main()

```

## Useful links with regards to viperx 250 arm
https://www.trossenrobotics.com/viperx-300-robot-arm.aspx#packages
https://github.com/Interbotix/interbotix_ros_manipulators
https://www.trossenrobotics.com/docs/interbotix_xsarms/ros_interface/software_setup.html
https://www.trossenrobotics.com/docs/interbotix_xsarms/specifications/vx250.html
https://www.trossenrobotics.com/docs/interbotix_xsarms/python_ros_interface/index.html
https://www.youtube.com/watch?v=KoqBEvz4GII&t=436s&ab_channel=TrossenRobotics
https://www.trossenrobotics.com/docs/interbotix_xsarms/ros_interface/quickstart.html

Lab tech
http://www-labs.iro.umontreal.ca/~lokbani/
