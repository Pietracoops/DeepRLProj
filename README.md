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
