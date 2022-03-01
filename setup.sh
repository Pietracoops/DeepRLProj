#!/bin/sh

# SETUP SCRIPT FOR ROBOSUITE

# 1 - Install required packages

tput setaf 2; echo "============= INSTALLING PACKAGES ============="
tput setaf 7;
sudo apt install curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
         libosmesa6-dev software-properties-common net-tools unzip vim \
         virtualenv wget xpra xserver-xorg-dev libglfw3-dev patchelf


tput setaf 1; echo "Please make sure that mujoco 210 is set up and the following environment variables are set:"

echo "LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:"


tput setaf 2; echo "Installing Brew..."
tput setaf 7;

brew install gcc@7 # make sure homebrew is installed

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
tput setaf 2; echo "Cloning Robosuite Repo"
tput setaf 7;

git clone https://github.com/ARISE-Initiative/robosuite.git

tput setaf 2; echo "Creating Conda Environment"
tput setaf 7;

conda create -n robosuite python=3.7 # make sure Anaconda is installed
conda activate robosuite
conda init robosuite
cd robosuite # go to the robosuite root folder

tput setaf 2; echo "Pip installing"
tput setaf 7;

pip install -e .
CC=gcc-7 python -c "import robosuite"  # this will trigger mujoco_py to compile
pip install robosuite


tput setaf 2; echo "You can now launch the demo by running the following command 'python -m robosuite.demos.demo_random_action' in conda environment 'robosuite'"
tput setaf 7;

