# Multimodal-AIF (MAIF)
Multimodal VAE Active Inference Controller* 

Preprint: Cristian Meo and Pablo Lanillos (2021). "Multimodal VAE Active Inference Controller". Arxiv. 

*This work is under review. 

*This repository includes the code for the MAIF torque controller code and the tests with the 7DOF Panda robot.*

## Requirements
- ROS (melodic)
- pytorch 1.7.0
- cv2
- seaborn 0.11.0

## Installation
Once the dependencies are installed, a catkin workspace has to be created. To do it:

Create a folder for your catkin_ws: $ mkdir -p your_catkin_ws/src
Move to the folder: $ cd your_catkin_ws/src
Clone the repository $ git clone https://github.com/Cmeo97/MAIF 
Move back to your_catkin_ws: $ cd ..
Build the workspace: $ catkin_make
Source: $ source devel/setup.bash

## Running the code
To run the controller:

- After building and sorcing the workspace you have to launch the simulation: $ roslaunch panda_simulation simulation_py.launch
The launch file launches a Gazebo simulation in pause, start the simulation pressing Gazebo play button. 
at this point, go to the controller folder: $ cd src/panda_simulation/panda_control_MAIF/src
You have to run the camera node, which subscribe images from Gazebo and publish them for the controller: $ python2.7 camera.py
then, in another terminal, run the controller: $ python MAIF_controller.py

- To run the Mental simulation, go to the controller folder: $ cd src/panda_simulation/panda_control_MAIF/src 
and run: $ python Brain_Simulation.py



