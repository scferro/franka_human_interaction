# franka_human_interaction

The repo contains a set of ROS 2 Packages to control the Franka Emika Panda during two different human-robot interaction scenarios.

## Human-Robot Interactions with Vision-Language-Action Model

The goal of this section of the project is to develop a system for training the robot to complete multi step tasks based on human demonstration and instruction. This portion of the prject is still under development. 

## Block Sorting with Online Human Feedback

### Summary

This project aimed to create an intelligent sorting system using a Franka Panda robotic arm. The system’s primary goal was to autonomously detect, classify, and sort blocks into two distinct piles based on their visual characteristics. Computer vision techniques and machine learning were combined to process depth and color images, accurately identifying and locating blocks in the robot’s workspace. A neural network, capable of both pre-training and online learning through human feedback, made real-time decisions on how to categorize each block. The system’s adaptive learning capability allows it to improve its sorting accuracy over time, making it suitable for dynamic environments and changing sorting criteria.

### Quickstart

These packages were designed to work with the Franka Panda robots in the Center for Robotics and Biosystems at Northwestern University. Below are instructions for running the code on robots in the lab.

**Launch the Franka Sorting System:**

On the Franka station PC: Run `ros2 launch franka_fer_moveit_config moveit.launch.py use_rviz:=false robot_ip:=panda0.robot` to launch MoveIt on the robot

On you local PC: Run `ros2 launch franka_hri_sorting sorting.launch.py` to start the required nodes to interact with the robot

With everything running and some blocks placed randomly in front of the robot, execute the command `ros2 action send_goal /sort_blocks franka_hri_interfaces/action/EmptyAction "{}"` to start sorting the blocks. The robot will do an overhead scan then proceed to sort each block, checking for human feedback after each sorting and updating it's neural network based on the feedback. Feedback is provided through the console via the `human_input` node. 

The network can also be pre-trained to improve sorting performance. Pre-training using visual feedback is more straightforward. To do this, first sort the blocks into two groupings, one toward the left pile and one toward the right. Once sorted, call the pre-training service with `ros2 service call /pretrain_franka std_srvs/srv/Empty`. When this service is called, the robot will observe how the blocks are sorted (left vs. right) and train the network based on those classifications. 

Finally, a network can also be pre-trained by using the `network_training.py` script in the `franka_hri_sorting` package. This script simulates the live training process by displaying images of blocks and asking the user to sort them into one of two groups. Once the blocks are classified, the network is retrained in the same way that it would be with the real robot.
