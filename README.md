# franka_human_interaction

Author - Stephen Ferro (scferro, stephencferro@gmail.com)

A ROS 2 framework for human-robot collaborative block sorting using online learning from visual and gestural feedback.

## System Overview

This system enables adaptive robot-human collaboration through:
- Real-time block detection and tracking
- Neural network-based classification with online learning
- Visual and gestural feedback for human interaction
- Continuous adaptation to human preferences

The system achieves this by combining computer vision, machine learning, and precise robot control to create an interactive sorting experience that improves over time based on human guidance.

## Prerequisites

- ROS 2 
- MoveIt
- OpenCV
- PyTorch
- Intel RealSense SDK
- MediaPipe
- The Sensor Bridge package for interfacing with the ECG/IMU sensor (https://github.com/duyayun/Human-robot-interaction/tree/main)

## Installation

1. Create a ROS2 workspace if you haven't already.

2. Clone this repository into the source directory of your ROS 2 workspace.

3. Build the packages:
```bash
colcon build
```

4. Source the workspace:
```bash
source install/setup.bash
```

## Quickstart

This package can be used to control the Franka in the sorting task with gesture interactions. It can also load a GUI interface for testing, training, and evaluating the different models without the robot.

1. Launch the full block sorting system on the robot:
    
    **On the CRB Frankas**:

    - Connect to the Franka robot Ubuntu 22.04 Docker Container via SSH
    ```bash
    ssh -p 9222 student@station
    ```
    - Start MoveIt on the Franka with the following:
    ```bash
    ros2 launch franka_moveit_config moveit.launch.py use_rviz:=false robot_ip:=panda0.robot
    ```

    **On your Local Machine**:
    - Run the following to start all relevant node for reading gestures, interacting the the networks, and controlling the robot:
    ```bash
    ros2 launch franka_hri_bringup hri_sorting.launch.py
    ```

2. Launch training without robot (train using saved images and/or live gestures):
    - To train the sorting network using block images and the complex gesture network:
    ```bash
    ros2 launch franka_hri_sorting sorting.launch.py mode:=images training_mode:=both gesture_network_type:=complex
    ```

    - To train the sorting network only:
    ```bash
    ros2 launch franka_hri_sorting sorting.launch.py mode:=images training_mode:=sorting_only
    ```

    - To train the complex gesture network only:
    ```bash
    ros2 launch franka_hri_sorting sorting.launch.py mode:=images training_mode:=gestures_only gesture_network_type:=complex
    ```

    - To train the binary gesture network only:
    ```bash
    ros2 launch franka_hri_sorting sorting.launch.py mode:=images training_mode:=gestures_only gesture_network_type:=binary
    ```

    - You can also load a pre-trained model by adding the following to any command above: 
    ```bash
    complex_sorting_model_path:=<path to saved model>   # To load a saved sorting model
    complex_gesture_model_path:=<path to saved model>   # To load a saved complex gesture model
    binary_gesture_model_path:=<path to saved model>    # To load a saved binary gesture model
    ```

## Core Nodes and Packages

### franka_control Package

#### manipulate_blocks (C++)
The primary robot control node manages all aspects of the Franka Panda's operations. It handles motion planning, grasping operations, and block placement while maintaining safety through collision detection. The node integrates with neural networks for decision-making and processes human feedback for continuous learning. It coordinates movements between four distinct sorting piles and manages the overall sorting workflow.

### franka_hri_sorting Package

#### blocks (Python)
This vision processing node handles all aspects of block detection and tracking. It processes both RGB and depth data from the RealSense D405 camera to perform robust block segmentation. The node generates accurate 3D pose estimates for detected blocks and maintains a consistent tracking system. It publishes visualization markers for monitoring and provides essential scanning services.

#### network_node (Python)
The neural network management node coordinates four distinct networks:
- Complex sorting network for 4-class block classification
- Binary gesture network for simple yes/no feedback
- Complex gesture network for 4-class gestural input
- Training management for online learning
It handles all aspects of training data collection, normalization, and buffer management while providing prediction services and maintaining detailed performance logs.

#### human_input (Python)
The GUI interface node enables direct human interaction through a graphical interface. It provides both binary and categorical input options, facilitates model saving operations, and displays real-time system status. The node manages both simple and complex gesture feedback modes and allows for direct network corrections when needed.

#### human_interaction (Python)
This node monitors human interventions using the RealSense D435i camera. It tracks block movements between piles, updates internal state representations, and coordinates network corrections. The node provides visual feedback of monitoring regions ensuring smooth integration of human actions into the system workflow.

#### network_training (Python)
A dedicated training node that manages interactive learning sessions. It handles image display during training, data collection sequences, and supports multiple training modes.

### Launch Files

#### sorting.launch.py (franka_hri_sorting Package)
This launch file configures the system for training without robot control. It is used to launch all nodes required for block sorting, either in training mode or with the robot. However, it does not launch any robot control nodes. It manages training modes, data logging, and model saving while enabling offline training with saved images. This is the recommended starting point for initial system training.

#### hri_sorting.launch.py (franka_hri_bringup Package)
The main launch file for robot operation, this file initializes the complete system including robot control, both RealSense cameras, all neural networks, and human interaction monitoring. It sets up the full sorting system with feedback capabilities and manages all necessary transformations and configurations.

## Usage Guide

The system provides two main operational modes - interactive robot sorting and network pre-training. Understanding these modes and how to effectively use them is key to getting the most out of the system.

### Visual Pre-training with Robot

This method leverages the robot's camera system to learn from physically arranged blocks. Start by manually organizing blocks into your desired category groups within the robot's workspace (see demo video for example). Then activate the pre-training service using ` ros2 service call /pretrain_franka std_srvs/srv/Empty {}`, and the system will:

- Scan the entire workspace
- Record block positions and categories
- Use this information to train the neural networks

This method is particularly effective for establishing initial sorting preferences quickly when using a new set of blocks or a new sorting pattern.

### Robot Sorting Mode

This is the primary operating mode where the robot actively sorts blocks while learning from human feedback. The setup requires coordinating between the robot control station and your local development machine.

With both the robot control and sorting system running, you can begin the sorting sequence by sending an action goal to the sort_blocks action server using `ros2 action send_goal /sort_blocks franka_hri_interfaces/action/EmptyAction "{}"`. This will trigger the robot to execute the block sorting routine with new blocks in the workspace.

## More Information
For more information on this project, please see the repo and project portfolio post linked below.

[Portfolio Post](https://scferro.github.io/projects/05-ml-robot-sorting)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
