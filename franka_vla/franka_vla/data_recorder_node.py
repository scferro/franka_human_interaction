#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from franka_hri_interfaces.msg import EECommand
import cv2
from cv_bridge import CvBridge
import numpy as np
import threading
import termios
import sys
import tty
import json
import time
from datetime import datetime
import os
from pathlib import Path

class DataRecorderNode(Node):
    def __init__(self):
        super().__init__('data_recorder_node')
        
        # Parameters
        self.declare_parameter('frequency', 5.0)
        self.declare_parameter('base_path', '~/Documents/final_project/training_data_vla')
        
        self.frequency = self.get_parameter('frequency').value
        self.base_path = os.path.expanduser(self.get_parameter('base_path').value)
        
        # Create session directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.base_path, f'session_{timestamp}')
        self.main_image_dir = os.path.join(self.session_dir, 'main_images')
        self.wrist_image_dir = os.path.join(self.session_dir, 'wrist_images')
        
        # Create directories
        for dir_path in [self.session_dir, self.main_image_dir, self.wrist_image_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Create data file
        self.data_file = open(os.path.join(self.session_dir, 'data.jsonl'), 'w')
        
        # Create subscribers
        self.main_image_sub = self.create_subscription(
            Image,
            '/camera/d435i/color/image_raw',
            self.main_image_callback,
            10)
            
        self.wrist_image_sub = self.create_subscription(
            Image,
            '/camera/d405/color/image_rect_raw',
            self.wrist_image_callback,
            10)
            
        self.ee_command_sub = self.create_subscription(
            EECommand,
            '/ee_command',
            self.ee_command_callback,
            10)
        
        # Initialize latest data holders
        self.latest_main_image = None
        self.latest_wrist_image = None
        self.latest_ee_command = None
        self.frame_count = 0
        
        # Initialize recording flag
        self.is_recording = False
        
        # Get training prompt
        self.prompt = input("\nEnter the natural language prompt for this training session: ")
        
        # Start keypress detection thread
        self.keypress_thread = threading.Thread(target=self.detect_keypress)
        self.keypress_thread.daemon = True
        self.keypress_thread.start()
        
        # Start countdown thread
        self.countdown_thread = threading.Thread(target=self.countdown_and_start)
        self.countdown_thread.start()

    def countdown_and_start(self):
        self.get_logger().info(f'\nRecording will begin in:')
        for i in range(3, 0, -1):
            self.get_logger().info(f'{i}...')
            time.sleep(1)
        self.get_logger().info('Recording started!')
        self.get_logger().info('Press "q" to stop recording and save data')
        
        # Create timer for recording at specified frequency
        self.is_recording = True
        self.timer = self.create_timer(1.0/self.frequency, self.record_data)

    def main_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_main_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting main camera image: {str(e)}')

    def wrist_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_wrist_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting wrist camera image: {str(e)}')

    def ee_command_callback(self, msg):
        self.latest_ee_command = msg

    def record_data(self):
        if not self.is_recording:
            return
            
        if self.latest_main_image is None or self.latest_wrist_image is None:
            self.get_logger().warn('Waiting for images...')
            return
        
        # Save images
        main_image_path = os.path.join(self.main_image_dir, f'frame_{self.frame_count:06d}.png')
        wrist_image_path = os.path.join(self.wrist_image_dir, f'frame_{self.frame_count:06d}.png')
        
        cv2.imwrite(main_image_path, self.latest_main_image)
        cv2.imwrite(wrist_image_path, self.latest_wrist_image)
        
        # Create action vector (defaults to zeros if no command received)
        action_vector = [0.0] * 7  # 6 for movement + 1 for gripper
        if self.latest_ee_command is not None:
            action_vector[0] = self.latest_ee_command.linear.x
            action_vector[1] = self.latest_ee_command.linear.y
            action_vector[2] = self.latest_ee_command.linear.z
            action_vector[3] = self.latest_ee_command.angular.x
            action_vector[4] = self.latest_ee_command.angular.y
            action_vector[5] = self.latest_ee_command.angular.z
            action_vector[6] = 1.0 if self.latest_ee_command.gripper else 0.0
        
        # Create data entry
        data_entry = {
            'prompt': self.prompt,
            'main_image': os.path.relpath(main_image_path, self.session_dir),
            'wrist_image': os.path.relpath(wrist_image_path, self.session_dir),
            'action': action_vector,
            'timestamp': self.get_clock().now().to_msg().sec
        }
        
        # Write to JSONL file
        self.data_file.write(json.dumps(data_entry) + '\n')
        self.data_file.flush()  # Ensure data is written to disk
        
        self.frame_count += 1
        if self.frame_count % 100 == 0:
            self.get_logger().info(f'Recorded {self.frame_count} frames')

    def detect_keypress(self):
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            # Configure terminal for raw input
            tty.setraw(sys.stdin.fileno())
            
            while True:
                if sys.stdin.read(1) == 'q':
                    self.get_logger().info('Stop signal received')
                    self.stop_recording()
                    break
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def stop_recording(self):
        self.is_recording = False
        
        # Close data file
        self.data_file.close()
        
        # Print recording statistics
        self.get_logger().info(f'\nRecording completed:')
        self.get_logger().info(f'Total frames: {self.frame_count}')
        self.get_logger().info(f'Data saved to: {self.session_dir}')
        
        # Shutdown the node
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = DataRecorderNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()