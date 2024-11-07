import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import glob
import os
import json
import pprint

from franka_hri_interfaces.srv import VLAService
from franka_hri_interfaces.msg import EECommand
from geometry_msgs.msg import Vector3

import numpy as np
import cv2
from cv_bridge import CvBridge
import jax

from octo.model.octo_model import OctoModel

class VLANode(Node):
    def __init__(self):
        super().__init__('vla_node')
        self.get_logger().info('Initializing VLA Service Node')

        # Add parameters
        self.declare_parameter('num_actions', 4)
        self.declare_parameter('checkpoint_dir', '/home/scferro/Documents/final_project/checkpoints_2')
        self.declare_parameter('checkpoint_step', -1)
        
        self.num_actions = self.get_parameter('num_actions').value
        checkpoint_dir = self.get_parameter('checkpoint_dir').value
        checkpoint_step = self.get_parameter('checkpoint_step').value

        # Load the checkpoint and statistics
        self.get_logger().info('Loading Octo checkpoint and statistics')
        try:
            if not checkpoint_dir:
                raise ValueError("checkpoint_dir parameter must be set")
            
            # Load action statistics first
            action_stats_path = os.path.join(checkpoint_dir, 'action_stats.json')
            if not os.path.exists(action_stats_path):
                raise ValueError(f"Action statistics not found at {action_stats_path}")
            
            self.get_logger().info(f'Loading action statistics from {action_stats_path}')
            with open(action_stats_path, 'r') as f:
                action_stats = json.load(f)
                
            # Convert to numpy arrays and construct the dataset_statistics structure
            self.dataset_statistics = {
                "bridge_dataset": {
                    "action": {
                        "mean": np.array(action_stats["mean"], dtype=np.float32),
                        "std": np.array(action_stats["std"], dtype=np.float32),
                        "mask": np.ones(len(action_stats["mean"]), dtype=bool)  # All dimensions are valid
                    }
                }
            }
            
            # Print statistics information
            self.log_statistics_info()
            
            def is_valid_checkpoint(step_dir):
                """Check if directory contains a valid checkpoint."""
                default_dir = os.path.join(checkpoint_dir, step_dir, 'default')
                if not os.path.isdir(default_dir):
                    return False
                if not os.path.exists(os.path.join(default_dir, 'checkpoint')):
                    return False
                required_files = ['config.json']
                for file in required_files:
                    if not os.path.exists(os.path.join(checkpoint_dir, file)):
                        return False
                return True
            
            # Get all potential checkpoint directories (numbered directories)
            checkpoints = [d for d in os.listdir(checkpoint_dir) 
                         if d.isdigit() and is_valid_checkpoint(d)]
            
            if not checkpoints:
                raise ValueError(f"No valid checkpoints found in {checkpoint_dir}")
                
            if checkpoint_step < 0:
                # Find latest valid checkpoint
                latest_step = max(int(d) for d in checkpoints)
                checkpoint_path = os.path.join(checkpoint_dir, str(latest_step), 'default')
                self.get_logger().info(f'Using latest checkpoint: step {latest_step}')
            else:
                # Use specific checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, str(checkpoint_step), 'default')
                if not os.path.exists(checkpoint_path) or not is_valid_checkpoint(str(checkpoint_step)):
                    raise ValueError(f"Valid checkpoint not found for step {checkpoint_step}")
                self.get_logger().info(f'Using checkpoint step {checkpoint_step}')

            # Initialize the model with the root config.json
            config_path = os.path.join(checkpoint_dir, 'config.json')
            self.get_logger().info(f'Loading config from: {config_path}')
            
            # Load the model with the checkpoint directory as the base path
            self.model = OctoModel.load_pretrained(checkpoint_dir)
            self.get_logger().info('Checkpoint loaded successfully')

        except Exception as e:
            self.get_logger().error(f'Failed to load checkpoint: {str(e)}')
            raise

        # Initialize remaining components
        self.cv_bridge = CvBridge()
        self.cb_group = ReentrantCallbackGroup()
        self.vla_service = self.create_service(
            VLAService,
            'vla_service',
            self.vla_service_callback,
            callback_group=self.cb_group
        )

        self.get_logger().info(f'VLA Service Node has been fully initialized (returning {self.num_actions} actions per request)')

    def log_statistics_info(self):
        """Log detailed information about loaded statistics."""
        try:
            self.get_logger().info("=== Dataset Statistics Information ===")
            
            # Check if bridge_dataset exists
            if 'bridge_dataset' not in self.dataset_statistics:
                self.get_logger().error("Missing 'bridge_dataset' in statistics!")
                return
            
            bridge_stats = self.dataset_statistics['bridge_dataset']
            self.get_logger().info("Found bridge_dataset statistics")
            
            # Check if action statistics exist
            if 'action' not in bridge_stats:
                self.get_logger().error("Missing 'action' in bridge_dataset statistics!")
                return
            
            action_stats = bridge_stats['action']
            self.get_logger().info("Action statistics found")
            
            # Print action statistics details
            self.get_logger().info("\nAction Statistics Structure:")
            for key, value in action_stats.items():
                if isinstance(value, np.ndarray):
                    self.get_logger().info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    self.get_logger().info(f"    values: min={np.min(value):.4f}, max={np.max(value):.4f}, mean={np.mean(value):.4f}")
                else:
                    self.get_logger().info(f"  {key}: {type(value)}")
                    if isinstance(value, (list, tuple)):
                        self.get_logger().info(f"    length: {len(value)}")
                    try:
                        self.get_logger().info(f"    value: {value}")
                    except Exception:
                        self.get_logger().info("    (value too complex to display)")
            
            # Print available keys at each level
            self.get_logger().info("\nAvailable Keys:")
            self.get_logger().info(f"Top level: {list(self.dataset_statistics.keys())}")
            self.get_logger().info(f"bridge_dataset level: {list(bridge_stats.keys())}")
            self.get_logger().info(f"action level: {list(action_stats.keys())}")
            
            self.get_logger().info("=== End Statistics Information ===")
            
        except Exception as e:
            self.get_logger().error(f"Error while logging statistics: {str(e)}")

    def crop_and_resize_image(self, image, target_size):
        """
        Crop image to square and resize to target size.
        
        Args:
            image: Input image array
            target_size: Tuple of (width, height) for final image
            
        Returns:
            Resized square image array
        """
        height, width = image.shape[:2]
        
        # Calculate dimensions for center crop
        size = min(height, width)
        start_y = (height - size) // 2
        start_x = (width - size) // 2
        
        # Crop to square
        cropped = image[start_y:start_y + size, start_x:start_x + size]
        
        # Resize to target size
        resized = cv2.resize(cropped, target_size)
        return resized

    def vla_service_callback(self, request, response):
        """Callback for the VLA service."""
        # Process main observation images
        observation_images_main = []
        for img_msg in request.observations_main:
            cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            processed_image = self.crop_and_resize_image(cv_image, (256, 256))
            observation_images_main.append(processed_image)

        input_images_main = np.array(observation_images_main)
        input_images_main = input_images_main.reshape(1, len(observation_images_main), 256, 256, 3)

        # Process wrist observation images
        observation_images_wrist = []
        for img_msg in request.observations_wrist:
            cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            processed_image = self.crop_and_resize_image(cv_image, (128, 128))
            observation_images_wrist.append(processed_image)

        input_images_wrist = np.array(observation_images_wrist)
        input_images_wrist = input_images_wrist.reshape(1, len(observation_images_wrist), 128, 128, 3)

        observation = {
            'image_primary': input_images_main,
            'image_wrist': input_images_wrist,
            'timestep_pad_mask': np.ones((1, len(observation_images_main)), dtype=bool),
        }

        try:
            current_goal = self.model.create_tasks(texts=[request.text_command])
            actions = self.model.sample_actions(
                observation, 
                current_goal, 
                unnormalization_statistics=self.dataset_statistics["bridge_dataset"]["action"], 
                rng=jax.random.PRNGKey(0)
            )
            
            # Get the requested number of actions from the first batch
            actions_to_return = min(self.num_actions, len(actions[0]))
            response.actions = []
            
            for i in range(actions_to_return):
                action = actions[0][i]  # Get the i-th action from the first batch
                ee_command = EECommand()
                ee_command.linear = Vector3(x=float(action[0]), y=float(action[1]), z=float(action[2]))
                ee_command.angular = Vector3(x=float(action[3]), y=float(action[4]), z=float(action[5]))
                ee_command.gripper = bool(action[6])
                response.actions.append(ee_command)
            
            self.get_logger().info(f"Returning {len(response.actions)} actions")
            for i, cmd in enumerate(response.actions):
                self.get_logger().info(f"Action {i + 1}: "
                            f"linear: ({cmd.linear.x:.4f}, {cmd.linear.y:.4f}, {cmd.linear.z:.4f}), "
                            f"angular: ({cmd.angular.x:.4f}, {cmd.angular.y:.4f}, {cmd.angular.z:.4f}), "
                            f"gripper: {cmd.gripper}")
                            
        except Exception as e:
            self.get_logger().error(f'Error in model inference: {str(e)}')
            # Return an empty action list in case of error
            response.actions = []

        return response

def main(args=None):
    rclpy.init(args=args)
    vla_node = VLANode()
    
    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        vla_node.get_logger().error(f'Unexpected error: {str(e)}')
    finally:
        vla_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()