import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

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

        # Add parameter for number of actions to return
        self.declare_parameter('num_actions', 4)
        self.num_actions = self.get_parameter('num_actions').value

        # Load the pre-trained OctoModel
        self.get_logger().info('Loading pre-trained OctoModel')
        try:
            self.model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
            self.get_logger().info('OctoModel loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load OctoModel: {str(e)}')
            raise

        # self.get_logger().info(str(self.model.config))

        # CV Bridge for converting between ROS and OpenCV images
        self.cv_bridge = CvBridge()

        # Create a callback group for the service
        self.cb_group = ReentrantCallbackGroup()

        # Service
        self.vla_service = self.create_service(
            VLAService,
            'vla_service',
            self.vla_service_callback,
            callback_group=self.cb_group
        )

        self.get_logger().info(f'VLA Service Node has been fully initialized (returning {self.num_actions} actions per request)')

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
                unnormalization_statistics=self.model.dataset_statistics["bridge_dataset"]["action"], 
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