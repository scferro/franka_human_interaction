import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from franka_hri_interfaces.srv import VLAService
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

        # Load the pre-trained OctoModel
        self.get_logger().info('Loading pre-trained OctoModel')
        try:
            self.model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
            self.get_logger().info('OctoModel loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load OctoModel: {str(e)}')
            raise

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

        self.get_logger().info('VLA Service Node has been fully initialized')

    def vla_service_callback(self, request, response):
        """Callback for the VLA service."""
        self.get_logger().info(f'Received service request with text command: {request.text_command}')

        # Process main observation images
        observation_images_main = []
        for img_msg in request.observations_main:
            cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            resized_image = cv2.resize(cv_image, (256, 256))
            observation_images_main.append(resized_image)
            self.get_logger().info(f'Received {len(observation_images_main)} main images.')

        input_images_main = np.array(observation_images_main)
        input_images_main = input_images_main.reshape(1, len(observation_images_main), 256, 256, 3)

        # Process wrist observation images
        observation_images_wrist = []
        for img_msg in request.observations_wrist:
            cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            resized_image = cv2.resize(cv_image, (128, 128))
            observation_images_wrist.append(resized_image)
            self.get_logger().info(f'Received {len(observation_images_wrist)} wrist images.')

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
            action = actions[0][0]  # Get the first action from the first batch
            
            response.linear = Vector3(x=float(action[0]), y=float(action[1]), z=float(action[2]))
            response.angular = Vector3(x=float(action[3]), y=float(action[4]), z=float(action[5]))
            response.gripper = bool(action[6])
            
            self.get_logger().info(f"Returning action: "
                        f"linear: ({response.linear.x:.4f}, {response.linear.y:.4f}, {response.linear.z:.4f}), "
                        f"angular: ({response.angular.x:.4f}, {response.angular.y:.4f}, {response.angular.z:.4f}), "
                        f"gripper: {response.gripper}")
        except Exception as e:
            self.get_logger().error(f'Error in model inference: {str(e)}')
            response.linear = Vector3()
            response.angular = Vector3()
            response.gripper = False

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