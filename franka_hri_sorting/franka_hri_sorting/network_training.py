import os
import random
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8
from sensor_msgs.msg import Image
from franka_hri_interfaces.srv import SortNet, GestNet
from cv_bridge import CvBridge, CvBridgeError
import time
import threading

class NetworkTrainingNode(Node):
    def __init__(self):
        super().__init__('network_training_node')

        # Initialize parameters
        self.declare_parameter('training_images_path', '/home/scferro/Documents/final_project/training_images')
        self.declare_parameter('display_time', 2.0)  # Time to display image in seconds
        self.declare_parameter('prediction_timeout', 5.0)  # Maximum time to wait for prediction
        
        self.training_path = self.get_parameter('training_images_path').value
        self.display_time = self.get_parameter('display_time').value
        self.prediction_timeout = self.get_parameter('prediction_timeout').value
        
        if not self.training_path:
            raise ValueError("Training images path parameter must be set")

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Create clients for network services
        self.train_sorting_client = self.create_client(SortNet, 'train_sorting')
        self.get_sorting_prediction_client = self.create_client(SortNet, 'get_sorting_prediction')
        self.train_gesture_client = self.create_client(GestNet, 'train_gesture')
        self.get_gesture_prediction_client = self.create_client(GestNet, 'get_gesture_prediction')

        # Subscribe to human input
        self.human_input_sub = self.create_subscription(
            Int8,
            'human_sorting',
            self.human_input_callback,
            10
        )

        # Initialize state variables
        self.current_ros_msg = None
        self.current_image = None
        self.waiting_for_input = False
        self.current_sorting_prediction = None
        self.current_gesture_prediction = None
        
        # Initialize display thread
        self.display_thread = threading.Thread(target=self.display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()

    def display_loop(self):
        """Separate thread for image display."""
        cv2.namedWindow('Training Image', cv2.WINDOW_NORMAL)
        while True:
            if self.current_image is not None:
                cv2.imshow('Training Image', self.current_image)
            cv2.waitKey(1)
            time.sleep(0.01)

    def get_gesture_prediction(self):
        """Get prediction from the gesture network."""
        request = GestNet.Request()
        
        try:
            if not self.get_gesture_prediction_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn('Gesture prediction service not available')
                return None

            future = self.get_gesture_prediction_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            
            if future.result() is not None:
                self.current_gesture_prediction = future.result().prediction
                self.get_logger().info(f'Gesture prediction: {self.current_gesture_prediction}')
                
                # Use gesture prediction as initial sorting prediction
                self.current_sorting_prediction = 1 if self.current_gesture_prediction > 0.5 else 0
                self.waiting_for_input = True
                self.get_logger().info(f'Initial sorting based on gesture: {self.current_sorting_prediction}')
                self.get_logger().info('Waiting for human input (Y/N)...')
            else:
                self.get_logger().error('Failed to get gesture prediction')
                
        except Exception as e:
            self.get_logger().error(f'Error getting gesture prediction: {str(e)}')

    def human_input_callback(self, msg):
        """Handle human input for training both networks."""
        if not self.waiting_for_input:
            return

        key = msg.data
        if self.current_sorting_prediction is not None:
            if key == 1:  # Y input - prediction was correct
                self.train_sorting_network(self.current_sorting_prediction)
                self.train_gesture_network(self.current_sorting_prediction)
            elif key == 0:  # N input - prediction was wrong
                opposite_label = 1 if self.current_sorting_prediction == 0 else 0
                self.train_sorting_network(opposite_label)
                self.train_gesture_network(opposite_label)

        self.current_image = None
        self.waiting_for_input = False

    def train_gesture_network(self, label):
        """Train the gesture network."""
        request = GestNet.Request()
        request.label = label
        
        try:
            future = self.train_gesture_client.call_async(request)
            self.get_logger().info(f'Training gesture network with label {label}')
        except Exception as e:
            self.get_logger().error(f'Failed to train gesture network: {str(e)}')

    def train_sorting_network(self, label):
        """Train the sorting network."""
        if self.current_ros_msg is None:
            return
            
        request = SortNet.Request()
        request.image = self.current_ros_msg
        request.label = label
        
        try:
            future = self.train_sorting_client.call_async(request)
            self.get_logger().info(f'Training sorting network with label {label}')
        except Exception as e:
            self.get_logger().error(f'Failed to train sorting network: {str(e)}')

    def run_training(self):
        """Main training loop."""
        while rclpy.ok():
            if self.waiting_for_input:
                rclpy.spin_once(self, timeout_sec=0.1)
                continue

            # Load new random image
            image = self.load_random_image()
            if image is None:
                continue

            # Display image
            self.current_image = image
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.current_ros_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
            except CvBridgeError as e:
                self.get_logger().error(f'CV Bridge error: {str(e)}')
                continue

            # Get gesture prediction (the network node handles the sensor data internally)
            if not self.get_gesture_prediction():
                self.get_logger().warn('Skipping image due to prediction failure')
                continue

            time.sleep(self.display_time)

    def load_random_image(self):
        """Load a random image from the training folder."""
        try:
            files = os.listdir(self.training_path)
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            
            if not image_files:
                self.get_logger().error("No image files found in the specified folder")
                return None

            random_image = random.choice(image_files)
            image_path = os.path.join(self.training_path, random_image)
            
            image = cv2.imread(image_path)
            if image is None:
                self.get_logger().error(f"Failed to load image: {image_path}")
                return None
                
            return image

        except Exception as e:
            self.get_logger().error(f'Error loading image: {str(e)}')
            return None

def main(args=None):
    rclpy.init(args=args)
    node = NetworkTrainingNode()
    try:
        node.run_training()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()