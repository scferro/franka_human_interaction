import os
import enum
import random
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8, String
from sensor_msgs.msg import Image
from franka_hri_interfaces.srv import SortNet, GestNet
from cv_bridge import CvBridge, CvBridgeError
import time
import threading

class TrainingMode(enum.Enum):
    SORTING_ONLY = "sorting_only"
    GESTURES_ONLY = "gestures_only"
    BOTH = "both"

class NetworkTrainingNode(Node):
    def __init__(self):
        super().__init__('network_training_node')
        self._init_parameters()
        self._init_clients_and_publishers()
        self._init_state()
        self._init_display()

    def _init_parameters(self):
        """Initialize node parameters."""
        self.declare_parameter('training_mode', 'sorting_only')
        self.declare_parameter('training_images_path', '/home/scferro/Documents/final_project/training_images')
        self.declare_parameter('display_time', 2.0)
        self.declare_parameter('gesture_warning_time', 3.0)
        self.declare_parameter('prediction_timeout', 5.0)

        self.training_mode = TrainingMode(self.get_parameter('training_mode').value)
        self.training_path = self.get_parameter('training_images_path').value
        self.display_time = self.get_parameter('display_time').value
        self.gesture_warning_time = self.get_parameter('gesture_warning_time').value
        self.prediction_timeout = self.get_parameter('prediction_timeout').value

        if not self.training_path:
            raise ValueError("Training images path parameter must be set")

    def _init_clients_and_publishers(self):
        """Initialize service clients and publishers."""
        self.bridge = CvBridge()
        
        # Service clients
        self.train_sorting_client = self.create_client(SortNet, 'train_sorting')
        self.get_sorting_prediction_client = self.create_client(SortNet, 'get_sorting_prediction')
        self.train_gesture_client = self.create_client(GestNet, 'train_gesture')
        self.get_gesture_prediction_client = self.create_client(GestNet, 'get_gesture_prediction')

        # Publishers
        self.notification_pub = self.create_publisher(String, 'user_notifications', 10)

        # Subscribers
        self.human_input_sub = self.create_subscription(
            Int8,
            'human_sorting',
            self.human_input_callback,
            10
        )

    def _init_state(self):
        """Initialize state variables."""
        self.current_ros_msg = None
        self.current_image = None
        self.waiting_for_input = False
        self.waiting_for_gesture = False
        self.current_sorting_prediction = None
        self.current_gesture_prediction = None

    def _init_display(self):
        """Initialize display thread."""
        self.display_thread = threading.Thread(target=self.display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()

    def notify_user(self, message: str):
        """Publish user notification."""
        msg = String()
        msg.data = message
        self.notification_pub.publish(msg)
        self.get_logger().info(message)

    def display_loop(self):
        """Image display thread."""
        cv2.namedWindow('Training Image', cv2.WINDOW_NORMAL)
        while True:
            if self.current_image is not None:
                cv2.imshow('Training Image', self.current_image)
            cv2.waitKey(1)
            time.sleep(0.01)

    def get_gesture_prediction(self) -> bool:
        """Get prediction from gesture network."""
        request = GestNet.Request()
        
        try:
            if not self.get_gesture_prediction_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn('Gesture prediction service not available')
                return False

            # Notify user to prepare for gesture
            self.notify_user("Make your gesture now!")
            time.sleep(self.gesture_warning_time)
            self.notify_user("Getting gesture prediction...")

            future = self.get_gesture_prediction_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            
            if future.result() is not None:
                self.current_gesture_prediction = future.result().prediction
                self.get_logger().info(f'Gesture prediction: {self.current_gesture_prediction}')
                return True
            else:
                self.get_logger().error('Failed to get gesture prediction')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Error getting gesture prediction: {str(e)}')
            return False

    def get_sorting_prediction(self) -> bool:
        """Get prediction from sorting network."""
        if self.current_ros_msg is None:
            return False

        request = SortNet.Request()
        request.image = self.current_ros_msg
        
        try:
            future = self.get_sorting_prediction_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=self.prediction_timeout)
            
            if future.result() is not None:
                raw_prediction = future.result().prediction
                self.current_sorting_prediction = 1 if raw_prediction > 0.5 else 0
                self.get_logger().info(f'Sorting prediction: {self.current_sorting_prediction}')
                return True
            else:
                self.get_logger().error('Failed to get sorting prediction')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Error getting sorting prediction: {str(e)}')
            return False

    def human_input_callback(self, msg):
        """Handle human input based on current mode."""
        if not self.waiting_for_input:
            return

        key = msg.data
        if self.training_mode == TrainingMode.SORTING_ONLY:
            if self.current_sorting_prediction is not None:
                self._handle_sorting_feedback(key)
        
        elif self.training_mode == TrainingMode.GESTURES_ONLY:
            if self.current_gesture_prediction is not None:
                self._handle_gesture_feedback(key)
        
        elif self.training_mode == TrainingMode.BOTH:
            if self.current_sorting_prediction is not None:
                # For both modes, the gesture network trains on the final sorting decision
                label = self.current_sorting_prediction if key == 1 else (1 - self.current_sorting_prediction)
                self._handle_sorting_feedback(key)
                self.train_gesture_network(label)

        self.current_image = None
        self.waiting_for_input = False

    def _handle_sorting_feedback(self, key):
        """Handle feedback for sorting network."""
        if key == 1:  # Correct prediction
            self.train_sorting_network(self.current_sorting_prediction)
        elif key == 0:  # Incorrect prediction
            self.train_sorting_network(1 - self.current_sorting_prediction)

    def _handle_gesture_feedback(self, key):
        """Handle feedback for gesture network."""
        binary_prediction = 1 if self.current_gesture_prediction > 0.5 else 0
        if key == 1:  # Correct prediction
            self.train_gesture_network(binary_prediction)
        elif key == 0:  # Incorrect prediction
            self.train_gesture_network(1 - binary_prediction)

    def train_gesture_network(self, label):
        """Train the gesture network."""
        request = GestNet.Request()
        request.label = label
        
        try:
            # No need to send sequence data - the network node will use its current sequence
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

    def process_iteration(self):
        """Process one training iteration based on current mode."""
        # Load and prepare image
        image = self.load_random_image()
        if image is None:
            return

        self.current_image = image
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.current_ros_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {str(e)}')
            return

        # Handle different modes
        if self.training_mode == TrainingMode.SORTING_ONLY:
            if self.get_sorting_prediction():
                self.waiting_for_input = True
                self.notify_user(f'Is the sorting prediction {self.current_sorting_prediction} correct? (Y/N)')

        elif self.training_mode == TrainingMode.GESTURES_ONLY:
            if self.get_gesture_prediction():
                self.waiting_for_input = True
                prediction = 1 if self.current_gesture_prediction > 0.5 else 0
                self.notify_user(f'Is the gesture prediction {prediction} correct? (Y/N)')

        elif self.training_mode == TrainingMode.BOTH:
            if self.get_gesture_prediction() and self.get_sorting_prediction():
                self.waiting_for_input = True
                self.notify_user(f'Is the sorting prediction {self.current_sorting_prediction} correct? (Y/N)')

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

    def run_training(self):
        """Main training loop."""
        self.get_logger().info(f'Starting training in {self.training_mode.value} mode')
        
        while rclpy.ok():
            if self.waiting_for_input:
                rclpy.spin_once(self, timeout_sec=0.1)
                continue
            
            self.process_iteration()

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