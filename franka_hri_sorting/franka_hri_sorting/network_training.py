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

class GestureNetworkType(enum.Enum):
    """Enum for different gesture network types."""
    BINARY = "binary"    # Simple binary gesture network
    COMPLEX = "complex"  # Multi-class gesture network

class TrainingMode(enum.Enum):
    """Enum for different training modes."""
    SORTING_ONLY = "sorting_only"
    GESTURES_ONLY = "gestures_only" 
    BOTH = "both"

class NetworkTrainingNode(Node):
    """Node for training sorting and gesture networks."""
    
    def __init__(self):
        super().__init__('network_training_node')
        self._init_parameters()
        self._init_clients_and_publishers()
        self._init_state()
        self._init_display()

    def _init_parameters(self):
        """Initialize node parameters."""
        # Training mode parameters
        self.declare_parameter('training_mode', 'sorting_only')
        self.declare_parameter('gesture_network_type', 'binary')
        self.declare_parameter('training_images_path', '/home/user/training_images')
        self.declare_parameter('display_time', 2.0)
        self.declare_parameter('gesture_warning_time', 3.0)
        self.declare_parameter('prediction_timeout', 5.0)

        # Get parameter values
        self.training_mode = TrainingMode(self.get_parameter('training_mode').value)
        self.gesture_network_type = GestureNetworkType(
            self.get_parameter('gesture_network_type').value)
        self.training_path = self.get_parameter('training_images_path').value
        self.display_time = self.get_parameter('display_time').value
        self.gesture_warning_time = self.get_parameter('gesture_warning_time').value
        self.prediction_timeout = self.get_parameter('prediction_timeout').value

        # Validate modes and configurations
        if self.training_mode == TrainingMode.BOTH:
            self.gesture_network_type = GestureNetworkType.COMPLEX
            self.get_logger().info("Forcing complex gesture network for combined training mode")
        
        if self.training_mode != TrainingMode.GESTURES_ONLY and not self.training_path:
            raise ValueError("Training images path must be set for non-gesture-only modes")

    def _init_clients_and_publishers(self):
        """Initialize service clients and publishers."""
        self.bridge = CvBridge()
        
        # Service clients for sorting network (always complex)
        self.train_sorting_client = self.create_client(
            SortNet, 'train_sorting')
        self.get_sorting_prediction_client = self.create_client(
            SortNet, 'get_sorting_prediction')
        
        # Service clients for gesture networks
        if self.gesture_network_type == GestureNetworkType.BINARY:
            self.train_gesture_client = self.create_client(
                GestNet, 'train_gesture')
            self.get_gesture_prediction_client = self.create_client(
                GestNet, 'get_gesture_prediction')
        else:
            self.train_gesture_client = self.create_client(
                GestNet, 'train_complex_gesture')
            self.get_gesture_prediction_client = self.create_client(
                GestNet, 'get_complex_gesture_prediction')

        # Publishers and subscribers
        self.notification_pub = self.create_publisher(
            String, 'user_notifications', 10)
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
        self.display_window_created = False
        self.waiting_for_input = False
        self.waiting_for_gesture = False
        
        # Prediction tracking
        self.current_predictions = {
            'sorting': {'prediction': None, 'confidence': None, 'id': None},
            'gesture': {'prediction': None, 'confidence': None, 'id': None}
        }

    def _init_display(self):
        """Initialize display thread if needed."""
        if self.training_mode != TrainingMode.GESTURES_ONLY:
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
        while True:
            if self.training_mode != TrainingMode.GESTURES_ONLY:
                if not self.display_window_created:
                    cv2.namedWindow('Training Image', cv2.WINDOW_NORMAL)
                    self.display_window_created = True
                
                if self.current_image is not None:
                    cv2.imshow('Training Image', self.current_image)
                cv2.waitKey(1)
            time.sleep(0.01)

    def get_gesture_prediction(self) -> bool:
        """Get prediction from appropriate gesture network."""
        request = GestNet.Request()
        
        try:
            if not self.get_gesture_prediction_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn('Gesture prediction service not available')
                return False

            time.sleep(0.5)
            self.notify_user("Make your gesture now!")
            time.sleep(self.gesture_warning_time)
            self.notify_user("Getting gesture prediction...")

            future = self.get_gesture_prediction_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            
            if future.result() is not None:
                result = future.result()
                
                # Store all prediction information
                self.current_predictions['gesture'] = {
                    'prediction': result.prediction,
                    'confidence': result.confidence,
                    'id': result.prediction_id
                }
                
                if self.gesture_network_type == GestureNetworkType.COMPLEX:
                    # For complex network, prediction is already the class (0-3)
                    self.get_logger().info(
                        f'Gesture prediction: {result.prediction} '
                        f'(confidence: {result.confidence:.2f}, ID: {result.prediction_id})'
                    )
                else:
                    # For binary network, prediction is a probability
                    self.get_logger().info(
                        f'Binary gesture prediction: {result.prediction:.2f} '
                        f'(confidence: {result.confidence:.2f}, ID: {result.prediction_id})'
                    )
                return True
            else:
                self.get_logger().error('Failed to get gesture prediction')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Error getting gesture prediction: {str(e)}')
            return False

    def get_sorting_prediction(self) -> bool:
        """Get prediction from complex sorting network."""
        if self.current_ros_msg is None:
            return False

        request = SortNet.Request()
        request.image = self.current_ros_msg
        request.index = -1  # Default value when not working with indexed blocks
        
        try:
            future = self.get_sorting_prediction_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=self.prediction_timeout)
            
            if future.result() is not None:
                result = future.result()
                
                # Store all prediction information
                self.current_predictions['sorting'] = {
                    'prediction': int(result.prediction),
                    'confidence': result.confidence,
                    'id': result.prediction_id
                }
                
                self.get_logger().info(
                    f'Sorting prediction: {result.prediction} '
                    f'(confidence: {result.confidence:.2f}, ID: {result.prediction_id})'
                )
                return True
            else:
                self.get_logger().error('Failed to get sorting prediction')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Error getting sorting prediction: {str(e)}')
            return False
            return False

    def human_input_callback(self, msg):
        """Handle human input based on current mode."""
        if not self.waiting_for_input:
            self.get_logger().warn("Received input but not waiting for input")
            return

        self.get_logger().info("Received human input")
        key = msg.data
        
        # Validate input based on mode
        if self.training_mode == TrainingMode.GESTURES_ONLY:
            if self.gesture_network_type == GestureNetworkType.BINARY:
                if key not in [0, 1]:
                    self.notify_user("Error: Binary gesture training only accepts 0 or 1 as input")
                    return
            else:
                if not 0 <= key <= 3:
                    self.notify_user("Error: Complex gesture training requires input between 0 and 3")
                    return
        else:
            if not 0 <= key <= 3:
                self.notify_user("Error: Input must be between 0 and 3")
                return

        self.get_logger().info(f"Processing valid input: {key}")

        # Process valid input
        if self.training_mode == TrainingMode.SORTING_ONLY:
            self._train_sorting_network(key)
        elif self.training_mode == TrainingMode.GESTURES_ONLY:
            self._train_gesture_network(key)
        elif self.training_mode == TrainingMode.BOTH:
            self._train_both_networks(key)

        # Clear image if not in gesture-only mode
        if self.training_mode != TrainingMode.GESTURES_ONLY:
            self.get_logger().info("Clearing current image")
            self.current_image = None
        
        self.get_logger().info("Resetting waiting_for_input flag")
        self.waiting_for_input = False
        self.get_logger().info("Human input processing complete")

    def _train_sorting_network(self, label):
        """Train the sorting network with new label."""
        if self.current_ros_msg is None:
            self.get_logger().error("No image available for training")
            return

        request = SortNet.Request()
        request.image = self.current_ros_msg
        request.label = label

        # Use the confidence from the prediction if available
        if self.current_predictions['sorting']['confidence'] is not None:
            request.confidence = self.current_predictions['sorting']['confidence']
        else:
            request.confidence = 0.0

        try:
            self.get_logger().info(f"Sending training request with label {label}")
            future = self.train_sorting_client.call_async(request)
            
            # Wait for the training response
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            
            if future.result() is not None:
                self.get_logger().info("Training request completed successfully")
                
                # Clear state after successful training
                self.get_logger().info("Clearing current image")
                self.current_image = None
                
                self.get_logger().info("Resetting waiting_for_input flag")
                self.waiting_for_input = False
                
                self.get_logger().info("Ready for next iteration")
            else:
                self.get_logger().error("Training request failed")
                
        except Exception as e:
            self.get_logger().error(f"Error training sorting network: {str(e)}")

    def _train_gesture_network(self, label):
        """Train the gesture network with new label."""
        request = GestNet.Request()
        request.label = label

        # Use the confidence from the prediction if available
        if self.current_predictions['gesture']['confidence'] is not None:
            request.confidence = self.current_predictions['gesture']['confidence']
        else:
            request.confidence = 0.0
        
        try:
            self.get_logger().info(f"Sending gesture training request with label {label}")
            future = self.train_gesture_client.call_async(request)
                
        except Exception as e:
            self.get_logger().error(f"Error training gesture network: {str(e)}")

    def _train_both_networks(self, label):
        """Train both networks with the same label."""
        self._train_sorting_network(label)
        self._train_gesture_network(label)

    def process_iteration(self):
        """Process one training iteration based on current mode."""
        if self.waiting_for_input:
            time.sleep(0.1)
            return
            
        self.get_logger().info("Starting new iteration")
        
        # Load image for non-gesture-only modes
        if self.training_mode != TrainingMode.GESTURES_ONLY:
            self.get_logger().info("Loading random image")
            image = self.load_random_image()
            if image is None:
                self.get_logger().error("Failed to load random image")
                return

            self.current_image = image
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.current_ros_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
                self.get_logger().info("Successfully loaded and converted new image")
            except CvBridgeError as e:
                self.get_logger().error(f'CV Bridge error: {str(e)}')
                return

        # Handle different modes
        if self.training_mode == TrainingMode.SORTING_ONLY:
            self.get_logger().info("Getting sorting prediction")
            if self.get_sorting_prediction():
                self.waiting_for_input = True
                sort_pred = self.current_predictions['sorting']
                self.notify_user(
                    f"Predicted class: {sort_pred['prediction']}\n"
                    f"Confidence: {sort_pred['confidence']:.2f}\n"
                    f"Enter correct class (0-3)"
                )
                self.get_logger().info("Waiting for human input")
            else:
                self.get_logger().error("Failed to get sorting prediction")

        elif self.training_mode == TrainingMode.GESTURES_ONLY:
            self.get_logger().info("Processing gesture training iteration")
            if self.get_gesture_prediction():
                self.waiting_for_input = True
                gest_pred = self.current_predictions['gesture']
                
                if self.gesture_network_type == GestureNetworkType.COMPLEX:
                    self.notify_user(
                        f"Predicted class: {gest_pred['prediction']}\n"
                        f"Confidence: {gest_pred['confidence']:.2f}\n"
                        f"Enter correct class (0-3)"
                    )
                else:
                    pred_class = "1" if gest_pred['prediction'] >= 0.5 else "0"
                    self.notify_user(
                        f"Binary prediction: {pred_class}\n"
                        f"Confidence: {gest_pred['confidence']:.2f}\n"
                        f"Enter correct class (0 or 1)"
                    )
                self.get_logger().info("Waiting for human input")
            else:
                self.get_logger().error("Failed to get gesture prediction")

        elif self.training_mode == TrainingMode.BOTH:
            self.get_logger().info("Processing combined training iteration")
            if self.get_sorting_prediction() and self.get_gesture_prediction():
                self.waiting_for_input = True
                sort_pred = self.current_predictions['sorting']
                gest_pred = self.current_predictions['gesture']
                self.notify_user(
                    f"Sorting prediction: {sort_pred['prediction']}\n"
                    f"Sorting confidence: {sort_pred['confidence']:.2f}\n"
                    f"Gesture prediction: {gest_pred['prediction']}\n"
                    f"Gesture confidence: {gest_pred['confidence']:.2f}\n"
                    f"Enter correct class (0-3)"
                )
                self.get_logger().info("Waiting for human input")
            else:
                self.get_logger().error("Failed to get predictions")

        # Display delay for non-gesture-only modes
        if self.training_mode != TrainingMode.GESTURES_ONLY:
            time.sleep(self.display_time)

    def load_random_image(self):
        """Load a random image from the training folder."""
        try:
            files = os.listdir(self.training_path)
            image_files = [f for f in files if f.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            
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
        mode_desc = f'Training Mode: {self.training_mode.value}'
        if self.training_mode == TrainingMode.GESTURES_ONLY:
            mode_desc += f' ({self.gesture_network_type.value} network)'
        self.get_logger().info(f'Starting training - {mode_desc}')
        
        while rclpy.ok():
            try:
                if self.waiting_for_input:
                    self.get_logger().debug("Waiting for input - spinning once")
                    rclpy.spin_once(self, timeout_sec=0.1)
                else:
                    self.get_logger().info("Ready for next iteration")
                    self.process_iteration()
            except Exception as e:
                self.get_logger().error(f"Error in training loop: {str(e)}")
                break

        self.get_logger().info("Training loop ended")


def main(args=None):
    """Main function to initialize and run the node."""
    rclpy.init(args=args)
    try:
        node = NetworkTrainingNode()
        node.run_training()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error running training node: {str(e)}")
    finally:
        if hasattr(node, 'display_window_created') and node.display_window_created:
            cv2.destroyAllWindows()
        if hasattr(node, 'destroy_node'):
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()