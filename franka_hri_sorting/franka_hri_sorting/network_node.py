import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from franka_hri_interfaces.srv import SortNet, GestNet, SaveModel
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import torch
import numpy as np
from franka_hri_sorting.network import SortingNet, GestureNet
import torchvision.transforms as transforms
from cv_bridge import CvBridge, CvBridgeError
from collections import deque
import random

class NetworkNode(Node):
    def __init__(self):
        super().__init__('network_node')
        
        # Initialize parameters
        self.declare_parameter('buffer_size', 25)
        self.declare_parameter('sequence_length', 20)
        self.declare_parameter('sorting_model_path', '')
        self.declare_parameter('gesture_model_path', '')
        self.declare_parameter('save_directory', '/home/user/saved_models')
        self.buffer_size = self.get_parameter('buffer_size').value
        self.sequence_length = self.get_parameter('sequence_length').value
        sorting_model = self.get_parameter('sorting_model_path').value
        gesture_model = self.get_parameter('gesture_model_path').value
        self.save_dir = self.get_parameter('save_directory').value
        
        # Initialize networks
        self.sorting_net = SortingNet(model_path=sorting_model)
        self.gesture_net = GestureNet(sequence_length=self.sequence_length, 
                                    model_path=gesture_model)
        
        self.bridge = CvBridge()
        
        # Initialize all training buffers
        self.class_0_images = deque(maxlen=self.buffer_size)
        self.class_1_images = deque(maxlen=self.buffer_size)
        self.class_0_gestures = deque(maxlen=self.buffer_size)
        self.class_1_gestures = deque(maxlen=self.buffer_size)
        
        # Buffer for sensor data
        self.mod48_buffer = None
        self.combined_sequence = []
        
        # Create services
        self.create_service(SortNet, 'train_sorting', self.train_sorting_callback)
        self.create_service(SortNet, 'get_sorting_prediction', self.get_sorting_prediction_callback)
        self.create_service(GestNet, 'train_gesture', self.train_gesture_callback)
        self.create_service(GestNet, 'get_gesture_prediction', self.get_gesture_prediction_callback)
        # Add save network services
        self.save_sorting_service = self.create_service(
            SaveModel, 
            'save_sorting_network',
            self.save_sorting_callback
        )
        
        self.save_gesture_service = self.create_service(
            SaveModel, 
            'save_gesture_network',
            self.save_gesture_callback
        )
        
        # Create subscribers for modality data
        self.mod48_sub = self.create_subscription(
            Float32MultiArray,
            '/sensor_data/modality_48',
            self.mod48_callback,
            10
        )
        self.mod51_sub = self.create_subscription(
            Float32MultiArray,
            '/sensor_data/modality_51',
            self.mod51_callback,
            10
        )

    def process_sensor_data(self, sequence):
        """Process sensor data into correct format for network."""
        try:
            # Ensure sequence has the right length
            if len(sequence) > self.sequence_length:
                sequence = sequence[-self.sequence_length:]
            elif len(sequence) < self.sequence_length:
                padding = [np.zeros((4,)) for _ in range(self.sequence_length - len(sequence))]
                sequence = padding + sequence
                
            # Stack with correct dimensions: [channels, sequence_length]
            data = np.stack(sequence, axis=0).T  # Transpose to get channels first
            
            # Convert to tensor and add batch dimension
            tensor_data = torch.FloatTensor(data).unsqueeze(0)
            
            return tensor_data
                
        except Exception as e:
            self.get_logger().error(f'Error processing sensor data: {str(e)}')
            return None
        
    def mod48_callback(self, msg):
        """Handle incoming modality 48 data."""
        try:
            data = np.array(msg.data).reshape(msg.layout.dim[0].size, msg.layout.dim[1].size)
            self.mod48_buffer = data[:, 1:]  # Remove timestamp, keep 3 feature columns
        except Exception as e:
            self.get_logger().error(f'Error in mod48_callback: {str(e)}')
        
    def mod51_callback(self, msg):
        """Handle incoming modality 51 data and combine with mod48 data."""
        try:
            if self.mod48_buffer is None:
                return

            # Extract mod51 data and reshape to get rows of [timestamp, value]
            data = np.array(msg.data)
            if len(data) > 100:  # If we have more than 50 pairs
                data = data[-100:]  # Take last 100 values (50 pairs)
            rows = 50
            cols = 2
            mod51_data = data.reshape(rows, cols)
            mod51_features = mod51_data[:, 1:]  # Just keep the values, not timestamps
            
            
            # Combine features horizontally to get 4 columns
            combined_features = np.hstack((self.mod48_buffer, mod51_features))  # Shape (50, 4)
            
            # Take just the first row as our current data point
            current_point = combined_features[0]  # Shape (4,)
            
            # Add to sequence
            self.combined_sequence.append(current_point)

            # Keep only most recent sequence_length points
            if len(self.combined_sequence) > self.sequence_length:
                self.combined_sequence = self.combined_sequence[-self.sequence_length:]

        except Exception as e:
            self.get_logger().error(f'Error processing modality 51 data: {str(e)}')

    def normalize_sequence(self, tensor_data):
        """Normalize the full sequence."""
        try:
            # Normalize across entire sequence
            mean = tensor_data.mean()
            std = tensor_data.std() + 1e-8
            normalized = (tensor_data - mean) / std
            return normalized
        except Exception as e:
            self.get_logger().error(f'Error normalizing sequence: {str(e)}')
            return tensor_data

    def get_gesture_prediction_callback(self, request, response):
        """Handle prediction requests for the gesture network."""
        try:
            if len(self.combined_sequence) < self.sequence_length:
                response.prediction = -1.0
                return response

            sequence_tensor = self.process_sensor_data(self.combined_sequence)
            if sequence_tensor is None:
                response.prediction = -1.0
                return response

            # Normalize the full sequence before prediction
            normalized_sequence = self.normalize_sequence(sequence_tensor)
            
            with torch.no_grad():
                prediction = self.gesture_net(normalized_sequence)
            
            response.prediction = float(prediction.detach().numpy())
            self.get_logger().info(f'Gesture prediction: {response.prediction}')
            return response
            
        except Exception as e:
            self.get_logger().error(f'Error getting gesture prediction: {str(e)}')
            response.prediction = -1.0
            return response

    def train_gesture_callback(self, request, response):
        """Handle training requests for the gesture network."""
        try:
            if len(self.combined_sequence) < self.sequence_length:
                self.get_logger().warn('Not enough data points for training')
                response.prediction = -1.0
                return response

            # Process current sequence
            sequence_tensor = self.process_sensor_data(self.combined_sequence)
            if sequence_tensor is None:
                response.prediction = -1.0
                return response

            # Normalize before adding to buffer
            normalized_sequence = self.normalize_sequence(sequence_tensor)

            # Add to appropriate buffer
            if request.label == 0:
                self.class_0_gestures.append(normalized_sequence)
            else:
                self.class_1_gestures.append(normalized_sequence)

            # Get balanced training data
            sequences, labels = self._balance_data(self.class_0_gestures, self.class_1_gestures)

            if sequences:
                self.gesture_net.train_network(sequences, labels)
                self.get_logger().info(f'Trained gesture network on {len(sequences)} sequences')

            response.prediction = 0.0
            return response

        except Exception as e:
            self.get_logger().error(f'Error in train_gesture: {str(e)}')
            response.prediction = -1.0
            return response
        
    def _balance_data(self, buffer_0, buffer_1):
        """
        Balance two buffers by duplicating samples from the smaller buffer.
        
        Args:
            buffer_0: Deque of class 0 samples
            buffer_1: Deque of class 1 samples
            
        Returns:
            tuple: (balanced_samples, balanced_labels)
        """
        # Convert deques to lists for easier manipulation
        samples_0 = list(buffer_0)
        samples_1 = list(buffer_1)
        
        # Get lengths
        len_0 = len(samples_0)
        len_1 = len(samples_1)
        
        if len_0 == 0 or len_1 == 0:
            self.get_logger().warn("One or both classes have no samples")
            return [], []
            
        # Calculate how many samples we need to duplicate
        target_size = max(len_0, len_1)
        
        # Duplicate samples from the smaller buffer
        if len_0 < target_size:
            additional_samples = random.choices(samples_0, k=target_size - len_0)
            samples_0.extend(additional_samples)
        elif len_1 < target_size:
            additional_samples = random.choices(samples_1, k=target_size - len_1)
            samples_1.extend(additional_samples)
            
        # Create balanced labels
        labels_0 = [torch.tensor([[0]], dtype=torch.float32) for _ in range(target_size)]
        labels_1 = [torch.tensor([[1]], dtype=torch.float32) for _ in range(target_size)]
        
        # Combine and shuffle
        all_samples = samples_0 + samples_1
        all_labels = labels_0 + labels_1
        
        # Shuffle both lists with the same random order
        combined = list(zip(all_samples, all_labels))
        random.shuffle(combined)
        all_samples, all_labels = zip(*combined)
        
        return list(all_samples), list(all_labels)

    def train_sorting_callback(self, request, response):
        """Handle training requests for the sorting network."""
        try:
            # Convert ROS Image to CV2 image and preprocess
            cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='rgb8')
            image_tensor = self._preprocess_image(cv_image)

            # Add to appropriate buffer
            if request.label == 0:
                self.class_0_images.append(image_tensor)
            else:
                self.class_1_images.append(image_tensor)

            # Get buffer sizes for logging
            size_0 = len(self.class_0_images)
            size_1 = len(self.class_1_images)
            self.get_logger().info(
                f"Current buffer sizes - Class 0: {size_0}, Class 1: {size_1}"
            )
            
            # Get balanced training data
            images, labels = self._balance_data(self.class_0_images, self.class_1_images)
            
            # Train if we have images
            if images:
                self.sorting_net.train_network(images, labels)
                self.get_logger().info(
                    f"Trained network on {len(images)} images "
                    f"({len(images)//2} per class)"
                )
            
            response.prediction = 0.0
            return response
            
        except Exception as e:
            self.get_logger().error(f"Error training sorting network: {str(e)}")
            response.prediction = -1.0
            return response

    def get_sorting_prediction_callback(self, request, response):
        """Handle prediction requests for the sorting network."""
        try:
            # Convert ROS Image to CV2 image
            cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='rgb8')
            
            # Convert to tensor
            image_tensor = self._preprocess_image(cv_image)
            
            # Get prediction from network
            with torch.no_grad():
                prediction = self.sorting_net(image_tensor)
            
            response.prediction = float(prediction.detach().numpy())
            self.get_logger().info(f'Sorting prediction: {response.prediction}')
            return response
            
        except Exception as e:
            self.get_logger().error(f'Error getting sorting prediction: {str(e)}')
            response.prediction = -1.0
            return response

    def _preprocess_image(self, cv_image):
        """Convert CV image to tensor and preprocess for sorting network."""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(cv_image).unsqueeze(0)
    
    def save_sorting_callback(self, request, response):
        """Save sorting network to file."""
        try:
            # Generate filename with timestamp
            timestamp = self.get_clock().now().to_msg().sec
            
            # Use custom filename if provided, otherwise use timestamp
            if request.filename:
                filepath = f"{self.save_dir}/{request.filename}"
            else:
                filepath = f"{self.save_dir}/sorting_net_{timestamp}.pt"
            
            # Save network
            self.sorting_net.save_model(filepath)
            
            response.filepath = filepath
            response.success = True
            
            self.get_logger().info(f"Saved sorting network to {filepath}")
            
        except Exception as e:
            self.get_logger().error(f"Error saving sorting network: {str(e)}")
            response.success = False
            response.filepath = ""
            
        return response

    def save_gesture_callback(self, request, response):
        """Save gesture network to file."""
        try:
            # Generate filename with timestamp
            timestamp = self.get_clock().now().to_msg().sec
            
            # Use custom filename if provided, otherwise use timestamp
            if request.filename:
                filepath = f"{self.save_dir}/{request.filename}"
            else:
                filepath = f"{self.save_dir}/gesture_net_{timestamp}.pt"
            
            # Save network
            self.gesture_net.save_model(filepath)
            
            response.filepath = filepath
            response.success = True
            
            self.get_logger().info(f"Saved gesture network to {filepath}")
            
        except Exception as e:
            self.get_logger().error(f"Error saving gesture network: {str(e)}")
            response.success = False
            response.filepath = ""
            
        return response

def main(args=None):
    rclpy.init(args=args)
    node = NetworkNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()