#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from franka_hri_interfaces.srv import SortNet, GestNet
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import torch
import numpy as np
from franka_hri_sorting.network import SortingNet, GestureNet
import torchvision.transforms as transforms
from cv_bridge import CvBridge, CvBridgeError
from collections import deque

class NetworkNode(Node):
    def __init__(self):
        super().__init__('network_node')
        
        # Initialize parameters
        self.declare_parameter('buffer_size', 25)
        self.declare_parameter('sequence_length', 20)
        self.buffer_size = self.get_parameter('buffer_size').value
        self.sequence_length = self.get_parameter('sequence_length').value
        
        # Initialize networks
        self.sorting_net = SortingNet()
        self.gesture_net = GestureNet(sequence_length=self.sequence_length)
        self.bridge = CvBridge()
        
        # Initialize training buffers for sorting network
        self.class_0_images = deque(maxlen=self.buffer_size)
        self.class_1_images = deque(maxlen=self.buffer_size)
        
        # Initialize training buffers for gesture network
        self.class_0_gestures = deque(maxlen=self.buffer_size)
        self.class_1_gestures = deque(maxlen=self.buffer_size)
        
        # Buffer for synchronizing modality data
        self.mod48_buffer = None  # Store most recent mod48 data
        self.combined_sequence = []  # Store combined data points
        
        # Create services for sorting network
        self.create_service(SortNet, 'train_sorting', self.train_sorting_callback)
        self.create_service(SortNet, 'get_sorting_prediction', self.get_sorting_prediction_callback)

        # Create services for gesture network
        self.create_service(GestNet, 'train_gesture', self.train_gesture_callback)
        self.create_service(GestNet, 'get_gesture_prediction', self.get_gesture_prediction_callback)
        
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

    def preprocess_sequence(self, sequence):
        """Convert sequence to tensor format for network input."""
        # Ensure sequence is the right length
        if len(sequence) > self.sequence_length:
            sequence = sequence[-self.sequence_length:]
        elif len(sequence) < self.sequence_length:
            # Pad with zeros if sequence is too short
            padding = [np.zeros((5,)) for _ in range(self.sequence_length - len(sequence))]
            sequence = padding + sequence
            
        # Stack sequence into array and convert to tensor
        data = np.stack(sequence, axis=1)  # Shape: (5, sequence_length)
        return torch.FloatTensor(data).unsqueeze(0)  # Add batch dimension

    def mod48_callback(self, msg):
        """Handle incoming modality 48 data."""
        try:
            # Extract data using correct dimensions
            rows = msg.layout.dim[0].size  # 50
            cols = msg.layout.dim[1].size  # 4
            data = np.array(msg.data).reshape(rows, cols)
            self.mod48_buffer = data[:, 1:]  # Remove timestamp column, should be 50x3

        except Exception as e:
            self.get_logger().error(f'Error processing modality 48 data: {str(e)}')

    def mod51_callback(self, msg):
        """Handle incoming modality 51 data and combine with mod48 data."""
        try:
            if self.mod48_buffer is None:
                return

            # Extract data using correct dimensions and handle variable length
            data = np.array(msg.data)
            if len(data) > 100:  # If we have more than 50 pairs (timestamp, value)
                # Take only the last 100 entries (50 pairs)
                data = data[-100:]
                
            rows = 50  # We always want 50 rows
            cols = 2   # timestamp and value
            mod51_data = data.reshape(rows, cols)
            mod51_features = mod51_data[:, 1:]  # Remove timestamp column, should be 50x1

            # Combine features horizontally (keeping time series intact)
            combined_features = np.hstack((self.mod48_buffer, mod51_features))  # Should be 50x4
            
            self.combined_sequence.append(combined_features)

            # Keep only the most recent sequence_length points
            if len(self.combined_sequence) > self.sequence_length:
                self.combined_sequence = self.combined_sequence[-self.sequence_length:]

        except Exception as e:
            self.get_logger().error(f'Error processing modality 51 data: {str(e)}')

    def get_gesture_prediction_callback(self, request, response):
        """Handle prediction requests for the gesture network."""
        try:
            # Check if we have enough data in the combined sequence
            if len(self.combined_sequence) < self.sequence_length:
                self.get_logger().warn('Not enough data points for prediction')
                response.prediction = -1.0
                return response

            # Take the last sequence_length points
            recent_sequence = self.combined_sequence[-self.sequence_length:]
            
            # Convert sequence to array and reshape
            # Should be [batch_size, channels, sequence_length]
            sequence_data = np.array(recent_sequence)  # Should be [sequence_length, 50, 4]
            sequence_data = sequence_data.transpose(2, 0, 1)  # Reorder to [4, sequence_length, 50]
            sequence_data = sequence_data[:, :, 0]  # Take first time point from each sequence, shape: [4, sequence_length]
            
            # Add batch dimension
            sequence_tensor = torch.FloatTensor(sequence_data).unsqueeze(0)  # Shape: [1, 4, sequence_length]
            
            # Debug print
            self.get_logger().info(f'Input tensor shape: {sequence_tensor.shape}')
            
            # Get prediction from network
            with torch.no_grad():
                prediction = self.gesture_net(sequence_tensor)
            
            response.prediction = float(prediction.item())
            self.get_logger().info(f'Gesture prediction: {response.prediction}')
            return response
                
        except Exception as e:
            self.get_logger().error(f'Error getting gesture prediction: {str(e)}')
            response.prediction = -1.0
            return response

    def train_gesture_callback(self, request, response):
        """Handle training requests for the gesture network."""
        try:
            # Reshape the sequence data
            sequence = np.array(request.sequence).reshape(request.sequence_length, request.channels).T
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            
            # Add to appropriate buffer
            if request.label == 0:
                self.class_0_gestures.append(sequence_tensor)
            else:
                self.class_1_gestures.append(sequence_tensor)

            # Train on all sequences in both buffers
            all_sequences = []
            all_labels = []

            # Add class 0 sequences
            for seq in self.class_0_gestures:
                all_sequences.append(seq)
                all_labels.append(torch.tensor([[0.0]], dtype=torch.float32))

            # Add class 1 sequences
            for seq in self.class_1_gestures:
                all_sequences.append(seq)
                all_labels.append(torch.tensor([[1.0]], dtype=torch.float32))

            # Train if we have sequences
            if all_sequences:
                # Randomly shuffle the training data
                combined = list(zip(all_sequences, all_labels))
                np.random.shuffle(combined)
                all_sequences, all_labels = zip(*combined)

                # Train on each sequence
                for sequence, label in zip(all_sequences, all_labels):
                    self.gesture_net.optimizer.zero_grad()
                    output = self.gesture_net(sequence)
                    loss = self.gesture_net.criterion(output, label)
                    loss.backward()
                    self.gesture_net.optimizer.step()

                self.gesture_net.scheduler.step()
                self.get_logger().info(f'Trained gesture network on {len(all_sequences)} sequences')

            response.prediction = 0.0
            return response

        except Exception as e:
            self.get_logger().error(f'Error training gesture network: {str(e)}')
            response.prediction = -1.0
            return response

    def train_sorting_callback(self, request, response):
        """Handle training requests for the sorting network."""
        try:
            # Convert ROS Image to CV2 image
            cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='rgb8')
            
            # Convert to tensor
            image_tensor = self._preprocess_image(cv_image)

            # Add to appropriate buffer
            if request.label == 0:
                self.class_0_images.append(image_tensor)
            else:
                self.class_1_images.append(image_tensor)

            self.get_logger().info(f"Buffer sizes - Class 0: {len(self.class_0_images)}, Class 1: {len(self.class_1_images)}")
            
            # Train on all images in both buffers
            all_images = []
            all_labels = []

            # Add class 0 images
            for img in self.class_0_images:
                all_images.append(img)
                all_labels.append(torch.tensor([[0]], dtype=torch.float32))

            # Add class 1 images
            for img in self.class_1_images:
                all_images.append(img)
                all_labels.append(torch.tensor([[1]], dtype=torch.float32))

            # Train if we have images
            if all_images:
                self.sorting_net.train_network(all_images, all_labels)
                self.get_logger().info(f"Trained network on {len(all_images)} images")
            
            response.prediction = 0.0  # Training doesn't need a prediction
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

def main(args=None):
    rclpy.init(args=args)
    node = NetworkNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()