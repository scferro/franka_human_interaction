import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from franka_hri_interfaces.srv import SortNet, GestNet, SaveModel
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import torch
import numpy as np
from franka_hri_sorting.network import SortingNet, GestureNet, ComplexSortingNet, ComplexGestureNet
import torchvision.transforms as transforms
from cv_bridge import CvBridge, CvBridgeError
from collections import deque
import random
import csv
import os
import json
from datetime import datetime
import threading
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

class NetworkLogger:
    """Manages all logging operations for network predictions and performance."""
    
    def __init__(self, log_dir: str):
        """Initialize the logger with specified directory."""
        self.log_dir = log_dir
        self.lock = threading.Lock()
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log files with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sorting_log = os.path.join(log_dir, f"sorting_predictions_{timestamp}.csv")
        self.gesture_log = os.path.join(log_dir, f"gesture_predictions_{timestamp}.csv")
        self.complex_gesture_log = os.path.join(log_dir, f"complex_gesture_predictions_{timestamp}.csv")
        self.stats_file = os.path.join(log_dir, f"network_stats_{timestamp}.json")
        
        # Initialize statistics tracking
        self.stats = {
            'complex_sorting': {
                'correct': 0,
                'total': 0,
                'confusion_matrix': [[0, 0, 0, 0], [0, 0, 0, 0], 
                                   [0, 0, 0, 0], [0, 0, 0, 0]],
                'average_confidence': 0.0,
                'total_confidence': 0.0
            },
            'binary_gesture': {
                'correct': 0,
                'total': 0,
                'confusion_matrix': [[0, 0], [0, 0]],
                'average_confidence': 0.0,
                'total_confidence': 0.0
            },
            'complex_gesture': {
                'correct': 0,
                'total': 0,
                'confusion_matrix': [[0, 0, 0, 0], [0, 0, 0, 0],
                                   [0, 0, 0, 0], [0, 0, 0, 0]],
                'average_confidence': 0.0,
                'total_confidence': 0.0
            }
        }
        
        self._init_log_files()

    def _init_log_files(self):
        """Create log files with headers."""
        headers = [
            'timestamp',
            'prediction',
            'confidence',
            'true_label',
            'was_correct',
            'response_time_ms'
        ]
        
        for log_file in [self.sorting_log, self.gesture_log, self.complex_gesture_log]:
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_prediction_and_feedback(self, network_type: str, prediction: float, 
                                  confidence: float, true_label: float) -> None:
        """Log a prediction and its immediate feedback."""
        with self.lock:
            timestamp = datetime.now().timestamp()
            
            # Determine prediction correctness based on network type
            if network_type in ['complex_sorting', 'complex_gesture']:
                was_correct = (np.argmax(prediction) == true_label)
            else:
                was_correct = (prediction >= 0.5) == (true_label >= 0.5)
            
            # Update statistics
            stats = self.stats[network_type]
            stats['total'] += 1
            stats['total_confidence'] += confidence
            stats['average_confidence'] = stats['total_confidence'] / stats['total']
            
            if was_correct:
                stats['correct'] += 1
            
            # Update confusion matrix
            if network_type in ['complex_sorting', 'complex_gesture']:
                pred_class = np.argmax(prediction)
                true_class = int(true_label)
            else:
                pred_class = 1 if prediction >= 0.5 else 0
                true_class = 1 if true_label >= 0.5 else 0
            
            stats['confusion_matrix'][true_class][pred_class] += 1
            
            # Write to appropriate log file
            log_file = self._get_log_file(network_type)
            
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.fromtimestamp(timestamp).isoformat(),
                    prediction,
                    confidence,
                    true_label,
                    was_correct,
                    0  # Response time is 0 since feedback is immediate
                ])
            
            self._save_stats()

    def _get_log_file(self, network_type: str) -> str:
        """Get the appropriate log file based on network type."""
        if network_type == 'complex_sorting':
            return self.sorting_log
        elif network_type == 'binary_gesture':
            return self.gesture_log
        elif network_type == 'complex_gesture':
            return self.complex_gesture_log
        else:
            raise ValueError(f"Unknown network type: {network_type}")

    def _save_stats(self):
        """Save current statistics to JSON file."""
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=4)

    def get_current_stats(self) -> dict:
        """Return current statistics for all networks."""
        with self.lock:
            stats_copy = json.loads(json.dumps(self.stats))  # Deep copy
            
            # Calculate additional metrics
            for network_type, stats in stats_copy.items():
                if stats['total'] > 0:
                    stats['accuracy'] = stats['correct'] / stats['total']
                else:
                    stats['accuracy'] = 0.0
                    
            return stats_copy

class NetworkNode(Node):
    """Main node for managing neural networks and predictions."""
    
    def __init__(self):
        super().__init__('network_node')
        
        # Initialize parameters and components
        self._init_parameters()
        self._init_networks()
        self._init_buffers()
        self._init_services()
        self._init_subscribers()
        
        # Create timer for periodic stats logging
        self.create_timer(300.0, self._periodic_stats_logging)
        
        self.get_logger().info("Network node initialized successfully")

    def _init_parameters(self):
        """Initialize all ROS parameters."""
        # Declare and get all parameters
        self.declare_parameter('buffer_size', 25)
        self.declare_parameter('sequence_length', 20)
        self.declare_parameter('sorting_model_path', '')
        self.declare_parameter('gesture_model_path', '')
        self.declare_parameter('complex_gesture_model_path', '')
        self.declare_parameter('save_directory', '/home/user/saved_models')
        self.declare_parameter('log_directory', '/home/user/network_logs')
        
        # Store parameter values
        self.buffer_size = self.get_parameter('buffer_size').value
        self.sequence_length = self.get_parameter('sequence_length').value
        self.sorting_model_path = self.get_parameter('sorting_model_path').value
        self.gesture_model_path = self.get_parameter('gesture_model_path').value
        self.complex_gesture_model_path = self.get_parameter('complex_gesture_model_path').value
        self.save_dir = self.get_parameter('save_directory').value
        self.log_dir = self.get_parameter('log_directory').value
        
        # Create logger
        self.logger = NetworkLogger(self.log_dir)

    def _init_networks(self):
        """Initialize neural networks."""
        try:
            # Initialize complex sorting network with 4 classes
            self.sorting_net = ComplexSortingNet(
                num_classes=4,
                model_path=self.sorting_model_path
            )
            
            # Initialize binary gesture network
            self.gesture_net = GestureNet(
                sequence_length=self.sequence_length,
                model_path=self.gesture_model_path
            )
            
            # Initialize complex gesture network with 4 classes
            self.complex_gesture_net = ComplexGestureNet(
                num_classes=4,
                sequence_length=self.sequence_length,
                model_path=self.complex_gesture_model_path
            )
            
            self.bridge = CvBridge()
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize networks: {str(e)}")
            raise

    def _init_buffers(self):
        """Initialize all data buffers."""
        # Training data buffers for complex sorting (4 classes)
        self.sorting_class_buffers = [
            deque(maxlen=self.buffer_size) for _ in range(4)
        ]
        
        # Training data buffers for binary gesture
        self.gesture_class_0 = deque(maxlen=self.buffer_size)
        self.gesture_class_1 = deque(maxlen=self.buffer_size)
        
        # Training data buffers for complex gesture (4 classes)
        self.complex_gesture_buffers = [
            deque(maxlen=self.buffer_size) for _ in range(4)
        ]
        
        # Sensor data buffers
        self.mod48_buffer = None
        self.combined_sequence = []
        self.last_inference_sequence = []

    def _init_services(self):
        """Initialize all ROS services."""
        # Sorting network services
        self.create_service(
            SortNet, 
            'train_sorting', 
            self.train_sorting_callback
        )
        self.create_service(
            SortNet, 
            'get_sorting_prediction',
            self.get_sorting_prediction_callback
        )
        
        # Binary gesture network services
        self.create_service(
            GestNet, 
            'train_gesture', 
            self.train_gesture_callback
        )
        self.create_service(
            GestNet, 
            'get_gesture_prediction',
            self.get_gesture_prediction_callback
        )
        
        # Complex gesture network services
        self.create_service(
            GestNet, 
            'train_complex_gesture',
            self.train_complex_gesture_callback
        )
        self.create_service(
            GestNet, 
            'get_complex_gesture_prediction',
            self.get_complex_gesture_prediction_callback
        )
        
        # Model saving services
        self.create_service(
            SaveModel, 
            'save_sorting_network',
            self.save_sorting_callback
        )
        self.create_service(
            SaveModel, 
            'save_gesture_network',
            self.save_gesture_callback
        )
        self.create_service(
            SaveModel, 
            'save_complex_gesture_network',
            self.save_complex_gesture_callback
        )

    def _init_subscribers(self):
        """Initialize all ROS subscribers."""
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

    def _periodic_stats_logging(self):
        """Timer callback to log periodic statistics."""
        try:
            stats = self.logger.get_current_stats()
            self.get_logger().info(
                f"Current network statistics: {json.dumps(stats, indent=2)}"
            )
        except Exception as e:
            self.get_logger().error(f"Error logging periodic stats: {str(e)}")

    def get_sorting_prediction_callback(self, request, response):
        """Handle prediction requests for the sorting network."""
        try:
            # Convert image and make prediction
            cv_image = self.bridge.imgmsg_to_cv2(
                request.image, 
                desired_encoding='rgb8'
            )
            image_tensor = self._preprocess_image(cv_image)
            
            with torch.no_grad():
                output = self.sorting_net(image_tensor)
                prediction = output.numpy()[0]  # Get probabilities for all classes
                confidence = float(torch.max(output).item())
            
            response.prediction = float(np.argmax(prediction))  # Return predicted class
            
            self.get_logger().info(
                f'Sorting prediction: {response.prediction} '
                f'(confidence: {confidence:.3f})'
            )
            return response
            
        except Exception as e:
            self.get_logger().error(f'Error in sorting prediction: {str(e)}')
            response.prediction = -1.0
            return response

    def get_gesture_prediction_callback(self, request, response):
        """Handle prediction requests for the binary gesture network."""
        try:
            if len(self.combined_sequence) < self.sequence_length:
                response.prediction = -1.0
                return response

            # Process and normalize sequence
            sequence_tensor = self.process_sensor_data(self.combined_sequence)
            if sequence_tensor is None:
                response.prediction = -1.0
                return response

            normalized_sequence = self.normalize_sequence(sequence_tensor)
            
            # Make prediction
            with torch.no_grad():
                output = self.gesture_net(normalized_sequence)
                prediction = float(output.item())
                confidence = float(output.item())
            
            response.prediction = prediction
            
            self.get_logger().info(
                f'Binary gesture prediction: {prediction:.3f} '
                f'(confidence: {confidence:.3f})'
            )
            
            self.last_inference_sequence = self.combined_sequence
            return response
            
        except Exception as e:
            self.get_logger().error(f'Error in gesture prediction: {str(e)}')
            response.prediction = -1.0
            return response

    def get_complex_gesture_prediction_callback(self, request, response):
        """Handle prediction requests for the complex gesture network."""
        try:
            if len(self.combined_sequence) < self.sequence_length:
                response.prediction = -1.0
                return response

            # Process and normalize sequence
            sequence_tensor = self.process_sensor_data(self.combined_sequence)
            if sequence_tensor is None:
                response.prediction = -1.0
                return response

            normalized_sequence = self.normalize_sequence(sequence_tensor)
            
            # Make prediction
            with torch.no_grad():
                output = self.complex_gesture_net(normalized_sequence)
                prediction = output.numpy()[0]  # Get probabilities for all classes
                confidence = float(torch.max(output).item())
            
            response.prediction = float(np.argmax(prediction))  # Return predicted class
            
            self.get_logger().info(
                f'Complex gesture prediction: {response.prediction} '
                f'(confidence: {confidence:.3f})'
            )
            
            self.last_inference_sequence = self.combined_sequence
            return response
            
        except Exception as e:
            self.get_logger().error(f'Error in complex gesture prediction: {str(e)}')
            response.prediction = -1.0
            return response

    def train_sorting_callback(self, request, response):
        """Handle training requests for the complex sorting network."""
        try:
            # Convert image and preprocess
            cv_image = self.bridge.imgmsg_to_cv2(request.image, 
                                                desired_encoding='rgb8')
            image_tensor = self._preprocess_image(cv_image)

            # Add to appropriate class buffer
            class_idx = int(request.label)
            if 0 <= class_idx < 4:
                self.sorting_class_buffers[class_idx].append(image_tensor)
            else:
                raise ValueError(f"Invalid class index: {class_idx}")

            # Log buffer sizes
            buffer_sizes = [len(buffer) for buffer in self.sorting_class_buffers]
            self.get_logger().info(f"Current sorting buffer sizes: {buffer_sizes}")
            
            # Get balanced training data
            images, labels = self._balance_multi_class_data(
                self.sorting_class_buffers)
            
            # Train if we have images
            if images:
                # Make prediction before training for logging
                with torch.no_grad():
                    output = self.sorting_net(image_tensor)
                    prediction = output.numpy()[0]
                    confidence = float(torch.max(output).item())
                
                # Train network
                self.sorting_net.train_network(images, labels)
                
                # Log prediction and feedback
                self.logger.log_prediction_and_feedback(
                    'complex_sorting',
                    prediction,
                    confidence,
                    request.label
                )
                
                self.get_logger().info(
                    f"Trained sorting network on {len(images)} images"
                )
            
            response.prediction = 0.0
            return response
            
        except Exception as e:
            self.get_logger().error(f"Error training sorting network: {str(e)}")
            response.prediction = -1.0
            return response

    def train_gesture_callback(self, request, response):
        """Handle training requests for the binary gesture network."""
        try:
            if len(self.last_inference_sequence) < self.sequence_length:
                self.get_logger().warn('Not enough data points for training')
                response.prediction = -1.0
                return response

            # Process and normalize sequence
            sequence_tensor = self.process_sensor_data(
                self.last_inference_sequence)
            if sequence_tensor is None:
                response.prediction = -1.0
                return response

            normalized_sequence = self.normalize_sequence(sequence_tensor)

            # Add to appropriate buffer
            if request.label == 0:
                self.gesture_class_0.append(normalized_sequence)
            else:
                self.gesture_class_1.append(normalized_sequence)

            # Get current prediction for logging
            with torch.no_grad():
                output = self.gesture_net(normalized_sequence)
                prediction = float(output.item())
                confidence = float(output.item())

            # Get balanced training data
            sequences, labels = self._balance_data(
                self.gesture_class_0,
                self.gesture_class_1
            )

            if sequences:
                # Train network
                self.gesture_net.train_network(sequences, labels)
                
                # Log prediction and feedback
                self.logger.log_prediction_and_feedback(
                    'binary_gesture',
                    prediction,
                    confidence,
                    request.label
                )
                
                self.get_logger().info(
                    f'Trained gesture network on {len(sequences)} sequences'
                )

            response.prediction = 0.0
            return response

        except Exception as e:
            self.get_logger().error(f'Error in train_gesture: {str(e)}')
            response.prediction = -1.0
            return response

    def train_complex_gesture_callback(self, request, response):
        """Handle training requests for the complex gesture network."""
        try:
            if len(self.last_inference_sequence) < self.sequence_length:
                self.get_logger().warn('Not enough data points for training')
                response.prediction = -1.0
                return response

            # Process and normalize sequence
            sequence_tensor = self.process_sensor_data(
                self.last_inference_sequence)
            if sequence_tensor is None:
                response.prediction = -1.0
                return response

            normalized_sequence = self.normalize_sequence(sequence_tensor)

            # Add to appropriate class buffer
            class_idx = int(request.label)
            if 0 <= class_idx < 4:
                self.complex_gesture_buffers[class_idx].append(
                    normalized_sequence)
            else:
                raise ValueError(f"Invalid class index: {class_idx}")

            # Get current prediction for logging
            with torch.no_grad():
                output = self.complex_gesture_net(normalized_sequence)
                prediction = output.numpy()[0]
                confidence = float(torch.max(output).item())

            # Get balanced training data
            sequences, labels = self._balance_multi_class_data(
                self.complex_gesture_buffers)

            if sequences:
                # Train network
                self.complex_gesture_net.train_network(sequences, labels)
                
                # Log prediction and feedback
                self.logger.log_prediction_and_feedback(
                    'complex_gesture',
                    prediction,
                    confidence,
                    request.label
                )
                
                self.get_logger().info(
                    f'Trained complex gesture network on {len(sequences)} sequences'
                )

            response.prediction = 0.0
            return response

        except Exception as e:
            self.get_logger().error(
                f'Error in train_complex_gesture: {str(e)}')
            response.prediction = -1.0
            return response

    def save_sorting_callback(self, request, response):
        """Save sorting network to file."""
        try:
            timestamp = self.get_clock().now().to_msg().sec
            
            if request.filename:
                filepath = f"{self.save_dir}/{request.filename}"
            else:
                filepath = f"{self.save_dir}/sorting_net_{timestamp}.pt"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
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
        """Save binary gesture network to file."""
        try:
            timestamp = self.get_clock().now().to_msg().sec
            
            if request.filename:
                filepath = f"{self.save_dir}/{request.filename}"
            else:
                filepath = f"{self.save_dir}/gesture_net_{timestamp}.pt"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
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

    def save_complex_gesture_callback(self, request, response):
        """Save complex gesture network to file."""
        try:
            timestamp = self.get_clock().now().to_msg().sec
            
            if request.filename:
                filepath = f"{self.save_dir}/{request.filename}"
            else:
                filepath = f"{self.save_dir}/complex_gesture_net_{timestamp}.pt"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save network
            self.complex_gesture_net.save_model(filepath)
            
            response.filepath = filepath
            response.success = True
            
            self.get_logger().info(f"Saved complex gesture network to {filepath}")
            
        except Exception as e:
            self.get_logger().error(
                f"Error saving complex gesture network: {str(e)}")
            response.success = False
            response.filepath = ""
            
        return response

    def _preprocess_image(self, cv_image):
        """Convert CV image to tensor and preprocess for sorting network."""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(cv_image).unsqueeze(0)

    def process_sensor_data(self, sequence):
        """Process sensor data into correct format for network."""
        try:
            if len(sequence) > self.sequence_length:
                sequence = sequence[-self.sequence_length:]
            elif len(sequence) < self.sequence_length:
                padding = [np.zeros((4,)) for _ in range(
                    self.sequence_length - len(sequence))]
                sequence = padding + sequence
                
            data = np.stack(sequence, axis=0).T
            tensor_data = torch.FloatTensor(data).unsqueeze(0)
            return tensor_data
                
        except Exception as e:
            self.get_logger().error(f'Error processing sensor data: {str(e)}')
            return None

    def normalize_sequence(self, tensor_data):
        """Normalize the full sequence."""
        try:
            mean = tensor_data.mean()
            std = tensor_data.std() + 1e-8
            normalized = (tensor_data - mean) / std
            return normalized
        except Exception as e:
            self.get_logger().error(f'Error normalizing sequence: {str(e)}')
            return tensor_data

    def _balance_data(self, buffer_0, buffer_1):
        """Balance two buffers by duplicating samples from smaller buffer."""
        samples_0 = list(buffer_0)
        samples_1 = list(buffer_1)
        
        len_0 = len(samples_0)
        len_1 = len(samples_1)
        
        if len_0 == 0 or len_1 == 0:
            self.get_logger().warn("One or both classes have no samples")
            return [], []
            
        target_size = max(len_0, len_1)
        
        if len_0 < target_size:
            additional_samples = random.choices(
                samples_0, 
                k=target_size - len_0
            )
            samples_0.extend(additional_samples)
        elif len_1 < target_size:
            additional_samples = random.choices(
                samples_1,
                k=target_size - len_1
            )
            samples_1.extend(additional_samples)
            
        labels_0 = [torch.tensor([[0]], dtype=torch.float32) 
                   for _ in range(target_size)]
        labels_1 = [torch.tensor([[1]], dtype=torch.float32)
                   for _ in range(target_size)]
        
        all_samples = samples_0 + samples_1
        all_labels = labels_0 + labels_1
        
        combined = list(zip(all_samples, all_labels))
        random.shuffle(combined)
        all_samples, all_labels = zip(*combined)
        
        return list(all_samples), list(all_labels)

    def _balance_multi_class_data(self, class_buffers):
        """Balance multiple class buffers for training."""
        # Convert all buffers to lists
        class_samples = [list(buffer) for buffer in class_buffers]
        
        # Get lengths of non-empty buffers
        lengths = [len(samples) for samples in class_samples]
        
        # Find which categories have samples
        non_empty_indices = [i for i, length in enumerate(lengths) if length > 0]
        
        if not non_empty_indices:
            self.get_logger().warn("No samples available for training")
            return [], []
        
        # Use only non-empty categories
        target_size = max(len(class_samples[i]) for i in non_empty_indices)
        
        balanced_samples = []
        balanced_labels = []
        
        # Process each non-empty category
        for class_idx in non_empty_indices:
            samples = class_samples[class_idx]
            if len(samples) < target_size:
                # Oversample from this category to match target size
                additional = random.choices(
                    samples,
                    k=target_size - len(samples)
                )
                samples.extend(additional)
            
            balanced_samples.extend(samples)
            balanced_labels.extend([
                torch.tensor([class_idx], dtype=torch.long)
                for _ in range(target_size)
            ])
        
        # Shuffle samples and labels together
        combined = list(zip(balanced_samples, balanced_labels))
        random.shuffle(combined)
        balanced_samples, balanced_labels = zip(*combined)
        
        self.get_logger().info(
            f"Created balanced training batch from categories {non_empty_indices} "
            f"with {target_size} samples per category"
        )
        
        return list(balanced_samples), list(balanced_labels)

    def mod48_callback(self, msg):
        """Handle incoming modality 48 data."""
        try:
            data = np.array(msg.data).reshape(
                msg.layout.dim[0].size,
                msg.layout.dim[1].size
            )
            self.mod48_buffer = data[:, 1:]  # Remove timestamp
        except Exception as e:
            self.get_logger().error(f'Error in mod48_callback: {str(e)}')
        
    def mod51_callback(self, msg):
        """Handle incoming modality 51 data and combine with mod48 data."""
        try:
            if self.mod48_buffer is None:
                return

            data = np.array(msg.data)
            if len(data) > 100:
                data = data[-100:]
            
            rows = 50
            cols = 2
            mod51_data = data.reshape(rows, cols)
            mod51_features = mod51_data[:, 1:]
            
            combined_features = np.hstack((self.mod48_buffer, mod51_features))
            current_point = combined_features[0]
            
            self.combined_sequence.append(current_point)

            if len(self.combined_sequence) > self.sequence_length:
                self.combined_sequence = self.combined_sequence[-self.sequence_length:]

        except Exception as e:
            self.get_logger().error(f'Error processing modality 51 data: {str(e)}')

def main(args=None):
    """Main function to initialize and run the node."""
    rclpy.init(args=args)
    try:
        node = NetworkNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()