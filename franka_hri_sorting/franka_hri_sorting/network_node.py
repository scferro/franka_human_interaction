import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from franka_hri_interfaces.srv import SortNet, GestNet, SaveModel, CorrectionService
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
import copy

class NetworkState:
    """Class to store network state for potential rollback."""
    def __init__(self, model_state, optimizer_state, scheduler_state):
        self.model_state = model_state
        self.optimizer_state = optimizer_state
        self.scheduler_state = scheduler_state

    @classmethod
    def from_network(cls, network):
        return cls(
            copy.deepcopy(network.state_dict()),
            copy.deepcopy(network.optimizer.state_dict()),
            copy.deepcopy(network.scheduler.state_dict())
        )

    def apply_to_network(self, network):
        network.load_state_dict(self.model_state)
        network.optimizer.load_state_dict(self.optimizer_state)
        network.scheduler.load_state_dict(self.scheduler_state)

@dataclass
class TrainingSample:
    """Stores information about a training sample."""
    input_data: any  # Image or sequence
    label: int
    network_state: NetworkState
    buffer_state: any  

class NetworkLogger:
    """Manages all logging operations for network predictions and performance."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.lock = threading.Lock()
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log files with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sorting_log = os.path.join(log_dir, f"sorting_predictions_{timestamp}.csv")
        self.gesture_log = os.path.join(log_dir, f"gesture_predictions_{timestamp}.csv")
        self.complex_gesture_log = os.path.join(log_dir, f"complex_gesture_predictions_{timestamp}.csv")
        self.correction_log = os.path.join(log_dir, f"corrections_{timestamp}.csv")
        self.stats_file = os.path.join(log_dir, f"network_stats_{timestamp}.json")
        
        # Initialize statistics tracking
        self.stats = {
            'complex_sorting': {
                'correct': 0,
                'total': 0,
                'corrections': 0,
                'confusion_matrix': [[0, 0, 0, 0], [0, 0, 0, 0], 
                                   [0, 0, 0, 0], [0, 0, 0, 0]],
                'average_confidence': 0.0,
                'total_confidence': 0.0
            },
            'binary_gesture': {
                'correct': 0,
                'total': 0,
                'corrections': 0,
                'confusion_matrix': [[0, 0], [0, 0]],
                'average_confidence': 0.0,
                'total_confidence': 0.0
            },
            'complex_gesture': {
                'correct': 0,
                'total': 0,
                'corrections': 0,
                'confusion_matrix': [[0, 0, 0, 0], [0, 0, 0, 0],
                                   [0, 0, 0, 0], [0, 0, 0, 0]],
                'average_confidence': 0.0,
                'total_confidence': 0.0
            }
        }
        
        self._init_log_files()

    def _init_log_files(self):
        """Create log files with headers."""
        prediction_headers = [
            'timestamp',
            'prediction',
            'confidence',
            'true_label',
            'was_correct',
            'response_time_ms'
        ]
        
        correction_headers = [
            'timestamp',
            'network_type',
            'old_label',
            'new_label',
            'success'
        ]
        
        for log_file in [self.sorting_log, self.gesture_log, self.complex_gesture_log]:
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(prediction_headers)
                
        with open(self.correction_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(correction_headers)

    def log_correction(self, network_type: str, old_label: int, new_label: int, success: bool):
        """Log a correction attempt."""
        with self.lock:
            timestamp = datetime.now().timestamp()
            
            with open(self.correction_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.fromtimestamp(timestamp).isoformat(),
                    network_type,
                    old_label,
                    new_label,
                    success
                ])
            
            if success:
                self.stats[network_type]['corrections'] += 1
                self._save_stats()

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
                    stats['correction_rate'] = stats['corrections'] / stats['total']
                else:
                    stats['accuracy'] = 0.0
                    stats['correction_rate'] = 0.0
                    
            return stats_copy
class DataSaver:
    """Manages saving of training data samples."""
    def __init__(self, base_dir: str, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            return
            
        # Create directory structure
        self.base_dir = Path(base_dir)
        self.image_dir = self.base_dir / 'images'
        self.gesture_dir = self.base_dir / 'gestures'
        
        # Ensure directories exist
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.gesture_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped files for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.image_file = self.image_dir / f"image_data_{timestamp}.csv"
        self.gesture_file = self.gesture_dir / f"gesture_data_{timestamp}.csv"
        
        # Initialize files with headers
        self._init_files()
        
    def _init_files(self):
        """Initialize CSV files with headers."""
        image_headers = ['timestamp', 'label', 'confidence', 'filename']
        gesture_headers = ['timestamp', 'label', 'confidence', 'sequence_data']
        
        with open(self.image_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(image_headers)
            
        with open(self.gesture_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(gesture_headers)
    
    def save_image_data(self, image_tensor: torch.Tensor, label: int, 
                       confidence: float = None):
        """Save image data and metadata."""
        if not self.enabled:
            return
            
        try:
            timestamp = datetime.now().isoformat()
            
            # Save image tensor as numpy array
            image_name = f"image_{timestamp}.npy"
            image_path = self.image_dir / image_name
            np.save(image_path, image_tensor.numpy())
            
            # Save metadata to CSV
            with open(self.image_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, label, confidence, image_name])
                
        except Exception as e:
            print(f"Error saving image data: {e}")
    
    def save_gesture_data(self, sequence: list, label: int, 
                         confidence: float = None):
        """Save gesture sequence data and metadata."""
        if not self.enabled:
            return
            
        try:
            timestamp = datetime.now().isoformat()
            
            # Convert sequence to string representation
            sequence_str = str(sequence)
            
            # Save to CSV
            with open(self.gesture_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, label, confidence, sequence_str])
                
        except Exception as e:
            print(f"Error saving gesture data: {e}")

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
        self._init_correction_state()
        
        # Create timer for periodic stats logging
        self.create_timer(300.0, self._periodic_stats_logging)
        
        self.get_logger().info("Network node initialized successfully")

    def _init_parameters(self):
        """Initialize all ROS parameters."""
        self.declare_parameter('buffer_size', 25)
        self.declare_parameter('sequence_length', 20)
        self.declare_parameter('sorting_model_path', '')
        self.declare_parameter('gesture_model_path', '')
        self.declare_parameter('complex_gesture_model_path', '')
        self.declare_parameter('save_directory', '/home/user/saved_models')
        self.declare_parameter('log_directory', '/home/user/network_logs')
        self.declare_parameter('save_training_data', True)
        self.declare_parameter('training_data_dir', '/home/user/training_data')
        
        self.buffer_size = self.get_parameter('buffer_size').value
        self.sequence_length = self.get_parameter('sequence_length').value
        self.sorting_model_path = self.get_parameter('sorting_model_path').value
        self.gesture_model_path = self.get_parameter('gesture_model_path').value
        self.complex_gesture_model_path = self.get_parameter('complex_gesture_model_path').value
        self.save_dir = self.get_parameter('save_directory').value
        self.log_dir = self.get_parameter('log_directory').value
        save_training = self.get_parameter('save_training_data').value
        training_data_dir = self.get_parameter('training_data_dir').value
        
        self.logger = NetworkLogger(self.log_dir)
        self.data_saver = DataSaver(training_data_dir, save_training)

    def _init_networks(self):
        """Initialize neural networks."""
        try:
            self.sorting_net = ComplexSortingNet(
                num_classes=4,
                model_path=self.sorting_model_path
            )
            
            self.gesture_net = GestureNet(
                sequence_length=self.sequence_length,
                model_path=self.gesture_model_path
            )
            
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
        self.sorting_class_buffers = [
            deque(maxlen=self.buffer_size) for _ in range(4)
        ]
        
        self.gesture_class_0 = deque(maxlen=self.buffer_size)
        self.gesture_class_1 = deque(maxlen=self.buffer_size)
        
        self.complex_gesture_buffers = [
            deque(maxlen=self.buffer_size) for _ in range(4)
        ]
        
        self.mod48_buffer = None
        self.combined_sequence = []
        self.last_inference_sequence = []

    def _init_correction_state(self):
        """Initialize state tracking for corrections."""
        self.last_training_samples = {
            'sorting': None,
            'gesture': None,
            'complex_gesture': None
        }

    def _init_services(self):
        """Initialize all ROS services."""
        # Training services
        self.create_service(SortNet, 'train_sorting', self.train_sorting_callback)
        self.create_service(GestNet, 'train_gesture', self.train_gesture_callback)
        self.create_service(GestNet, 'train_complex_gesture', self.train_complex_gesture_callback)
        
        # Prediction services
        self.create_service(SortNet, 'get_sorting_prediction', self.get_sorting_prediction_callback)
        self.create_service(GestNet, 'get_gesture_prediction', self.get_gesture_prediction_callback)
        self.create_service(GestNet, 'get_complex_gesture_prediction', self.get_complex_gesture_prediction_callback)
        
        # Model saving services
        self.create_service(SaveModel, 'save_sorting_network', self.save_sorting_callback)
        self.create_service(SaveModel, 'save_gesture_network', self.save_gesture_callback)
        self.create_service(SaveModel, 'save_complex_gesture_network', self.save_complex_gesture_callback)
        
        # Correction services
        self.create_service(CorrectionService, 'correct_sorting', self.correct_sorting_callback)
        self.create_service(CorrectionService, 'correct_gesture', self.correct_gesture_callback)
        self.create_service(CorrectionService, 'correct_complex_gesture', self.correct_complex_gesture_callback)

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

    def _cache_training_state(self, network_type: str, input_data: any, label: int):
        """Cache network and buffer state before training."""
        if network_type == 'sorting':
            network = self.sorting_net
            buffer_state = [list(buffer) for buffer in self.sorting_class_buffers]
        elif network_type == 'gesture':
            network = self.gesture_net
            buffer_state = (list(self.gesture_class_0), list(self.gesture_class_1))
        else:  # complex_gesture
            network = self.complex_gesture_net
            buffer_state = [list(buffer) for buffer in self.complex_gesture_buffers]
            
        network_state = NetworkState.from_network(network)
        
        self.last_training_samples[network_type] = TrainingSample(
            input_data=copy.deepcopy(input_data),
            label=label,
            network_state=network_state,
            buffer_state=buffer_state
        )

    def _restore_training_state(self, network_type: str) -> Optional[TrainingSample]:
        """Restore network and buffer state from cache."""
        sample = self.last_training_samples[network_type]
        if sample is None:
            return None
            
        if network_type == 'sorting':
            network = self.sorting_net
            self.sorting_class_buffers = [deque(buffer, maxlen=self.buffer_size) 
                                        for buffer in sample.buffer_state]
        elif network_type == 'gesture':
            network = self.gesture_net
            self.gesture_class_0 = deque(sample.buffer_state[0], maxlen=self.buffer_size)
            self.gesture_class_1 = deque(sample.buffer_state[1], maxlen=self.buffer_size)
        else:  # complex_gesture
            network = self.complex_gesture_net
            self.complex_gesture_buffers = [deque(buffer, maxlen=self.buffer_size) 
                                          for buffer in sample.buffer_state]
            
        sample.network_state.apply_to_network(network)
        return sample

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
            cv_image = self.bridge.imgmsg_to_cv2(
                request.image, 
                desired_encoding='rgb8'
            )
            image_tensor = self._preprocess_image(cv_image)
            
            with torch.no_grad():
                output = self.sorting_net(image_tensor)
                prediction = output.numpy()[0]
                confidence = float(torch.max(output).item())
            
            response.prediction = float(np.argmax(prediction))
            
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

            sequence_tensor = self.process_sensor_data(self.combined_sequence)
            if sequence_tensor is None:
                response.prediction = -1.0
                return response

            normalized_sequence = self.normalize_sequence(sequence_tensor)
            
            with torch.no_grad():
                output = self.gesture_net(normalized_sequence)
                prediction = float(output.item())
                confidence = float(output.item())
            
            response.prediction = prediction
            
            self.get_logger().info(
                f'Binary gesture prediction: {prediction:.3f} '
                f'(confidence: {confidence:.3f})'
            )
            
            self.last_inference_sequence = self.combined_sequence.copy()
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

            sequence_tensor = self.process_sensor_data(self.combined_sequence)
            if sequence_tensor is None:
                response.prediction = -1.0
                return response

            normalized_sequence = self.normalize_sequence(sequence_tensor)
            
            with torch.no_grad():
                output = self.complex_gesture_net(normalized_sequence)
                prediction = output.numpy()[0]
                confidence = float(torch.max(output).item())
            
            response.prediction = float(np.argmax(prediction))
            
            self.get_logger().info(
                f'Complex gesture prediction: {response.prediction} '
                f'(confidence: {confidence:.3f})'
            )
            
            self.last_inference_sequence = self.combined_sequence.copy()
            return response
            
        except Exception as e:
            self.get_logger().error(f'Error in complex gesture prediction: {str(e)}')
            response.prediction = -1.0
            return response

    def train_sorting_callback(self, request, response):
        """Handle training requests for the complex sorting network."""
        try:
            # Convert image and cache state
            cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='rgb8')
            image_tensor = self._preprocess_image(cv_image)
            
            # Cache current state
            self._cache_training_state('sorting', request.image, request.label)

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

                # Save training data
                self.data_saver.save_image_data(
                    image_tensor, 
                    request.label,
                    confidence
                )
                
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

            # Cache current state
            self._cache_training_state('gesture', self.last_inference_sequence, request.label)

            # Process sequence
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

            # Save training data
            self.data_saver.save_gesture_data(
                self.last_inference_sequence,
                request.label,
                confidence
            )

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

            # Cache current state
            self._cache_training_state('complex_gesture', 
                                     self.last_inference_sequence, 
                                     request.label)

            # Process sequence
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

    def correct_sorting_callback(self, request, response):
        """Handle correction requests for sorting network."""
        try:
            # Restore previous state
            sample = self._restore_training_state('sorting')
            if sample is None:
                response.success = False
                response.message = "No previous training sample found"
                return response

            if request.old_label != sample.label:
                response.success = False
                response.message = "Old label doesn't match last training"
                return response

            # Create new training request with corrected label
            new_request = SortNet.Request()
            new_request.image = sample.input_data
            new_request.label = request.new_label

            # Retrain with corrected label
            train_response = self.train_sorting_callback(new_request, SortNet.Response())
            success = train_response.prediction >= 0

            # Log correction
            self.logger.log_correction('complex_sorting', 
                                     request.old_label,
                                     request.new_label, 
                                     success)

            response.success = success
            response.message = "Correction applied successfully"
            return response

        except Exception as e:
            response.success = False
            response.message = f"Error applying correction: {str(e)}"
            return response

    def correct_gesture_callback(self, request, response):
        """Handle correction requests for gesture network."""
        try:
            # Restore previous state
            sample = self._restore_training_state('gesture')
            if sample is None:
                response.success = False
                response.message = "No previous training sample found"
                return response

            if request.old_label != sample.label:
                response.success = False
                response.message = "Old label doesn't match last training"
                return response

            # Recreate last sequence
            self.last_inference_sequence = sample.input_data.copy()

            # Create new training request with corrected label
            new_request = GestNet.Request()
            new_request.label = request.new_label

            # Retrain with corrected label
            train_response = self.train_gesture_callback(new_request, GestNet.Response())
            success = train_response.prediction >= 0

            # Log correction
            self.logger.log_correction('binary_gesture', 
                                     request.old_label,
                                     request.new_label, 
                                     success)

            response.success = success
            response.message = "Correction applied successfully"
            return response

        except Exception as e:
            response.success = False
            response.message = f"Error applying correction: {str(e)}"
            return response

    def correct_complex_gesture_callback(self, request, response):
        """Handle correction requests for complex gesture network."""
        try:
            # Restore previous state
            sample = self._restore_training_state('complex_gesture')
            if sample is None:
                response.success = False
                response.message = "No previous training sample found"
                return response

            if request.old_label != sample.label:
                response.success = False
                response.message = "Old label doesn't match last training"
                return response

            # Recreate last sequence
            self.last_inference_sequence = sample.input_data.copy()

            # Create new training request with corrected label
            new_request = GestNet.Request()
            new_request.label = request.new_label

            # Retrain with corrected label
            train_response = self.train_complex_gesture_callback(
                new_request, GestNet.Response())
            success = train_response.prediction >= 0

            # Log correction
            self.logger.log_correction('complex_gesture', 
                                     request.old_label,
                                     request.new_label, 
                                     success)

            response.success = success
            response.message = "Correction applied successfully"
            return response

        except Exception as e:
            response.success = False
            response.message = f"Error applying correction: {str(e)}"
            return response

    def save_sorting_callback(self, request, response):
        """Save sorting network to file."""
        try:
            timestamp = self.get_clock().now().to_msg().sec
            
            if request.filename:
                filepath = f"{self.save_dir}/{request.filename}"
            else:
                filepath = f"{self.save_dir}/sorting_net_{timestamp}.pt"
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
        class_samples = [list(buffer) for buffer in class_buffers]
        lengths = [len(samples) for samples in class_samples]
        non_empty_indices = [i for i, length in enumerate(lengths) if length > 0]
        
        if not non_empty_indices:
            self.get_logger().warn("No samples available for training")
            return [], []
        
        target_size = max(len(class_samples[i]) for i in non_empty_indices)
        
        balanced_samples = []
        balanced_labels = []
        
        for class_idx in non_empty_indices:
            samples = class_samples[class_idx]
            if len(samples) < target_size:
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