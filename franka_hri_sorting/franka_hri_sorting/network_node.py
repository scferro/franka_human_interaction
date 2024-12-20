"""
Neural network management node for training and inference with sorting and gesture networks.

This node manages multiple neural networks for sorting and gesture recognition,
handles training data collection, performs inference, and provides correction capabilities.

PUBLISHERS:
    None

SUBSCRIBERS:
    + /sensor_data/modality_48 (std_msgs/Float32MultiArray) - Sensor data from modality 48
    + /sensor_data/modality_51 (std_msgs/Float32MultiArray) - Sensor data from modality 51

SERVICES:
    + /train_sorting (franka_hri_interfaces/SortNet) - Train sorting network with new data
    + /train_gesture (franka_hri_interfaces/GestNet) - Train gesture network with new data
    + /train_complex_gesture (franka_hri_interfaces/GestNet) - Train complex gesture network with new data
    + /get_sorting_prediction (franka_hri_interfaces/SortNet) - Get prediction from sorting network
    + /get_gesture_prediction (franka_hri_interfaces/GestNet) - Get prediction from gesture network
    + /get_complex_gesture_prediction (franka_hri_interfaces/GestNet) - Get prediction from complex gesture network
    + /save_sorting_network (franka_hri_interfaces/SaveModel) - Save sorting network to file
    + /save_gesture_network (franka_hri_interfaces/SaveModel) - Save gesture network to file
    + /save_complex_gesture_network (franka_hri_interfaces/SaveModel) - Save complex gesture network to file
    + /correct_sorting (franka_hri_interfaces/CorrectionService) - Apply corrections to sorting network
    + /correct_gesture (franka_hri_interfaces/CorrectionService) - Apply corrections to gesture network
    + /correct_complex_gesture (franka_hri_interfaces/CorrectionService) - Apply corrections to complex gesture network
    + /collect_norm_values (std_srvs/Trigger) - Collect normalization values for sensor data

PARAMETERS:
    + buffer_size (int) - Size of prediction buffer for sorting network
    + buffer_size_gest (int) - Size of prediction buffer for gesture networks
    + sequence_length (int) - Length of sequence for gesture recognition
    + sorting_model_path (string) - Path to sorting network model file
    + gesture_model_path (string) - Path to gesture network model file
    + complex_gesture_model_path (string) - Path to complex gesture network model file
    + save_directory (string) - Directory for saving network models
    + log_directory (string) - Directory for saving network logs
    + save_training_data (bool) - Whether to save training data
    + training_data_dir (string) - Directory for saving training data
    + norm_values_path (string) - Path to normalization values file
"""


import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
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
import csv
from pathlib import Path
from datetime import datetime

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
    label: int  # The true/corrected label
    original_label: int  # The original predicted label
    network_state: NetworkState
    buffer_state: any  
    prediction: float
    confidence: any

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
        """Update existing log entry with correction information."""
        with self.lock:
            timestamp = datetime.now().timestamp()
            
            # Get the appropriate log file
            log_file = self._get_log_file(network_type)
            
            # Read all lines from the log file
            lines = []
            with open(log_file, 'r', newline='') as f:
                reader = csv.reader(f)
                lines = list(reader)
                
            if len(lines) < 2:  # Just header or empty file
                self.get_logger().warn(f"No entries to update in {log_file}")
                return
                
            # Update the last entry with the new label and recalculate correctness
            last_entry = lines[-1]
            
            # Parse prediction based on network type
            if network_type in ['complex_sorting', 'complex_gesture']:
                # Convert string representation of array back to numpy array
                prediction_str = last_entry[1].strip('[]').split()
                prediction = np.array([float(x) for x in prediction_str])
            else:
                prediction = float(last_entry[1])
            
            # Update true label with the new label
            last_entry[3] = str(new_label)
            
            # Recalculate was_correct based on network type
            if network_type in ['complex_sorting', 'complex_gesture']:
                was_correct = (np.argmax(prediction) == new_label)
            else:
                was_correct = (prediction >= 0.5) == (new_label >= 0.5)
            last_entry[4] = str(was_correct)
            
            # Write back all lines including the updated one
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(lines)
            
            # Update statistics
            if success:
                self.stats[network_type]['corrections'] += 1
                # Update confusion matrix - subtract old prediction and add new one
                if network_type in ['complex_sorting', 'complex_gesture']:
                    pred_class = np.argmax(prediction)
                    self.stats[network_type]['confusion_matrix'][old_label][pred_class] -= 1
                    self.stats[network_type]['confusion_matrix'][new_label][pred_class] += 1
                else:
                    pred_class = 1 if prediction >= 0.5 else 0
                    old_class = 1 if old_label >= 0.5 else 0
                    new_class = 1 if new_label >= 0.5 else 0
                    self.stats[network_type]['confusion_matrix'][old_class][pred_class] -= 1
                    self.stats[network_type]['confusion_matrix'][new_class][pred_class] += 1
                
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
    """Data saver that stores each gesture as an individual file."""
    def __init__(self, base_dir: str, enabled: bool = True):
        self.enabled = enabled
        
        if not enabled:
            return
            
        # Create base directory structure
        self.base_dir = Path(base_dir)
        self.image_dir = self.base_dir / 'images'
        self.gesture_dir = self.base_dir / 'gestures'
        
        # Create subdirectories for different gesture types
        self.binary_gesture_dir = self.gesture_dir / 'binary'
        self.complex_gesture_dir = self.gesture_dir / 'complex'
        
        # Ensure all directories exist
        for directory in [
            self.image_dir, 
            self.binary_gesture_dir,
            self.complex_gesture_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped image file for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.image_file = self.image_dir / f"image_data_{timestamp}.csv"
        
        # Initialize image file with headers
        self._init_image_file()
        
    def _init_image_file(self):
        """Initialize the image CSV file with headers."""
        image_headers = ['timestamp', 'label', 'confidence', 'filename']
        with open(self.image_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(image_headers)

    def save_image_data(self, image_tensor: torch.Tensor, label: int, confidence: float = None):
        """Save image data as PNG file with proper tensor to image conversion."""
        if not self.enabled:
            return
            
        try:
            # Create a timestamp-based filename
            timestamp = datetime.now().isoformat()
            image_name = f"image_{timestamp}.png"
            image_path = self.image_dir / image_name
            
            # First denormalize the tensor
            denorm = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
            denormalized = denorm(image_tensor[0])
            
            # Convert to PIL Image and save
            image = transforms.ToPILImage()(denormalized)
            image.save(str(image_path))
            
            # Log metadata to CSV
            with open(self.image_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    label,
                    confidence if confidence is not None else '',
                    image_name
                ])
        except Exception as e:
            print(f"Error saving image data: {e}")
    
    def save_gesture_data(self, sequence: list, label: int, confidence: float = None, 
                         gesture_type: str = 'binary'):
        """Save gesture sequence data as individual CSV files."""
        if not self.enabled:
            return
            
        try:
            # Generate timestamp for unique identification
            timestamp = datetime.now().isoformat()
            
            # Determine gesture type and directory based on the gesture_type parameter
            base_dir = self.binary_gesture_dir if gesture_type == 'binary' else self.complex_gesture_dir
            
            # Create filename for this specific gesture
            gesture_file = base_dir / f"gesture_{timestamp}.csv"
            
            # Convert sequence to numpy array for easier handling
            sequence_array = np.array(sequence)
            
            # Save sequence data as CSV with metadata in header row
            with open(gesture_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write metadata as header
                writer.writerow(['timestamp', 'label', 'confidence', 'sequence_length', 'feature_count'])
                writer.writerow([timestamp, label, confidence, len(sequence), sequence_array.shape[1]])
                # Write column headers for the sequence data
                writer.writerow(['timestamp'] + [f'feature_{i}' for i in range(sequence_array.shape[1])])
                # Write the actual sequence data
                for i, datapoint in enumerate(sequence_array):
                    writer.writerow([f't{i}'] + list(datapoint))
                    
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
        self.declare_parameter('buffer_size_gest', 100)
        self.declare_parameter('sequence_length', 20)
        self.declare_parameter('sorting_model_path', '')
        self.declare_parameter('gesture_model_path', '')
        self.declare_parameter('complex_gesture_model_path', '')
        self.declare_parameter('save_directory', '/home/user/saved_models')
        self.declare_parameter('log_directory', '/home/user/network_logs')
        self.declare_parameter('save_training_data', True)
        self.declare_parameter('training_data_dir', '/home/user/training_data')
        self.declare_parameter('norm_values_path', '/home/user/network_logs/norm_values.csv')
        
        self.buffer_size = self.get_parameter('buffer_size').value
        self.buffer_size_gest = self.get_parameter('buffer_size_gest').value
        self.sequence_length = self.get_parameter('sequence_length').value
        self.sorting_model_path = self.get_parameter('sorting_model_path').value
        self.gesture_model_path = self.get_parameter('gesture_model_path').value
        self.complex_gesture_model_path = self.get_parameter('complex_gesture_model_path').value
        self.save_dir = self.get_parameter('save_directory').value
        self.log_dir = self.get_parameter('log_directory').value
        save_training = self.get_parameter('save_training_data').value
        training_data_dir = self.get_parameter('training_data_dir').value
        self.norm_values_path = self.get_parameter('norm_values_path').value
        
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
        
        self.gesture_class_0 = deque(maxlen=self.buffer_size_gest)
        self.gesture_class_1 = deque(maxlen=self.buffer_size_gest)
        
        self.complex_gesture_buffers = [
            deque(maxlen=self.buffer_size_gest) for _ in range(4)
        ]
        
        self.mod48_buffer = None
        self.combined_sequence = []
        self.last_inference_sequence = []
        self.norm_values = np.ones(50)
        self._load_norm_values()

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
        self.norm_trigger = self.create_service( Trigger, 'collect_norm_values', self.handle_norm_trigger)

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

    def _load_norm_values(self):
        """Load normalization values from CSV."""
        try:
            if os.path.exists(self.norm_values_path):
                self.norm_values = np.loadtxt(self.norm_values_path)
                self.get_logger().info(f"Loaded normalization values: {self.norm_values}")
        except Exception as e:
            self.get_logger().error(f"Error loading normalization values: {e}")

    def _save_norm_values(self):
        """Save normalization values to CSV."""
        try:
            os.makedirs(os.path.dirname(self.norm_values_path), exist_ok=True)
            np.savetxt(self.norm_values_path, self.norm_values)
            self.get_logger().info(f"Saved normalization values: {self.norm_values}")
        except Exception as e:
            self.get_logger().error(f"Error saving normalization values: {e}")

    def handle_norm_trigger(self, request, response):
        """Process current sequence to extract norm values."""
        try:
            if not self.combined_sequence:
                response.success = False 
                response.message = "No sequence data available"
                return response
                
            sequence_array = np.array(self.combined_sequence)  # Shape: [seq_len, 200]
            # Get absolute maximum values across sequence for each feature
            self.norm_values = np.abs(sequence_array).max(axis=0)  # Shape: [200]
            self._save_norm_values()
            
            response.success = True
            response.message = f"Updated normalization values: {self.norm_values}"
            return response
            
        except Exception as e:
            response.success = False
            response.message = f"Error: {str(e)}"
            return response

    def _cache_training_state(self, network_type: str, input_data: any, label: int, original_label: int = None):
        """Cache network and buffer state before training."""
        if network_type == 'sorting':
            network = self.sorting_net
            buffer_state = [list(buffer) for buffer in self.sorting_class_buffers]
            
            # Convert tensor to ROS Image message if it isn't already
            if isinstance(input_data, torch.Tensor):
                try:
                    # Denormalize the image tensor
                    denorm = transforms.Normalize(
                        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                        std=[1/0.229, 1/0.224, 1/0.225]
                    )
                    denormalized = denorm(input_data[0])
                    
                    # Convert to numpy array and then to CV2 image
                    np_image = denormalized.permute(1, 2, 0).numpy()
                    np_image = (np_image * 255).astype(np.uint8)
                    
                    # Convert BGR to RGB
                    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
                    
                    # Convert to ROS message
                    input_data = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
                except Exception as e:
                    self.get_logger().error(f"Error converting tensor to ROS message: {e}")
                    return

            with torch.no_grad():
                if isinstance(input_data, torch.Tensor):
                    output = network(input_data)
                else:
                    # Convert ROS Image to tensor for prediction
                    cv_image = self.bridge.imgmsg_to_cv2(input_data, desired_encoding='rgb8')
                    image_tensor = self._preprocess_image(cv_image)
                    output = network(image_tensor)
                prediction = output.numpy()[0]
                confidence = float(torch.max(output).item())

        elif network_type == 'gesture':
            network = self.gesture_net
            buffer_state = (list(self.gesture_class_0), list(self.gesture_class_1))
            # For gesture network, prediction is a single value
            with torch.no_grad():
                output = network(self.normalize_sequence(self.process_sensor_data(input_data)))
                prediction = float(output.item())
                confidence = float(output.item())

        else:  # complex_gesture
            network = self.complex_gesture_net
            buffer_state = [list(buffer) for buffer in self.complex_gesture_buffers]
            with torch.no_grad():
                output = network(self.normalize_sequence(self.process_sensor_data(input_data)))
                prediction = output.numpy()[0]
                confidence = float(torch.max(output).item())
                
        network_state = NetworkState.from_network(network)
        
        self.last_training_samples[network_type] = TrainingSample(
            input_data=input_data,
            label=label,
            original_label=original_label if original_label is not None else label,
            network_state=network_state,
            buffer_state=buffer_state,
            prediction=prediction,
            confidence=confidence
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
            self.gesture_class_0 = deque(sample.buffer_state[0], maxlen=self.buffer_size_gest)
            self.gesture_class_1 = deque(sample.buffer_state[1], maxlen=self.buffer_size_gest)
        else:  # complex_gesture
            network = self.complex_gesture_net
            self.complex_gesture_buffers = [deque(buffer, maxlen=self.buffer_size_gest) 
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
        try:
            cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='rgb8')
            image_tensor = self._preprocess_image(cv_image)
            
            with torch.no_grad():
                output = self.sorting_net(image_tensor)
                prediction = output.numpy()[0]
                confidence = float(torch.max(output).item())
            
            # Get the predicted class
            predicted_label = int(np.argmax(prediction))
            
            # Cache with both the predicted label and itself as original
            self._cache_training_state('sorting', request.image, predicted_label, predicted_label)
            
            prediction_id = f"sort_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            
            response.prediction = float(predicted_label)
            response.confidence = confidence
            response.prediction_id = prediction_id
            
            return response
        
        except Exception as e:
            self.get_logger().error(f'Error in sorting prediction: {str(e)}')
            response.prediction = -1.0
            response.confidence = 0.0
            response.prediction_id = ""
            return response

    def get_gesture_prediction_callback(self, request, response):
        """Handle prediction requests for the binary gesture network."""
        try:
            if len(self.combined_sequence) < self.sequence_length:
                response.prediction = -1.0
                response.confidence = 0.0
                response.prediction_id = ""
                return response

            sequence_tensor = self.process_sensor_data(self.combined_sequence)
            if sequence_tensor is None:
                response.prediction = -1.0
                response.confidence = 0.0
                response.prediction_id = ""
                return response

            normalized_sequence = self.normalize_sequence(sequence_tensor)
            
            with torch.no_grad():
                output = self.gesture_net(normalized_sequence)
                prediction = float(output.item())
                confidence = float(output.item())
            
            # Generate unique prediction ID
            prediction_id = f"gest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            
            response.prediction = prediction
            response.confidence = confidence
            response.prediction_id = prediction_id
            
            self.last_inference_sequence = self.combined_sequence.copy()
            return response
            
        except Exception as e:
            self.get_logger().error(f'Error in gesture prediction: {str(e)}')
            response.prediction = -1.0
            response.confidence = 0.0
            response.prediction_id = ""
            return response

    def get_complex_gesture_prediction_callback(self, request, response):
        """Handle prediction requests for the complex gesture network."""
        try:
            if len(self.combined_sequence) < self.sequence_length:
                response.prediction = -1.0
                response.confidence = 0.0
                response.prediction_id = ""
                return response

            sequence_tensor = self.process_sensor_data(self.combined_sequence)
            if sequence_tensor is None:
                response.prediction = -1.0
                response.confidence = 0.0
                response.prediction_id = ""
                return response

            normalized_sequence = self.normalize_sequence(sequence_tensor)
            
            with torch.no_grad():
                output = self.complex_gesture_net(normalized_sequence)
                prediction = output.numpy()[0]
                confidence = float(torch.max(output).item())
            
            # Generate unique prediction ID
            prediction_id = f"cgest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            
            response.prediction = float(np.argmax(prediction))
            response.confidence = confidence
            response.prediction_id = prediction_id
            
            self.last_inference_sequence = self.combined_sequence.copy()
            return response
            
        except Exception as e:
            self.get_logger().error(f'Error in complex gesture prediction: {str(e)}')
            response.prediction = -1.0
            response.confidence = 0.0
            response.prediction_id = ""
            return response
        
    def train_sorting_callback(self, request, response):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='rgb8')
            image_tensor = self._preprocess_image(cv_image)

            # Update the cached state with the correct label
            self._cache_training_state('sorting', request.image, request.label)
            
            # Get the stored prediction info for saving
            last_sample = self.last_training_samples['sorting']
            if last_sample:
                # Log the prediction and feedback
                self.logger.log_prediction_and_feedback(
                    'complex_sorting',
                    last_sample.prediction,  # Use stored prediction array
                    last_sample.confidence,  # Use stored confidence
                    request.label  # True label from request
                )

            # Add to appropriate class buffer
            class_idx = int(request.label)
            if 0 <= class_idx < 4:
                self.sorting_class_buffers[class_idx].append(image_tensor)
            else:
                raise ValueError(f"Invalid class index: {class_idx}")

            # Save the training data before network training
            self.data_saver.save_image_data(
                image_tensor,
                request.label,
                last_sample.confidence if last_sample else None
            )

            # Get balanced training data and train
            images, labels = self._balance_multi_class_data(self.sorting_class_buffers)
            if images:
                self.sorting_net.train_network(images, labels)
                self.get_logger().info(
                    f'Trained sorting network with {len(images)} images'
                )
                    
            response.prediction = 0.0
            response.confidence = request.confidence
            response.prediction_id = f"train_sort_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            return response

        except Exception as e:
            self.get_logger().error(f"Error training sorting network: {str(e)}")
            response.prediction = -1.0
            response.confidence = 0.0
            response.prediction_id = ""
            return response

    def train_gesture_callback(self, request, response):
        """Handle training requests for the binary gesture network."""
        try:
            if len(self.last_inference_sequence) < self.sequence_length:
                self.get_logger().warn('Not enough data points for training')
                response.prediction = -1.0
                response.confidence = 0.0
                response.prediction_id = ""
                return response

            # Cache current state
            self._cache_training_state('gesture', self.last_inference_sequence, request.label)

            # Process sequence
            sequence_tensor = self.process_sensor_data(self.last_inference_sequence)
            if sequence_tensor is None:
                response.prediction = -1.0
                response.confidence = 0.0
                response.prediction_id = ""
                return response

            normalized_sequence = self.normalize_sequence(sequence_tensor)

            # Add to appropriate buffer
            if request.label == 0:
                self.gesture_class_0.append(normalized_sequence)
            else:
                self.gesture_class_1.append(normalized_sequence)

            last_sample = self.last_training_samples['gesture']

            # Save training data using stored prediction values
            self.data_saver.save_gesture_data(
                self.last_inference_sequence,
                request.label,
                last_sample.confidence
            )

            # Get balanced training data
            sequences, labels = self._balance_data(
                self.gesture_class_0,
                self.gesture_class_1
            )

            if sequences:
                # Train network
                self.gesture_net.train_network(sequences, labels)
                
                # Log prediction and feedback using provided confidence
                self.logger.log_prediction_and_feedback(
                    'binary_gesture',
                    last_sample.prediction,  # Use stored prediction
                    last_sample.confidence,  # Use stored confidence
                    request.label
                )
                
                self.get_logger().info(
                    f'Trained gesture network on {len(sequences)} sequences'
                )

            # Generate prediction ID for response
            prediction_id = f"train_gest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            
            response.prediction = 0.0
            response.confidence = request.confidence
            response.prediction_id = prediction_id
            return response

        except Exception as e:
            self.get_logger().error(f'Error in train_gesture: {str(e)}')
            response.prediction = -1.0
            response.confidence = 0.0
            response.prediction_id = ""
            return response
        
    def train_complex_gesture_callback(self, request, response):
        """Handle training requests for the complex gesture network."""
        try:
            if len(self.last_inference_sequence) < self.sequence_length:
                self.get_logger().warn('Not enough data points for training')
                response.prediction = -1.0
                response.confidence = 0.0
                response.prediction_id = ""
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
                response.confidence = 0.0
                response.prediction_id = ""
                return response

            normalized_sequence = self.normalize_sequence(sequence_tensor)

            # Add to appropriate class buffer
            class_idx = int(request.label)
            if 0 <= class_idx < 4:
                self.complex_gesture_buffers[class_idx].append(
                    normalized_sequence)
            else:
                raise ValueError(f"Invalid class index: {class_idx}")

            # Get the stored prediction from the last training sample
            last_sample = self.last_training_samples['complex_gesture']
            if last_sample is None:
                self.get_logger().warn('No previous prediction found')
                response.prediction = -1.0
                response.confidence = 0.0
                response.prediction_id = ""
                return response

            # Save training data using stored prediction values
            self.data_saver.save_gesture_data(
                self.last_inference_sequence,
                request.label,
                last_sample.confidence,
                "complex"
            )

            # Get balanced training data
            sequences, labels = self._balance_multi_class_data(
                self.complex_gesture_buffers)

            if sequences:
                # Train network
                self.complex_gesture_net.train_network(sequences, labels)
                
                # Log prediction and feedback using stored values
                self.logger.log_prediction_and_feedback(
                    'complex_gesture',
                    last_sample.prediction,  # Use stored prediction
                    last_sample.confidence,  # Use stored confidence
                    request.label
                )
                
                self.get_logger().info(
                    f'Trained complex gesture network on {len(sequences)} sequences'
                )

            # Generate prediction ID for response
            prediction_id = f"train_cgest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            
            response.prediction = 0.0
            response.confidence = request.confidence
            response.prediction_id = prediction_id
            return response

        except Exception as e:
            self.get_logger().error(
                f'Error in train_complex_gesture: {str(e)}')
            response.prediction = -1.0
            response.confidence = 0.0
            response.prediction_id = ""
            return response

    def correct_sorting_callback(self, request, response):
        """Handle correction requests for sorting network."""
        try:
            # Get previous state (but don't restore it)
            sample = self.last_training_samples['sorting']
            if sample is None:
                response.success = False
                response.message = "No previous training sample found"
                return response

            # Convert ROS Image to tensor for buffer storage
            cv_image = self.bridge.imgmsg_to_cv2(sample.input_data, desired_encoding='rgb8')
            image_tensor = self._preprocess_image(cv_image)

            # Use the original predicted label for removal
            old_class_idx = request.old_label
            new_class_idx = request.new_label

            # Log buffer states before correction
            self.get_logger().info(f"Before correction - Buffer sizes: {[len(buf) for buf in self.sorting_class_buffers]}")
            self.get_logger().info(f"Removing from original class {old_class_idx}, adding to new class {new_class_idx}")

            # Remove from old buffer
            old_buffer = self.sorting_class_buffers[old_class_idx]
            if old_buffer:
                old_buffer.pop()  # Remove the last tensor in the buffer
                self.get_logger().info(f"Removed tensor from class {old_class_idx}")

            # Log buffer states after correction
            self.get_logger().info(f"After correction - Buffer sizes: {[len(buf) for buf in self.sorting_class_buffers]}")

            # Create new training request with corrected label
            new_request = SortNet.Request()
            new_request.image = sample.input_data
            new_request.label = request.new_label

            # Train with corrected label
            train_response = self.train_sorting_callback(new_request, SortNet.Response())

            # Log correction
            self.logger.log_correction('complex_sorting', 
                                    old_class_idx,
                                    request.new_label, 
                                    False)

            response.success = True
            response.message = "Correction applied successfully"
            return response

        except Exception as e:
            self.get_logger().error(f"Error in correct_sorting_callback: {str(e)}")
            response.success = False
            response.message = f"Error applying correction: {str(e)}"
            return response

    def correct_gesture_callback(self, request, response):
        """Handle correction requests for gesture network."""
        try:
            # Get previous state (but don't restore it)
            sample = self.last_training_samples['gesture']
            if sample is None:
                response.success = False
                response.message = "No previous training sample found"
                return response

            # Process sequence for buffer storage
            sequence_tensor = self.process_sensor_data(sample.input_data)
            normalized_sequence = self.normalize_sequence(sequence_tensor)

            # Remove from old buffer
            if sample.label == 0:
                for seq in self.gesture_class_0:
                    if torch.equal(seq, normalized_sequence):
                        self.gesture_class_0.remove(seq)
                        break
            else:
                for seq in self.gesture_class_1:
                    if torch.equal(seq, normalized_sequence):
                        self.gesture_class_1.remove(seq)
                        break

            # Add to new buffer
            if request.new_label == 0:
                self.gesture_class_0.append(normalized_sequence)
            else:
                self.gesture_class_1.append(normalized_sequence)

            # Set up sequence for retraining
            self.last_inference_sequence = sample.input_data.copy()

            # Create new training request with corrected label
            new_request = GestNet.Request()
            new_request.label = request.new_label

            # Train with corrected label
            train_response = self.train_gesture_callback(new_request, GestNet.Response())
            success = train_response.prediction >= 0

            # Log correction
            self.logger.log_correction('binary_gesture', 
                                    sample.label,
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
            # Get previous state (but don't restore it)
            sample = self.last_training_samples['complex_gesture']
            if sample is None:
                response.success = False
                response.message = "No previous training sample found"
                return response

            # Process sequence for buffer storage
            sequence_tensor = self.process_sensor_data(sample.input_data)
            if sequence_tensor is None:
                response.success = False
                response.message = "Failed to process sequence data"
                return response

            normalized_sequence = self.normalize_sequence(sequence_tensor)

            old_class_idx = request.old_label
            new_class_idx = request.new_label

            # Remove from old buffer
            old_buffer = self.complex_gesture_buffers[old_class_idx]
            for seq in old_buffer:
                if torch.equal(seq, normalized_sequence):
                    old_buffer.remove(seq)
                    break

            # Add to new buffer
            self.complex_gesture_buffers[new_class_idx].append(normalized_sequence)

            # Set up sequence for retraining
            self.last_inference_sequence = sample.input_data.copy()

            # Create new training request with corrected label
            new_request = GestNet.Request()
            new_request.label = request.new_label

            # Train with corrected label
            train_response = self.train_complex_gesture_callback(
                new_request, GestNet.Response())
            success = train_response.prediction >= 0

            # Log correction
            self.logger.log_correction('complex_gesture', 
                                    sample.label,
                                    request.new_label, 
                                    success)

            response.success = success
            response.message = "Correction applied successfully"
            return response

        except Exception as e:
            self.get_logger().error(f"Error in correct_complex_gesture: {str(e)}")
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
            day_offset = -18
            timestamp = self.get_clock().now().to_msg().sec + (day_offset * 86400)
            
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
        """Normalize sequence using saved normalization values."""
        try:
            # Convert norm values to tensor and match device/dtype
            norm_tensor = torch.from_numpy(self.norm_values).to(
                device=tensor_data.device,
                dtype=tensor_data.dtype
            )
            # Normalize using maximum values
            norm_tensor = norm_tensor.view(1, -1, 1)
            normalized = tensor_data / norm_tensor
            return normalized
        except Exception as e:
            self.get_logger().error(f"Error normalizing sequence: {e}")
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
            
            # Flatten and combine features to match 200 total features
            combined_features = mod51_features.flatten()
            if self.mod48_buffer is not None:
                mod48_features = self.mod48_buffer.flatten()
                combined_features = np.concatenate([mod48_features, combined_features])
            
            self.combined_sequence.append(combined_features)

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