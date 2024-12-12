"""
Node for monitoring human interaction with blocks and updating block positions.

This node tracks block positions using a D435i camera, detects hand interactions, 
and updates block stack tracking based on observed movements.

PUBLISHERS:
    + /hands_detected (std_msgs/Bool) - Indicates if hands are currently detected in frame
    + /monitoring_visualization (sensor_msgs/Image) - Visualization of monitoring regions

SUBSCRIBERS:
    + {camera_info_topic} (sensor_msgs/CameraInfo) - Camera calibration information
    + {color_topic} (sensor_msgs/Image) - RGB image from camera
    + {depth_topic} (sensor_msgs/Image) - Depth image from camera
    + /block_placement_info (franka_hri_interfaces/BlockPlacementInfo) - Information about block placements
    + /blocks (visualization_msgs/MarkerArray) - Current block marker positions

SERVICES:
    + /localize_d435 (std_srvs/Trigger) - Service to localize D435 camera using AprilTag

SERVICE CLIENTS:
    + /update_piles (franka_hri_interfaces/MoveBlock) - Updates block stack tracking
    + /correct_sorting (franka_hri_interfaces/CorrectionService) - Correction service for sorting
    + /correct_gesture (franka_hri_interfaces/CorrectionService) - Correction service for gestures
    + /correct_complex_gesture (franka_hri_interfaces/CorrectionService) - Correction service for complex gestures

PARAMETERS:
    + calibration_file (string) - Path to camera calibration file
    + calibration_timeout (float) - Timeout for camera calibration in seconds
    + camera_info_topic (string) - Topic for camera calibration info
    + depth_topic (string) - Topic for depth image
    + color_topic (string) - Topic for color image
    + visualization_rate (float) - Rate in Hz for visualization updates
    + monitoring_window_size (int) - Size of monitoring window in pixels
    + depth_change_threshold (float) - Threshold for detecting depth changes in meters
    + min_valid_depth_ratio (float) - Minimum ratio of valid depth pixels

BROADCASTS:
    + world -> d435i_link - Transform from world frame to D435 camera
"""


import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer, TransformBroadcaster, StaticTransformBroadcaster
import tf_transformations
from geometry_msgs.msg import TransformStamped, PointStamped, Pose
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import MarkerArray
from franka_hri_interfaces.msg import BlockPlacementInfo
from franka_hri_interfaces.srv import UpdateMarkers
from franka_hri_interfaces.srv import MoveBlock
from franka_hri_interfaces.srv import CorrectionService
import mediapipe as mp
import numpy as np
import cv2
import csv
import os
import time
from tf2_geometry_msgs import do_transform_point

class HumanInteraction(Node):
    def __init__(self):
        super().__init__('human_interaction')
        
        # Declare parameters
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('calibration_timeout', 10.0)
        self.declare_parameter('camera_info_topic', '/camera/d435i/aligned_depth_to_color/camera_info')
        self.declare_parameter('depth_topic', '/camera/d435i/aligned_depth_to_color/image_raw')
        self.declare_parameter('color_topic', '/camera/d435i/color/image_raw')
        self.declare_parameter('visualization_rate', 10.0)  # Hz for visualization updates
        self.declare_parameter('monitoring_window_size', 100)  # Size in pixels
        self.declare_parameter('depth_change_threshold', 0.005)  # In meters
        self.declare_parameter('min_valid_depth_ratio', 0.2)  # Minimum ratio of valid depth pixels

        self.monitoring_window_size = self.get_parameter('monitoring_window_size').value
        self.depth_change_threshold = self.get_parameter('depth_change_threshold').value
        self.min_valid_depth_ratio = self.get_parameter('min_valid_depth_ratio').value
        
        # Initialize camera matrix and monitoring state
        self.camera_matrix = None
        self.current_block_info = None
        self.monitoring_active = True
        self.monitoring_positions = None
        self.current_block_index = None
        self.block_markers = None
        self.monitoring_regions = []  # Store current monitoring regions
        
        # Initialize frame storage
        self.last_color_frame = None
        self.last_depth_frame = None 
        self.pre_interaction_color = None
        self.pre_interaction_depth = None
        
        # Define visualization colors
        self.ORIGINAL_COLOR = (0, 0, 255)    # Red for original position
        self.TARGET_COLOR = (0, 255, 0)      # Green for target positions
        self.ACTIVE_COLOR = (255, 165, 0)    # Orange for regions with detected changes
        
        # Initialize OpenCV bridge and MediaPipe hands
        self.bridge = CvBridge()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        
        # Initialize TF components
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.d435_transform = None
        
        # Create subscribers
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self.camera_info_callback,
            10
        )
        
        self.color_sub = self.create_subscription(
            Image,
            self.get_parameter('color_topic').value,
            self.color_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            self.get_parameter('depth_topic').value,
            self.depth_callback,
            10
        )
        
        self.block_info_sub = self.create_subscription(
            BlockPlacementInfo,
            'block_placement_info',
            self.block_placement_callback,
            10
        )

        self.block_markers_sub = self.create_subscription(
            MarkerArray,
            'blocks',
            self.marker_array_callback,
            10
        )

        # Create hands callback group
        self.hand_callback_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        
        # Create publishers and service clients
        self.hands_detected_pub = self.create_publisher(Bool, 'hands_detected', 10)
        self.monitoring_image_pub = self.create_publisher(Image, 'monitoring_visualization', 10)

        self.blocks_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        self.move_block_client = self.create_client(
            MoveBlock, 
            'update_piles',
            callback_group=self.blocks_callback_group,
            )
        
        # Create a reentrant callback group for service clients
        self.corrections_group = rclpy.callback_groups.ReentrantCallbackGroup()
        # Add correction service clients
        self.sort_correction_client = self.create_client(
            CorrectionService, 
            'correct_sorting',
            callback_group=self.corrections_group
        )
        self.gesture_correction_client = self.create_client(
            CorrectionService, 
            'correct_gesture',
            callback_group=self.corrections_group
        )
        self.complex_gesture_correction_client = self.create_client(
            CorrectionService, 
            'correct_complex_gesture',
            callback_group=self.corrections_group
        )
        
        # Load calibration if exists
        calibration_file = self.get_parameter('calibration_file').value
        if calibration_file and os.path.exists(calibration_file):
            self.load_calibration(calibration_file)
            if self.d435_transform is not None:
                self.broadcast_stored_transform()
        
        # Create localization service
        self.localization_srv = self.create_service(
            Trigger,
            'localize_d435',
            self.localize_camera_callback
        )
        
        # Create timers
        self.tf_timer = self.create_timer(0.1, self.broadcast_stored_transform)
        viz_period = 1.0 / self.get_parameter('visualization_rate').value
        self.viz_timer = self.create_timer(viz_period, self.publish_visualization)
        self.monitoring_timeout = 3.0  # Seconds to keep monitoring after hands leave
        self.reset_timer = None
        
        self.get_logger().info('Human Interaction node initialized')

    def publish_visualization(self):
        """Publish visualization of monitoring regions."""
        if self.last_color_frame is None:
            self.get_logger().debug('No color frame available')
            return
            
        if not self.monitoring_active:
            self.get_logger().debug('Monitoring not active')
            return
            
        if not self.monitoring_regions:
            self.get_logger().debug('No monitoring regions available')
            return
                
        try:
            # Create copy of current frame for visualization
            viz_frame = self.last_color_frame.copy()
            
            # Draw monitoring regions
            if self.monitoring_regions:
                # Draw original position (first region) in red
                region = self.monitoring_regions[0]
                self.draw_monitoring_region(viz_frame, region, self.ORIGINAL_COLOR, "Original")
                
                # Draw target positions in green
                for i, region in enumerate(self.monitoring_regions[1:], 1):
                    self.draw_monitoring_region(viz_frame, region, self.TARGET_COLOR, f"Target {i}")
                    
                    # If we have depth data, check for changes and highlight active regions
                    if self.pre_interaction_depth is not None and self.last_depth_frame is not None:
                        if self.detect_region_change(region, expect_addition=True):
                            self.draw_monitoring_region(viz_frame, region, self.ACTIVE_COLOR, f"Active {i}")
            
            # Convert to ROS message and publish
            viz_msg = self.bridge.cv2_to_imgmsg(viz_frame, encoding="bgr8")
            self.monitoring_image_pub.publish(viz_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing visualization: {str(e)}')

    def draw_monitoring_region(self, image, region, color, label):
        """Draw a single monitoring region with label."""
        # Draw rectangle
        pt1 = (region['x'], region['y'])
        pt2 = (region['x'] + region['width'], region['y'] + region['height'])
        cv2.rectangle(image, pt1, pt2, color, 2)
        
        # Add label above rectangle
        label_pos = (region['x'], region['y'] - 10)
        cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)

    def marker_array_callback(self, msg):
        """Store the current marker array state."""
        self.block_markers = msg

    def camera_info_callback(self, msg):
        """Store camera matrix from camera info message."""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def color_callback(self, msg):
        """Process new color frames and check for hands if monitoring."""
        try:
            self.last_color_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.monitoring_active:
                self.check_for_hands()
        except CvBridgeError as e:
            self.get_logger().error(f'Error converting color image: {str(e)}')

    def depth_callback(self, msg):
        """Store latest depth frame."""
        try:
            self.last_depth_frame = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            self.get_logger().error(f'Error converting depth image: {str(e)}')

    def block_placement_callback(self, msg):
        """Handle new block placement information and start monitoring."""
        self.current_block_info = msg
        self.current_block_index = msg.last_block_index
        self.monitoring_active = True
        
        # Add debug logging
        self.get_logger().info('Received block placement info')
        
        self.monitoring_positions = self.transform_positions_to_camera(msg)
        if self.monitoring_positions:
            # Log transformed positions
            self.get_logger().info(f'Transformed {len(self.monitoring_positions)} positions to camera frame')
            
            self.monitoring_regions = self.get_monitoring_regions(self.monitoring_positions)
            # Log created regions
            self.get_logger().info(f'Created {len(self.monitoring_regions)} monitoring regions')
        else:
            self.get_logger().warn('Failed to transform positions to camera frame')
        
        self.get_logger().info(f'Starting block monitoring for block {self.current_block_index}')

    def check_for_hands(self):
        """Use MediaPipe to detect hands and manage interaction monitoring."""
        if self.last_color_frame is None:
            return
                
        rgb_frame = cv2.cvtColor(self.last_color_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        hands_present = results.multi_hand_landmarks is not None
        
        if hands_present:
            # Cancel any pending reset when hands are detected
            if self.reset_timer is not None:
                self.reset_timer.cancel()
                self.reset_timer = None
                
            if self.pre_interaction_color is None:
                # Save frames from just before hands appeared
                self.pre_interaction_color = self.last_color_frame.copy()
                self.pre_interaction_depth = self.last_depth_frame.copy()
                self.hands_detected_pub.publish(Bool(data=True))
                self.get_logger().info('Hands detected - saving pre-interaction state')
        else:
            if self.pre_interaction_color is not None:
                # Hands were present but now gone - check for changes
                time.sleep(1)
                changes_detected, max_change_region = self.detect_block_movement()
                
                if changes_detected and max_change_region is not None:
                    # Clear pre-interaction state
                    self.pre_interaction_color = None
                    self.pre_interaction_depth = None
                    
                    # Call move block service to update stack tracking
                    move_request = MoveBlock.Request()
                    move_request.block_index = self.current_block_index
                    move_request.new_category = max_change_region - 1  # Categories are 0-based
                    
                    try:
                        # Use callback group for service calls
                        future = self.move_block_client.call_async(move_request)
                        
                        # Create correction requests
                        sort_correction = CorrectionService.Request()
                        sort_correction.old_label = self.current_block_info.last_block_category
                        sort_correction.new_label = max_change_region - 1
                        
                        complex_gesture_correction = CorrectionService.Request()
                        complex_gesture_correction.old_label = self.current_block_info.last_block_category
                        complex_gesture_correction.new_label = max_change_region - 1
                        
                        gesture_correction = CorrectionService.Request()
                        gesture_correction.old_label = 0
                        gesture_correction.new_label = 0
                        
                        # Send correction service requests
                        self.sort_correction_client.call_async(sort_correction)
                        self.gesture_correction_client.call_async(gesture_correction)
                        self.complex_gesture_correction_client.call_async(complex_gesture_correction)
                        
                        self.get_logger().info('Changes detected - clearing pre-interaction state and updating stack tracking')
                        
                    except Exception as e:
                        self.get_logger().error(f'Error updating stack tracking: {str(e)}')
                else:
                    # Start a timer to reset if no changes detected
                    if self.reset_timer is None:
                        self.reset_timer = self.create_timer(
                            self.monitoring_timeout,
                            self.reset_monitoring_state
                        )
                        self.get_logger().info('No immediate changes detected - starting reset timer')
                
                self.hands_detected_pub.publish(Bool(data=False))


    def reset_monitoring_state(self):
        """Reset the monitoring state if no changes were detected."""
        self.pre_interaction_color = None
        self.pre_interaction_depth = None
        self.reset_timer = None
        self.get_logger().info('Monitoring timeout reached - resetting state')

    def transform_positions_to_camera(self, block_info):
        """Transform world frame positions to camera optical frame coordinates."""
        transformed_positions = []
        
        try:
            # Get transform from world to camera optical frame
            # We lookup the transform in the opposite direction of our desired transformation
            # because we'll apply it to points in the world frame to get camera frame points
            transform = self.tf_buffer.lookup_transform(
                'd435i_color_optical_frame',  # Target frame (where we want points)
                'world',                      # Source frame (where points start)
                self.get_clock().now()        # Get latest transform
            )
            
            # Transform each position from the block info message
            # The block info contains poses in world frame coordinates
            poses_to_transform = [block_info.last_block_pose] + list(block_info.next_positions)
            
            for pose in poses_to_transform:
                # Create a PointStamped from the pose's position
                # We only need position for monitoring regions, not orientation
                point_stamped = PointStamped()
                point_stamped.header.frame_id = 'world'  # Points start in world frame
                point_stamped.header.stamp = self.get_clock().now().to_msg()
                # Copy the position from the pose
                point_stamped.point.x = pose.position.x
                point_stamped.point.y = pose.position.y
                point_stamped.point.z = pose.position.z

                # Transform the point to camera optical frame coordinates
                transformed_point = do_transform_point(point_stamped, transform)
                
                # Create a pose in the camera frame to store the transformed point
                camera_frame_pose = Pose()
                camera_frame_pose.position = transformed_point.point
                # Keep orientation as identity since we only need position for monitoring
                camera_frame_pose.orientation.w = 1.0
                
                transformed_positions.append(camera_frame_pose)
                
            return transformed_positions
                
        except Exception as e:
            self.get_logger().error(f'Position transform failed: {str(e)}')
            return None

    def project_to_image(self, pose):
        """Project 3D camera optical frame coordinates to 2D image coordinates."""
        if self.camera_matrix is None:
            return None
            
        # In the optical frame, Z points forward (into the scene)
        # X points right in the image, Y points down
        point_3d = np.array([
            [pose.position.x],  # Right in image
            [pose.position.y],  # Down in image
            [pose.position.z]   # Forward from camera
        ])
        
        # Skip if point is behind camera
        if point_3d[2] <= 0:
            return None
        
        # Project to image using camera intrinsics
        pixel = self.camera_matrix @ point_3d
        pixel = pixel / pixel[2]
        
        # Convert to integers and return only x,y
        return int(pixel[0]), int(pixel[1])

    def get_monitoring_regions(self, positions):
        """Convert 3D positions to 2D pixel regions for monitoring, excluding the next position
        in the current stack.
        
        Args:
            positions: List of Pose objects, where first is current block position and rest
                    are possible next positions.
        Returns:
            List of monitoring region dictionaries.
        """
        regions = []
        half_size = self.monitoring_window_size // 2
        
        # First, identify which positions we actually want to monitor
        positions_to_monitor = []
        
        # Always monitor the current block position (first position)
        positions_to_monitor.append(positions[0])
        
        # For the remaining positions (next possible positions)
        for i, position in enumerate(positions[1:], 1):
            # Get stack index for this position (i-1 since categories are 0-based)
            position_category = i - 1
            
            # Skip if this position is in the same stack as the current block
            if position_category == self.current_block_info.last_block_category:
                self.get_logger().info(
                    f'Skipping monitoring of position in current stack '
                    f'(category {position_category})'
                )
                continue
                
            positions_to_monitor.append(position)
        
        # Now create monitoring regions for the positions we want to track
        for position in positions_to_monitor:
            pixel_coords = self.project_to_image(position)
            if pixel_coords:
                x, y = pixel_coords
                
                # Create monitoring region centered on projected point
                region = {
                    'x': max(0, x - half_size),
                    'y': max(0, y - half_size),
                    'width': self.monitoring_window_size,
                    'height': self.monitoring_window_size
                }
                
                # Adjust region if it would go past image bounds
                if self.last_color_frame is not None:
                    height, width = self.last_color_frame.shape[:2]
                    if region['x'] + region['width'] > width:
                        region['width'] = width - region['x']
                    if region['y'] + region['height'] > height:
                        region['height'] = height - region['y']
                
                regions.append(region)
        
        return regions

    def detect_region_change(self, region, expect_addition=False):
        """Detect significant depth changes in the specified region."""
        pre_depth = self.get_region_depth(self.pre_interaction_depth, region)
        post_depth = self.get_region_depth(self.last_depth_frame, region)
        
        # Skip if either depth measurement is invalid
        if pre_depth == 0.0 or post_depth == 0.0:
            return False
        
        # Calculate depth change
        depth_change = post_depth - pre_depth
        
        # Log the depth change for debugging
        self.get_logger().debug(
            f'Depth change: {depth_change:.3f}m '
            f'(pre: {pre_depth:.3f}m, post: {post_depth:.3f}m)'
        )
        
        if expect_addition:
            detected = depth_change < -self.depth_change_threshold
        else:
            detected = depth_change > self.depth_change_threshold
            
        if detected:
            self.get_logger().info(
                f'{"Addition" if expect_addition else "Removal"} '
                f'detected with depth change of {depth_change:.3f}m'
            )
            
        return detected

    def detect_block_movement(self):
        """Compare pre and post interaction frames to detect block movement and update tracking.
        Returns:
            tuple: (changes_detected, max_change_region) where changes_detected is a boolean and
                max_change_region is the index of the region with the largest change or None.
        """
        # Check if all required data is available
        required_data = [
            self.pre_interaction_color is not None,
            self.last_color_frame is not None,
            self.pre_interaction_depth is not None,
            self.last_depth_frame is not None,
            self.current_block_info is not None,
            self.monitoring_positions is not None,
            self.block_markers is not None,
            self.monitoring_regions is not None
        ]
        
        if not all(required_data):
            return False, None
                
        # First region is the original block position
        original_region = self.monitoring_regions[0]
        
        # Check if block was removed from original position
        removal_detected = self.detect_region_change(original_region)
        self.get_logger().info(f'Checking original position - removal detected: {removal_detected}')
        
        # Track which regions show addition
        max_change_region = None
        max_change = 0
        
        # Check all possible destination positions
        for i, region in enumerate(self.monitoring_regions[1:], 1):
            if self.detect_region_change(region, expect_addition=True):
                pre_depth = self.get_region_depth(self.pre_interaction_depth, region)
                post_depth = self.get_region_depth(self.last_depth_frame, region)
                change = abs(post_depth - pre_depth)
                
                if change > max_change:
                    max_change = change
                    max_change_region = i
                    self.get_logger().info(f'New max change detected in region {i} with change {change:.3f}m')
        
        return max_change_region is not None, max_change_region

    def get_region_depth(self, depth_image, region):
        """Calculate median depth in the specified region with resolution handling."""
        # Get the scale factors between color and depth images
        if self.last_color_frame is not None and depth_image is not None:
            scale_y = depth_image.shape[0] / self.last_color_frame.shape[0]
            scale_x = depth_image.shape[1] / self.last_color_frame.shape[1]
        else:
            return 0.0

        # Scale the region coordinates to match depth image resolution
        x = int(region['x'] * scale_x)
        y = int(region['y'] * scale_y)
        w = int(region['width'] * scale_x)
        h = int(region['height'] * scale_y)
        
        # Ensure coordinates are within depth image bounds
        height, width = depth_image.shape
        x = min(max(x, 0), width - w)
        y = min(max(y, 0), height - h)
        
        # Extract region and calculate median of non-zero depths
        roi = depth_image[y:y+h, x:x+w]
        valid_depths = roi[roi > 0]
        
        # Calculate ratio of valid depth pixels
        valid_ratio = len(valid_depths) / (w * h)
        
        if len(valid_depths) > 0 and valid_ratio >= self.min_valid_depth_ratio:
            median_depth = np.mean(valid_depths) / 1000.0  # Convert to meters
            self.get_logger().debug(f'Region depth: {median_depth:.3f}m (valid ratio: {valid_ratio:.2f})')
            return median_depth
        
        self.get_logger().debug(f'Invalid depth region (valid ratio: {valid_ratio:.2f})')
        return 0.0

    def localize_camera_callback(self, request, response):
        """Service callback to localize D435 camera using AprilTag."""
        try:
            # Lookup transform: tag -> d435i_link
            tag_to_d435 = self.tf_buffer.lookup_transform(
                'd435_tag',
                'd435i_link',
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=self.get_parameter('calibration_timeout').value)
            )

            # Lookup transform: world -> d405_tag
            world_to_d405_tag = self.tf_buffer.lookup_transform(
                'world',
                'd405_tag',
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=self.get_parameter('calibration_timeout').value)
            )

            # Extract and convert transforms
            t_world_to_d405 = world_to_d405_tag.transform.translation
            r_world_to_d405 = world_to_d405_tag.transform.rotation

            t_tag_to_d435 = tag_to_d435.transform.translation
            r_tag_to_d435 = tag_to_d435.transform.rotation

            # Convert to transformation matrices
            mat_world_to_d405 = tf_transformations.translation_matrix(
                [t_world_to_d405.x, t_world_to_d405.y, t_world_to_d405.z]
            )
            mat_world_to_d405[:3, :3] = tf_transformations.quaternion_matrix(
                [r_world_to_d405.x, r_world_to_d405.y, r_world_to_d405.z, r_world_to_d405.w]
            )[:3, :3]

            mat_tag_to_d435 = tf_transformations.translation_matrix(
                [t_tag_to_d435.x, t_tag_to_d435.y, t_tag_to_d435.z]
            )
            mat_tag_to_d435[:3, :3] = tf_transformations.quaternion_matrix(
                [r_tag_to_d435.x, r_tag_to_d435.y, r_tag_to_d435.z, r_tag_to_d435.w]
            )[:3, :3]

            # Combine transformations
            mat_world_to_d435 = mat_world_to_d405 @ mat_tag_to_d435

            # Extract final transform components
            t_world_to_d435 = tf_transformations.translation_from_matrix(mat_world_to_d435)
            r_world_to_d435 = tf_transformations.quaternion_from_matrix(mat_world_to_d435)

            # Create transform message
            self.d435_transform = TransformStamped()
            self.d435_transform.header.stamp = self.get_clock().now().to_msg()
            self.d435_transform.header.frame_id = 'world'
            self.d435_transform.child_frame_id = 'd435i_link'
            
            # Set translation
            self.d435_transform.transform.translation.x = t_world_to_d435[0]
            self.d435_transform.transform.translation.y = t_world_to_d435[1]
            self.d435_transform.transform.translation.z = t_world_to_d435[2]
            
            # Set rotation
            self.d435_transform.transform.rotation.x = r_world_to_d435[0]
            self.d435_transform.transform.rotation.y = r_world_to_d435[1]
            self.d435_transform.transform.rotation.z = r_world_to_d435[2]
            self.d435_transform.transform.rotation.w = r_world_to_d435[3]

            # Save calibration if configured
            calibration_file = self.get_parameter('calibration_file').value
            if calibration_file:
                self.save_calibration(calibration_file)

            # Broadcast the transform
            self.broadcast_stored_transform()

            # Return success response
            response.success = True
            response.message = "D435 camera successfully localized"
            return response

        except Exception as e:
            self.get_logger().error(f'Error in localization: {str(e)}')
            response.success = False
            response.message = f"Localization failed: {str(e)}"
            return response

    def broadcast_stored_transform(self):
        """Broadcast the stored D435 transform."""
        if self.d435_transform is not None:
            self.d435_transform.header.stamp = self.get_clock().now().to_msg()
            self.tf_broadcaster.sendTransform(self.d435_transform)

    def save_calibration(self, filename):
        """Save camera calibration to CSV file."""
        if self.d435_transform is None:
            return
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            t = self.d435_transform.transform
            writer.writerow([
                t.translation.x, t.translation.y, t.translation.z,
                t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
            ])

    def load_calibration(self, filename):
        """Load camera calibration from CSV file."""
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                row = next(reader)
                values = [float(x) for x in row]
                
                transform = TransformStamped()
                transform.header.frame_id = "world"
                transform.child_frame_id = "d435i_link"
                transform.transform.translation.x = values[0]
                transform.transform.translation.y = values[1]
                transform.transform.translation.z = values[2]
                transform.transform.rotation.x = values[3]
                transform.transform.rotation.y = values[4]
                transform.transform.rotation.z = values[5]
                transform.transform.rotation.w = values[6]
                
                self.d435_transform = transform
                
        except Exception as e:
            self.get_logger().error(f'Failed to load calibration: {str(e)}')

    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()

def main(args=None):
    rclpy.init(args=args)
    node = HumanInteraction()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()