import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer, TransformBroadcaster, StaticTransformBroadcaster
import tf_transformations
from geometry_msgs.msg import TransformStamped, Pose
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import MarkerArray
from franka_hri_interfaces.msg import BlockPlacementInfo
from franka_hri_interfaces.srv import UpdateMarkers
from franka_hri_interfaces.srv import MoveBlock
import mediapipe as mp
import numpy as np
import cv2
import csv
import os
from tf2_geometry_msgs import do_transform_pose

class HumanInteraction(Node):
    def __init__(self):
        super().__init__('human_interaction')
        
        # Declare parameters
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('calibration_timeout', 10.0)
        self.declare_parameter('camera_info_topic', '/camera/d435i/aligned_depth_to_color/camera_info')
        self.declare_parameter('depth_topic', '/camera/d435i/aligned_depth_to_color/image_raw')
        self.declare_parameter('color_topic', '/camera/d435i/color/image_raw')
        
        # Initialize camera matrix and monitoring state
        self.camera_matrix = None
        self.current_block_info = None
        self.monitoring_active = False
        self.monitoring_positions = None
        self.current_block_index = None
        self.block_markers = None
        
        # Initialize frame storage
        self.last_color_frame = None
        self.last_depth_frame = None 
        self.pre_interaction_color = None
        self.pre_interaction_depth = None
        
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
            'blocks',  # Topic where blocks node publishes markers
            self.marker_array_callback,
            10
        )
        
        # Create publishers and service clients
        self.hands_detected_pub = self.create_publisher(Bool, 'hands_detected', 10)
        self.update_markers_cli = self.create_client(UpdateMarkers, 'update_markers')
        self.move_block_client = self.create_client(MoveBlock, 'update_block_tracking')
        
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
        
        # Add transform broadcast timer
        self.tf_timer = self.create_timer(0.1, self.broadcast_stored_transform)
        
        self.get_logger().info('Human Interaction node initialized')

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
        self.monitoring_positions = self.transform_positions_to_camera(msg)
        self.get_logger().info(f'Starting block monitoring for block {self.current_block_index}')

    def check_for_hands(self):
        """Use MediaPipe to detect hands and manage interaction monitoring."""
        if self.last_color_frame is None:
            return
            
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(self.last_color_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Handle hand detection state changes
        hands_present = results.multi_hand_landmarks is not None
        
        if hands_present:
            if self.pre_interaction_color is None:
                # Save frames from just before hands appeared
                self.pre_interaction_color = self.last_color_frame.copy()
                self.pre_interaction_depth = self.last_depth_frame.copy()
                self.hands_detected_pub.publish(Bool(data=True))
                self.get_logger().info('Hands detected - saving pre-interaction state')
        else:
            if self.pre_interaction_color is not None:
                # Hands were present but now gone - check for changes
                self.detect_block_movement()
                self.pre_interaction_color = None
                self.pre_interaction_depth = None
                self.hands_detected_pub.publish(Bool(data=False))
                self.get_logger().info('Hands removed - checking for block movement')

    def transform_positions_to_camera(self, block_info):
        """Transform world frame positions to camera frame coordinates."""
        transformed_positions = []
        
        try:
            # Get transform from world to camera
            transform = self.tf_buffer.lookup_transform(
                'd435i_link',
                'world',
                rclpy.time.Time()
            )
            
            # Transform each position
            poses_to_transform = [block_info.last_block_pose] + list(block_info.next_positions)
            for pose in poses_to_transform:
                pose_stamped = Pose()
                pose_stamped = pose
                transformed = do_transform_pose(pose_stamped, transform)
                transformed_positions.append(transformed)
                
            return transformed_positions
            
        except Exception as e:
            self.get_logger().error(f'Position transform failed: {str(e)}')
            return None

    def project_to_image(self, pose):
        """Project 3D camera frame coordinates to 2D image coordinates."""
        if self.camera_matrix is None:
            return None
            
        # Extract 3D point
        point_3d = np.array([
            [pose.position.x],
            [pose.position.y],
            [pose.position.z]
        ])
        
        # Project to image
        pixel = self.camera_matrix @ point_3d
        pixel = pixel / pixel[2]
        
        return int(pixel[0]), int(pixel[1])

    def get_monitoring_regions(self, positions):
        """Convert 3D positions to 2D pixel regions for monitoring."""
        regions = []
        for position in positions:
            pixel_coords = self.project_to_image(position)
            if pixel_coords:
                x, y = pixel_coords
                region = {
                    'x': max(0, x - 20),  # 40x40 pixel region centered on point
                    'y': max(0, y - 20),
                    'width': 50,
                    'height': 50
                }
                regions.append(region)
        return regions

    def detect_block_movement(self):
        """Compare pre and post interaction frames to detect block movement and update tracking."""
        # Check if all required data is available
        required_data = [
            self.pre_interaction_color is not None,
            self.last_color_frame is not None,
            self.pre_interaction_depth is not None,
            self.last_depth_frame is not None,
            self.current_block_info is not None,
            self.monitoring_positions is not None,
            self.block_markers is not None
        ]
        
        if not all(required_data):
            return
            
        # First region is the original block position
        original_region = regions[0]
        
        # Check if block was removed from original position
        if self.detect_region_change(original_region):
            # Check all possible destination positions
            for i, region in enumerate(regions[1:], 1):
                if self.detect_region_change(region, expect_addition=True):
                    self.get_logger().info(f'Block {self.current_block_index} moved to stack {i-1}')
                    
                    # Create MoveBlock request to update pile tracking
                    move_request = MoveBlock.Request()
                    move_request.block_index = self.current_block_index
                    move_request.new_category = i-1  # Categories are 0-based
                    
                    # Call the move block service
                    try:
                        future = self.move_block_client.call_async(move_request)
                        rclpy.spin_until_future_complete(self, future)
                        
                        if future.result() is not None and future.result().success:
                            self.get_logger().info('Successfully updated block tracking')
                        else:
                            self.get_logger().error('Failed to update block tracking')
                            
                    except Exception as e:
                        self.get_logger().error(f'Error calling move_block service: {str(e)}')
                    
                    # We still update the marker position for visualization
                    marker_request = UpdateMarkers.Request()
                    marker_request.input_markers = self.block_markers
                    marker_request.markers_to_update = [self.current_block_index]
                    marker_request.update_scale = False
                    
                    for marker in marker_request.input_markers.markers:
                        if marker.id == self.current_block_index:
                            marker.pose = self.current_block_info.next_positions[i-1]
                            break
                    
                    try:
                        future = self.update_markers_cli.call_async(marker_request)
                        rclpy.spin_until_future_complete(self, future)
                        if future.result() is not None:
                            self.block_markers = future.result().output_markers
                            self.get_logger().info('Successfully updated marker position')
                        else:
                            self.get_logger().error('Failed to update marker position')
                    except Exception as e:
                        self.get_logger().error(f'Error calling update_markers service: {str(e)}')
                    
                    # Reset monitoring state
                    self.monitoring_active = False
                    self.current_block_index = None
                    return

    def detect_region_change(self, region, expect_addition=False):
        """Detect significant depth changes in the specified region."""
        pre_depth = self.get_region_depth(self.pre_interaction_depth, region)
        post_depth = self.get_region_depth(self.last_depth_frame, region)
        
        # Calculate depth change
        depth_change = post_depth - pre_depth
        
        # Thresholds for significant change
        REMOVAL_THRESHOLD = 0.02  # 2cm
        ADDITION_THRESHOLD = -0.02  # -2cm
        
        if expect_addition:
            return depth_change < ADDITION_THRESHOLD
        return depth_change > REMOVAL_THRESHOLD

    def get_region_depth(self, depth_image, region):
        """Calculate median depth in the specified region."""
        x, y = region['x'], region['y']
        w, h = region['width'], region['height']
        
        # Ensure coordinates are within image bounds
        height, width = depth_image.shape
        x = min(max(x, 0), width - w)
        y = min(max(y, 0), height - h)
        
        # Extract region and calculate median of non-zero depths
        roi = depth_image[y:y+h, x:x+w]
        valid_depths = roi[roi > 0]
        if len(valid_depths) > 0:
            return np.median(valid_depths) / 1000.0  # Convert to meters
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