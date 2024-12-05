import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer, TransformBroadcaster, StaticTransformBroadcaster
import tf_transformations
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Trigger
import csv
import os

class HumanInteraction(Node):
    def __init__(self):
        super().__init__('human_interaction')
        
        # Declare parameters
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('calibration_timeout', 10.0)
        
        # Initialize TF listeners and broadcasters
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        
        # Initialize stored transform
        self.d435_transform = None
        
        # Load calibration if file exists
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
        
        # Add timer for broadcasting transform
        self.tf_timer = self.create_timer(0.1, self.broadcast_transform_timer)
        
        self.get_logger().info('Human Interaction node initialized')
        
    def broadcast_transform_timer(self):
    """Timer callback to continuously broadcast D435 transform.""" 
    if self.d435_transform is not None:
        self.d435_transform.header.stamp = self.get_clock().now().to_msg()
        self.tf_broadcaster.sendTransform(self.d435_transform)

    def localize_camera_callback(self, request, response):
        """Service callback to localize D435 camera using AprilTag."""
        try:
            # Lookup transform: tag -> d435i_link
            tag_to_d435 = self.tf_buffer.lookup_transform(
                'd435_tag',  # The tag frame associated with D435
                'd435i_link',  # The D435 camera link
                rclpy.time.Time(),
                rclpy.time.Duration(seconds=self.get_parameter('calibration_timeout').value)
            )

            # Lookup transform: world -> d405_tag
            world_to_d405_tag = self.tf_buffer.lookup_transform(
                'world',  # World frame
                'd405_tag',  # The tag frame associated with D405
                rclpy.time.Time(),
                rclpy.time.Duration(seconds=self.get_parameter('calibration_timeout').value)
            )

            # Extract translation and rotation from the transforms
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

            # Combine the matrices
            mat_world_to_d435 = mat_world_to_d405 @ mat_tag_to_d435

            # Extract translation and rotation from the combined matrix
            t_world_to_d435 = tf_transformations.translation_from_matrix(mat_world_to_d435)
            r_world_to_d435 = tf_transformations.quaternion_from_matrix(mat_world_to_d435)

            # Build the TransformStamped message
            world_to_d435 = TransformStamped()
            world_to_d435.header.stamp = self.get_clock().now().to_msg()
            world_to_d435.header.frame_id = 'world'
            world_to_d435.child_frame_id = 'd435i_link'
            world_to_d435.transform.translation.x = t_world_to_d435[0]
            world_to_d435.transform.translation.y = t_world_to_d435[1]
            world_to_d435.transform.translation.z = t_world_to_d435[2]
            world_to_d435.transform.rotation.x = r_world_to_d435[0]
            world_to_d435.transform.rotation.y = r_world_to_d435[1]
            world_to_d435.transform.rotation.z = r_world_to_d435[2]
            world_to_d435.transform.rotation.w = r_world_to_d435[3]

            # Save the resulting transform
            self.d435_transform = world_to_d435

            # Optionally save calibration to file
            calibration_file = self.get_parameter('calibration_file').value
            if calibration_file:
                self.save_calibration(calibration_file)

            # Broadcast the stored transform
            self.broadcast_stored_transform()

            # Respond success
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

def main(args=None):
    rclpy.init(args=args)
    node = HumanInteraction()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()