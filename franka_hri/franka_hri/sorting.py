import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_srvs.srv import Empty
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Pose

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster
from tf2_ros import Buffer, TransformListener
import tf_transformations


class Sorting(Node):
    def __init__(self):
        super().__init__('sorting')
        
        # Subscriber to the Image topic
        self.img_sub = self.create_subscription(
            Image,
            '/camera/d405/color/image_rect_raw',  
            self.image_callback,
            10)
        
        # Subsscriber to the Depth image
        self.depth_sub = self.create_subscription(
            Image, 
            '/camera/d405/aligned_depth_to_color/image_raw', 
            self.depth_callback, 
            10)
        
        # Subscriber to the Camera Info topic
        self.info_sub = self.create_subscription(
            CameraInfo, 
            '/camera/d405/aligned_depth_to_color/camera_info', 
            self.info_callback, 
            10)
        
        # Publisher for MarkerArray
        self.block_pub = self.create_publisher(
            MarkerArray,
            'blocks',  
            10)
        
        # Publisher for image with contours
        self.img_out_pub = self.create_publisher(
            Image,
            'img_out',  
            10)
        
        # Create timer
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Create testing service
        self.scan_srv = self.create_service(Empty, 'scan', self.scan_srv_callback)

        # Create TF broadcaster and buffer
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.listener = TransformListener(self.tf_buffer, self)

        # Publish static transform for camera once at startup
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.make_static_transforms()

        # Store block poses in a list
        self.block_tfs = []

    def timer_callback(self):
        for tf in self.block_tfs:
            # Send the transformation
            self.tf_broadcaster.sendTransform(tf)

    def scan_srv_callback(self, request, response):
        self.scan()
        return response

    def scan(self):
        # Extract image from message
        image = self.ros2_image_to_cv2(self.last_img_msg)

        # Convert to grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Canny edge detection
        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
        
        # Find contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        square_vertices = []
        image_out = image
        min_distance = 10

        # Filter non-squares from detected contours
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            simplified_approx = approx.reshape(-1, 2)

            # If the polygon has 4 vertices, it is likely a rectangle
            if len(simplified_approx) == 4:
                valid = True
                for i in range(4):
                    if not valid:
                        break
                    # Check if points are too close
                    pt1, pt2 = simplified_approx[i-1], simplified_approx[i]
                    distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                    if distance < min_distance:
                        valid = False
                if valid:
                    square_vertices.append(simplified_approx)
                    cv2.drawContours(image_out, [simplified_approx], 0, (0, 255, 0), 3)

        count = 0
        self.block_tfs = []
        for vert in square_vertices:
            # Get the midpoint of the square
            x1, y1 = vert[0][0], vert[0][1]
            x2, y2 = vert[2][0], vert[2][1]
            x_mid = int((x1 + x2) / 2)
            y_mid = int((y1 + y2) / 2)

            # Get the 3D position
            x_3d, y_3d, z_3d = self.get_position(x_mid, y_mid)

            pose = Pose()

            block_id = 'block' + str(count)

            # Read message content and assign it to
            # corresponding tf variables
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'd405_link'

            # Position
            pose.pose.position.x = float(x_3d)
            pose.pose.position.y = float(y_3d)
            pose.pose.position.z = float(z_3d)

            # Orientation
            # q = quaternion_from_euler(0, 0, msg.theta)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0

            # Convert to base link frame
            world_pose = self.tf_buffer.transform(pose, 'world')

            # Get world transform
            t_world = TransformStamped()

            t_world.header.stamp = self.get_clock().now().to_msg()
            t_world.header.frame_id = 'world'
            t_world.child_frame_id = block_id

            t_world.transform.translation.x = pose.pose.position.x
            t_world.transform.translation.y = pose.pose.position.y
            t_world.transform.translation.z = pose.pose.position.z
            
            t_world.transform.rotation.x = 0.0
            t_world.transform.rotation.y = 0.0
            t_world.transform.rotation.z = 0.0
            t_world.transform.rotation.w = 1.0

            # Save TF
            self.block_tfs.append(t_world)
            count += 1

        img_msg_out = self.cv2_to_ros2_image(image_out)
        self.img_out_pub.publish(img_msg_out)

    def get_position(self, u, v):
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            self.get_logger().warn('Camera info not yet received.')
            return

        # Convert the depth image to a Numpy array
        cv_image = self.ros2_image_to_cv2(self.last_depth_msg)
        depth_array = np.array(cv_image, dtype=np.float32)
        z = depth_array[v, u] / 1000.0  # Depth in meters

        while np.isnan(z) or z == 0:
            # self.get_logger().warn('Depth value is NaN or zero.')
            v += 1
            u += 1
            z = depth_array[v, u] / 1000.0  # Depth in meters

        # Calculate the 3D coordinates
        X = (u - self.cx) * z / self.fx
        Y = (v - self.cy) * z / self.fy
        Z = z

        return X, Y, Z

    def info_callback(self, msg):
        self.fx = msg.k[0] 
        self.fy = msg.k[4] 
        self.cx = msg.k[2]  
        self.cy = msg.k[5]  
        # self.get_logger().info(f'Received camera info: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')

    def image_callback(self, msg):
        self.last_img_msg = msg

    def depth_callback(self, msg):
        self.last_depth_msg = msg

    def ros2_image_to_cv2(self, ros_image_msg, encoding='passthrough'):
        bridge = CvBridge()
        try:
            cv_image = bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding=encoding)
            return cv_image
        except CvBridgeError as e:
            print(f"Error converting ROS2 image to OpenCV: {e}")
            return None
        
    def cv2_to_ros2_image(self, cv_image, encoding="bgr8"):
        bridge = CvBridge()
        try:
            ros_image_msg = bridge.cv2_to_imgmsg(cv_image, encoding=encoding)
            return ros_image_msg
        except CvBridgeError as e:
            print(f"Error converting OpenCV image to ROS2 message: {e}")
            return None

    def make_static_transforms(self):
        t_cam = TransformStamped()

        t_cam.header.stamp = self.get_clock().now().to_msg()
        t_cam.header.frame_id = 'panda_hand'
        t_cam.child_frame_id = 'd405_link'

        t_cam.transform.translation.x = 0.05
        t_cam.transform.translation.y = 0.0
        t_cam.transform.translation.z = 0.04
        
        t_cam.transform.rotation.x = 0.0
        t_cam.transform.rotation.y = 0.0
        t_cam.transform.rotation.z = 0.0
        t_cam.transform.rotation.w = 1.0

        self.tf_static_broadcaster.sendTransform(t_cam)

        t_world = TransformStamped()

        t_world.header.stamp = self.get_clock().now().to_msg()
        t_world.header.frame_id = 'world'
        t_world.child_frame_id = 'panda_link_0'

        t_world.transform.translation.x = 0.0
        t_world.transform.translation.y = 0.0
        t_world.transform.translation.z = 0.0
        
        t_world.transform.rotation.x = 0.0
        t_world.transform.rotation.y = 0.0
        t_world.transform.rotation.z = 0.0
        t_world.transform.rotation.w = 1.0

        self.tf_static_broadcaster.sendTransform(t_world)


def main(args=None):
    """ The main() function. """
    rclpy.init(args=args)
    node = Sorting()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()