import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_srvs.srv import Empty

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
        self.timer = self.create_timer(1.0, self.timer_callback)

        # Create testing service
        self.srv = self.create_service(Empty, 'empty_service', self.empty_service_callback)

    def timer_callback(self):
        # self.scan()
        pass

    def empty_service_callback(self, request, response):
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
                    print(distance)
                    if distance < min_distance:
                        valid = False
                if valid:
                    square_vertices.append(simplified_approx)
                    cv2.drawContours(image_out, [simplified_approx], 0, (0, 255, 0), 3)

        for vert in square_vertices:
            # Get the midpoint of the square
            x1, y1 = vert[0][0], vert[0][1]
            x2, y2 = vert[2][0], vert[2][1]
            x_mid = int((x1 + x2) / 2)
            y_mid = int((y1 + y2) / 2)

            # Get the 3D position
            x_3d, y_3d, z_3d = self.get_position(x_mid, y_mid)

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


def main(args=None):
    """ The main() function. """
    rclpy.init(args=args)
    node = Sorting()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()