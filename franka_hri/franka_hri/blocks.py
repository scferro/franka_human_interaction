import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import ultralytics

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import MarkerArray, Marker
from std_srvs.srv import Empty
from franka_hri_interfaces.srv import UpdateMarkers, SortNet
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, PointStamped

from geometry_msgs.msg import Quaternion
import tf_transformations

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point

from franka_hri.network import SortingNet
import torchvision.transforms as transforms
import torch
from datetime import datetime
import random

class Blocks(Node):
    def __init__(self):
        super().__init__('blocks')

        # Subscriber to the d405 Image topic
        self.img_sub = self.create_subscription(
            Image,
            '/camera/d405/color/image_rect_raw',
            self.image_callback,
            10)
        self.img_msg_out = Image()
        self.last_img_msg = None

        # Subscriber to the Depth image
        self.depth_sub = self.create_subscription(
            Image, '/camera/d405/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.last_depth_msg = None

        # Subscriber to the Camera Info topic
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/d405/aligned_depth_to_color/camera_info', self.info_callback, 10)

        # Publisher for masked image
        self.img_out_pub = self.create_publisher(Image, 'img_out', 10)

        # Publisher for MarkerArray
        self.block_pub = self.create_publisher(MarkerArray, 'blocks', 10)

        # Create scan_overhead service
        self.scan_overhead_srv = self.create_service(UpdateMarkers, 'scan_overhead', self.scan_overhead_callback)

        # Create update_marker service
        self.update_markers_srv = self.create_service(UpdateMarkers, 'update_markers', self.update_markers_callback)

        # Create the nn object and prediction and training services
        self.network = SortingNet()
        self.train_network_srv = self.create_service(SortNet, 'train_network', self.train_network_callback)
        self.get_network_prediction_srv = self.create_service(SortNet, 'get_network_prediction', self.get_network_prediction_callback)

        # Create TF broadcaster and buffer
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.listener = TransformListener(self.tf_buffer, self)

        # Publish static transform for camera once at startup
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Create timer
        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.timer_count = 0.0

        # Create timer for publishing masked image
        self.marker_timer_period = 0.1  # seconds
        self.marker_timer = self.create_timer(self.marker_timer_period, self.marker_timer_callback)

        # Load yolo model
        self.yolo = ultralytics.YOLO("yolov8x.pt")

        # Other setup
        self.bridge = CvBridge()
        self.block_markers = MarkerArray()
        self.block_images = []
        self.make_static_transforms()
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.image_tensors = []
        self.label_tensors = []

    def train_network_callback(self, request, response):
        # Get block index and label from message
        block_index = request.index
        label = request.label

        # Get images of block
        images = self.block_images[block_index]
        
        filepath = "/home/scferro/Documents/final_project/training_images/"

        # Train the network using the images
        for img in images:
            # Generate the current time as a string
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            # Construct the filename with the current time
            filename = f'image_{current_time}.png'  
            cv2.imwrite(filepath+filename, img)
            transformed_images = self.transform_image(img)
            for img_tf in transformed_images:
                img_tensor = self.preprocess_image(img_tf)
                label_tensor = torch.tensor([[label]], dtype=torch.float32)
                self.image_tensors.append(img_tensor)
                self.label_tensors.append(label_tensor)

        # Limit length of lists to 100 items
        if len(self.image_tensors) > 100:
            self.image_tensors = self.image_tensors[-100:-1]
            self.label_tensors = self.label_tensors[-100:-1]

        self.network.train_network(self.image_tensors, self.label_tensors)
        self.get_logger().info("Trained network.")

        return response

    def get_network_prediction_callback(self, request, response):
        # Get block index from message
        block_index = request.index

        # Get images of block
        images = self.block_images[block_index]

        # Get prediction for each image of the block
        pred_list = []
        for img in images:
            img_tensor = self.preprocess_image(img)
            pred = self.network.forward(img_tensor)
            pred_list.append(pred.detach().numpy()) 

        # Return average prediction for the images
        response.prediction = float(np.mean(pred_list)) 
        self.get_logger().info(f"Prediction: {float(np.mean(pred_list)) }")
        return response

    def preprocess_image(self, image):
        # Define the transformation pipeline
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Apply the transformations to the image
        image_tensor = transform(image)

        # Add a batch dimension to the tensor
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def update_markers_callback(self, request, response):
        # Extract markers and indices from request
        new_markers = request.input_markers.markers
        marker_index_list = request.markers_to_update

        # update markers
        for i in range(len(marker_index_list)):
            index = marker_index_list[i]
            marker = new_markers[i]

            self.block_markers.markers[index] = marker

        # Return updated markers
        response.output_markers = self.block_markers

        return response

    def segment_depth(self, rgb_image, depth_image):
        h, w = depth_image.shape
        
        # Convert depth image to 8-bit for processing
        depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=255.0 / (np.max(depth_image) - np.min(depth_image)))

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(depth_image_8bit, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blur, 30, 150)

        # Apply Hough Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)

        # Create a mask image with black background
        mask = np.zeros_like(depth_image_8bit)

        # Iterate over the detected lines and draw them on the mask
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)

        # Find the contours in the mask
        inverted_mask = cv2.bitwise_not(mask)
        contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a new mask image with black background
        table_mask = np.zeros_like(depth_image_8bit)

        # Draw the largest contour (assumed to be the table) on the new mask with white color
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(table_mask, [largest_contour], -1, 255, -1)

        # Apply morphological closing to remove noise
        kernel = np.ones((11,11), np.uint8)
        closed_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel)

        # Apply the mask to the depth image to extract the table region
        table_segmented = cv2.bitwise_and(rgb_image, rgb_image, mask=closed_mask)

        return table_segmented

    def segment_image(self, image):
        # Create a copy of the image to avoid modifying the original
        img_copy = image.copy()

        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)

        # Define the color range for the objects you want to segment (adjust as needed)
        lower_color = np.array([0, 100, 80])  
        upper_color = np.array([180, 255, 255]) 

        # Create a binary mask based on the color range
        mask = cv2.inRange(hsv_image, lower_color, upper_color)

        # Find the contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the mask with white color
        mask_image = np.zeros_like(image)
        cv2.drawContours(mask_image, contours, -1, (255, 255, 255), -1)

        # Use opening to remove noise
        kernel = np.ones((7,7), np.uint8)
        open_mask = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel)
        
        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(image, open_mask)

        # Find the contours in the mask
        color_mask = cv2.cvtColor(open_mask, cv2.COLOR_BGR2GRAY)
        contours_out, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours_out, masked_image

    def scan_overhead_callback(self, request, response):
        """Function to scan table to determine approximate size and position of blocks."""
        if self.last_img_msg is None:
            return
        
        # Get last image from realsense, convert to cv2
        image = self.ros2_image_to_cv2(self.last_img_msg, encoding='bgr8')
        depth = self.ros2_image_to_cv2(self.last_depth_msg, encoding='8UC1')

        img_table = self.segment_depth(image, depth)
        contours, img_mask = self.segment_image(img_table)

        # Create a copy of the original image to draw boxes on
        img_with_boxes = img_mask.copy()

        # List to store the sub-images
        sub_images = []

        buffer = 5

        # Create marker list
        markers = request.input_markers
        markers_to_update = list(request.markers_to_update)

        update_scale = request.update_scale

        # Iterate through each contour
        for contour in contours:
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw the bounding box on the image
            cv2.rectangle(img_with_boxes, (x - buffer, y - buffer), (x + w + buffer, y + h + buffer), (0, 255, 0), 2)
            
            # Create a mask for the current contour
            contour_mask = np.zeros_like(img_mask)
            cv2.drawContours(contour_mask, [contour], 0, (255, 255, 255), -1)
            
            # Extract the sub-image using the contour mask
            sub_image = cv2.bitwise_and(img_table, contour_mask)

            # Crop the sub-image to the bounding box dimensions
            cropped_sub_image = sub_image[max(0, y-buffer):min(y+h+buffer, sub_image.shape[0]),
                                        max(0, x-buffer):min(x+w+buffer, sub_image.shape[1])]

            # Get the dimensions of the cropped sub-image
            sub_height, sub_width, _ = cropped_sub_image.shape

            # Calculate the size of the square image
            square_size = max(sub_height, sub_width)

            # Create a new square image with black background
            square_image = np.zeros((square_size, square_size, 3), dtype=np.uint8)

            # Calculate the position to place the sub-image in the square image
            y_pos = (square_size - sub_height) // 2
            x_pos = (square_size - sub_width) // 2

            # Place the sub-image in the center of the square image
            square_image[y_pos:y_pos+sub_height, x_pos:x_pos+sub_width] = cropped_sub_image

            # Resize the square image to 128x128 pixels
            resized_image = cv2.resize(square_image, (128, 128))

            # Draw angled minimum angle rectangle
            rect = cv2.minAreaRect(contour)
            box_center = rect[0]
            box_dimensions = rect[1]
            box_angle = rect[2]
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_with_boxes,[box],0,(0,0,255),2)

            # Adjust angle so oriented along long axis
            if box_dimensions[1] > box_dimensions[0]:
                box_angle = -box_angle
            else:
                box_angle = 90 - box_angle

            box_angle_rad = box_angle / 180 * np.pi

            # Get box center in 3d space
            center_x = int(box_center[0])
            center_y = int(box_center[1])
            center_3d = self.get_position_from_image(center_x, center_y, self.last_depth_msg)

            # Get 3d points for box edges
            points_3d = []
            for point in box:
                pt_3d = self.get_position_from_image(point[0], point[1], self.last_depth_msg)
                points_3d.append(pt_3d)

            # Extract box side lengths
            side_lengths = []
            for i in range(len(points_3d)):
                length = np.sqrt((points_3d[i-1][0] - points_3d[i][0])**2 + (points_3d[i-1][1] - points_3d[i][1])**2)
                side_lengths.append(length)

            side1_length = (side_lengths[0] + side_lengths[2]) / 2
            side2_length = (side_lengths[1] + side_lengths[3]) / 2

            # Get the top position of the box
            appx_height = self.get_depth_percentile(depth, contour)

            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.type = 1

            # Create PointStamped object for transformation
            point_stamped = PointStamped()
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.header.frame_id = 'd405_link'
            point_stamped.point.x = center_3d[0]
            point_stamped.point.y = center_3d[1]
            point_stamped.point.z = center_3d[2]

            # Convert to base link frame
            try:
                transform = self.tf_buffer.lookup_transform('world', point_stamped.header.frame_id, rclpy.time.Time())
                world_point = do_transform_point(point_stamped, transform)

                # Set the box pose
                marker.pose.position.x = world_point.point.x
                marker.pose.position.y = world_point.point.y
                marker.pose.position.z = world_point.point.z / 2

                # Get the block orientation
                # euler_angles = tf_transformations.euler_from_quaternion(quaternion)
                # camera_angle = euler_angles[2]
                quaternion = tf_transformations.quaternion_from_euler(np.pi, 0, box_angle_rad)
                marker.pose.orientation.x = quaternion[0]
                marker.pose.orientation.y = quaternion[1]
                marker.pose.orientation.z = quaternion[2]
                marker.pose.orientation.w = quaternion[3]

                # Set the box size
                marker.scale.x = max(side1_length, side2_length)
                marker.scale.y = min(side1_length, side2_length)
                marker.scale.z = world_point.point.z

                # Set the box color
                avg_color = self.get_average_color(img_mask, contour)
                marker.color.r = avg_color[2] / 255
                marker.color.g = avg_color[1] / 255
                marker.color.b = avg_color[0] / 255
                marker.color.a = 1.0

                # Check if the marker has already been scanned
                similar_marker_index = self.find_similar_marker(marker, markers)

                if not update_scale:
                    marker.scale = markers.markers[similar_marker_index].scale
                    marker.pose.position.z = world_point.point.z  - (marker.scale.z / 2)

                update_marker = False
                if markers_to_update==[] or (similar_marker_index in markers_to_update):
                    update_marker = True

                if similar_marker_index==-1 and update_marker:
                    # Set marker ID and add marker to array
                    marker.id = len(markers.markers)
                    markers.markers.append(marker)
                    self.block_images.append([resized_image])
                elif update_marker:
                    marker.id = markers.markers[similar_marker_index].id
                    markers.markers[similar_marker_index] = marker
                    self.block_images[similar_marker_index].append(resized_image)

            except Exception as e:
                self.get_logger().info(f"Failed to transform to 'world' frame: {str(e)}")

        self.block_markers = markers
        response.output_markers = markers
        img_with_boxes_msg = self.cv2_to_ros2_image(img_with_boxes)
        self.img_msg_out = img_with_boxes_msg
        self.get_logger().info('Block scanning complete!')

        return response

    def get_position_from_image(self, u, v, depth_msg):
        """Retrieve the 3D coordinates based on the depth image data."""
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            self.get_logger().warn('Camera info not yet received.')
            return

        # Convert the depth image to a Numpy array
        depth_cv_image = self.ros2_image_to_cv2(depth_msg)
        depth_array = np.array(depth_cv_image, dtype=np.float32)
        depth_height, depth_width = depth_array.shape

        if u >= depth_width: u = depth_width - 1
        if v >= depth_height: v = depth_height - 1

        z = depth_array[v, u] / 1000.0  # Depth in meters

        half_window = 11
        u_min = max(0, u - half_window)
        u_max = min(depth_array.shape[1], u + half_window + 1)
        v_min = max(0, v - half_window)
        v_max = min(depth_array.shape[0], v + half_window + 1)

        # Extract the window and calculate the average depth, ignoring NaNs and zeros
        window = depth_array[v_min:v_max, u_min:u_max]
        valid_depths = window[np.logical_and(window > 0, ~np.isnan(window))]
        z = np.mean(valid_depths) / 1000.0  # Convert to meters

        # Calculate the 3D coordinates
        X = (u - self.cx) * z / self.fx
        Y = (v - self.cy) * z / self.fy
        Z = z

        return [X, Y, Z]

    def find_similar_marker(self, single_marker, marker_array, distance_threshold=0.02, size_threshold=0.03):
        for i, marker in enumerate(marker_array.markers):
            if marker.type == single_marker.type and marker.action == single_marker.action:
                # Check if the markers have similar positions
                point1 = single_marker.pose.position
                point2 = marker.pose.position
                distance = ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2) ** 0.5
                if distance <= distance_threshold:
                    # Check if the markers have similar sizes
                    size_difference = self.calculate_size_difference(single_marker.scale, marker.scale)
                    if size_difference <= size_threshold:
                        return i
        return -1

    def calculate_size_difference(self, scale1, scale2):
        return max(abs(scale1.x - scale2.x), abs(scale1.y - scale2.y), abs(scale1.z - scale2.z))
    
    def get_average_color(self, color_image, contour):
        # Create a mask for the contour
        mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        # Apply the mask to the color image
        masked_color = cv2.bitwise_and(color_image, color_image, mask=mask)
        
        # Get the color values within the contour area
        color_values = masked_color[mask > 0]

        # Calculate the average color values
        average_color = np.mean(color_values, axis=0)
        return average_color

    def get_depth_percentile(self, depth_image, contour, percentile=0.15):
        # Create a mask for the contour
        mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        # Apply the mask to the depth image
        masked_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)
        
        # Get the depth values under the mask
        depth_values = masked_depth[masked_depth > 0]
        
        # Check if there are any depth values
        if len(depth_values) > 0:
            # Calculate the 75th percentile depth value
            percentile_depth = np.percentile(depth_values, percentile * 100)
            return percentile_depth
        else:
            return None

    def info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def marker_timer_callback(self):
        if self.img_msg_out != None:
            self.img_out_pub.publish(self.img_msg_out)
            self.block_pub.publish(self.block_markers)

    def timer_callback(self):
        self.timer_count += self.timer_period

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
            self.get_logger().info(f"Error converting ROS2 image to OpenCV: {e}")
            return None

    def cv2_to_ros2_image(self, cv_image, encoding="bgr8"):
        bridge = CvBridge()
        try:
            ros_image_msg = bridge.cv2_to_imgmsg(cv_image, encoding=encoding)
            return ros_image_msg
        except CvBridgeError as e:
            self.get_logger().info(f"Error converting OpenCV image to ROS2 message: {e}")
            return None

    def make_static_transforms(self):
        t_cam = TransformStamped()

        t_cam.header.stamp = self.get_clock().now().to_msg()
        t_cam.header.frame_id = 'panda_hand'
        t_cam.child_frame_id = 'd405_link'

        t_cam.transform.translation.x = 0.07
        t_cam.transform.translation.y = 0.0
        t_cam.transform.translation.z = 0.05

        t_cam.transform.rotation.x = 0.0
        t_cam.transform.rotation.y = 0.0
        t_cam.transform.rotation.z = 0.7071068
        t_cam.transform.rotation.w = 0.7071068

        self.tf_static_broadcaster.sendTransform(t_cam)

        t_world = TransformStamped()

        t_world.header.stamp = self.get_clock().now().to_msg()
        t_world.header.frame_id = 'world'
        t_world.child_frame_id = 'panda_link0'

        t_world.transform.translation.x = 0.0
        t_world.transform.translation.y = 0.0
        t_world.transform.translation.z = 0.0

        t_world.transform.rotation.x = 0.0
        t_world.transform.rotation.y = 0.0
        t_world.transform.rotation.z = 0.0
        t_world.transform.rotation.w = 1.0

        self.tf_static_broadcaster.sendTransform(t_world)
        self.get_logger().info("Published static transforms.")

    def transform_image(self, image):
        height, width = image.shape[:2]

        def rotate_image(image, angle):
            M = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
            return cv2.warpAffine(image, M, (width, height))

        def mirror_image(image):
            return cv2.flip(image, 1)

        transformations = []
        
        for angle in [0, 90, 180, 270]:
            rotated_image = rotate_image(image, angle)
            mirrored_image = mirror_image(rotated_image)
            for img in [rotated_image, mirrored_image]:
                transformations.append(img)

        return transformations


def main(args=None):
    """The main function."""
    rclpy.init(args=args)
    node = Blocks()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
