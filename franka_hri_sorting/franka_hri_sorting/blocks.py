import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import mediapipe as mp

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import MarkerArray, Marker
from std_srvs.srv import Empty
from std_msgs.msg import Bool
from franka_hri_interfaces.srv import UpdateMarkers, SortNet
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, PointStamped

from geometry_msgs.msg import Quaternion
import tf_transformations

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point

import torchvision.transforms as transforms
from datetime import datetime
import random
from sklearn.linear_model import RANSACRegressor
import rclpy.time

class Blocks(Node):
    def __init__(self):
        super().__init__('blocks')
    
        # Create callback groups for concurrent service handling
        self.marker_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        self.prediction_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        self.network_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()

        # Parameter for x-axis division
        self.declare_parameter('scan_position_x', 0.5)
        self.scan_position_x = self.get_parameter('scan_position_x').value
        
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
        self.update_markers_srv = self.create_service(
            UpdateMarkers, 
            'update_markers', 
            self.update_markers_callback,
            callback_group=self.marker_callback_group
        )

        # Create pre train blocks service
        self.pretrain_network_srv = self.create_service(Empty, 'pretrain_network', self.pretrain_network_callback)

        # Create reset blocks service
        self.reset_blocks_srv = self.create_service(Empty, 'reset_blocks', self.reset_blocks_callback)

        # Create the nn object and prediction and training services
        self.train_network_srv = self.create_service(
            SortNet, 
            'train_network', 
            self.train_network_callback,
            callback_group=self.prediction_callback_group
        )
        self.get_network_prediction_srv = self.create_service(
            SortNet, 
            'get_network_prediction', 
            self.get_network_prediction_callback,
            callback_group=self.prediction_callback_group
        )

        # Add in __init__
        self.train_sorting_client = self.create_client(
            SortNet, 
            'train_sorting',
            callback_group=self.network_callback_group
        )
        # Create clients with callback groups
        self.get_sorting_prediction_client = self.create_client(
            SortNet, 
            'get_sorting_prediction',
            callback_group=self.network_callback_group
        )

        # Create TF broadcaster and buffer
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.listener = TransformListener(self.tf_buffer, self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Create timer
        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.timer_count = 0.0

        # Create timer for publishing masked image
        self.marker_timer_period = 0.1  # seconds
        self.marker_timer = self.create_timer(self.marker_timer_period, self.marker_timer_callback)

        # MediaPipe hands detector
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        
        # Publisher for hand detection
        self.hands_detected_pub = self.create_publisher(
            Bool, 
            'hands_detected', 
            10
        )
        
        # Timer for hand detection
        self.hand_detection_period = 0.2
        self.hand_detection_timer = self.create_timer(
            self.hand_detection_period, 
            self.hand_detection_timer_callback
        )
        
        # Initialize hand detection state
        self.last_hand_state = False

        # Other setup
        self.bridge = CvBridge()
        self.block_markers = MarkerArray()
        self.block_images = []
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.table_z_offset = 0.02

        self.get_logger().info("Blocks node started successfully!")

    async def pretrain_network_callback(self, request, response):
        self.block_markers = MarkerArray()
        self.block_images = []

        # Scan for blocks
        marker_array, img_with_boxes = self.scan_overhead(self.block_markers, [], True)
        img_with_boxes_msg = self.cv2_to_ros2_image(img_with_boxes)
        self.img_msg_out = img_with_boxes_msg
        markers = marker_array.markers

        # Train network with each block's images
        for i in range(len(markers)):
            marker = markers[i]
            images = self.block_images[i]
            
            # Determine category based on position relative to scan_position_x and y=0
            # Categories:
            # 0: Back left  (x < scan_position_x, y < 0)
            # 1: Front left (x < scan_position_x, y > 0)
            # 2: Back right (x > scan_position_x, y < 0)
            # 3: Front right (x > scan_position_x, y > 0)
            x_pos = marker.pose.position.x
            y_pos = marker.pose.position.y
            
            if x_pos < self.scan_position_x:
                label = 0 if y_pos < 0 else 1
            else:
                label = 2 if y_pos < 0 else 3

            # Train network using each image
            for img in images:
                try:
                    # Convert image to ROS msg
                    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ros_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
                    
                    # Create training request
                    req = SortNet.Request()
                    req.image = ros_msg
                    req.label = label
                    
                    # Call training service with timeout
                    future = self.train_sorting_client.call_async(req)
                    try:
                        await rclpy.task.Future.wait_for(future, timeout=5.0)
                    except TimeoutError:
                        self.get_logger().warn("Training request timed out")
                        continue
                    except Exception as e:
                        self.get_logger().error(f"Error in training request: {str(e)}")
                        continue
                    
                except Exception as e:
                    self.get_logger().error(f"Error training network: {str(e)}")

        self.get_logger().info("Pre-training complete")
        self.block_images = []
        self.block_markers = MarkerArray()
        return response                                                        

    def reset_blocks_callback(self, request, response):
        self.block_markers = MarkerArray()
        self.block_images = []
        self.image_tensors = []
        self.label_tensors = []
    
        return response

    async def train_network_callback(self, request, response):
        # Get block index and label from message
        block_index = request.index
        label = request.label
        images = self.block_images[block_index]

        # Train network using each image 
        for img in images:
            try:
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ros_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
                
                req = SortNet.Request()
                req.image = ros_msg
                req.label = label
                
                future = self.train_sorting_client.call_async(req)
                try:
                    await rclpy.task.Future.wait_for(future, timeout=5.0)
                except TimeoutError:
                    self.get_logger().warn("Training request timed out")
                    continue
                except Exception as e:
                    self.get_logger().error(f"Error in training request: {str(e)}")
                    continue
                    
            except Exception as e:
                self.get_logger().error(f"Error training network: {str(e)}")

        self.get_logger().info("Training complete")
        return response

    async def get_network_prediction_callback(self, request, response):
        block_index = request.index
        images = self.block_images[block_index]
        predictions = []

        self.get_logger().info("Received prediction request.")

        # Get prediction for each image
        for img in images:
            try:
                self.get_logger().info("Converting image.")
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                ros_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')

                self.get_logger().info("Sending request.")
                req = SortNet.Request()
                req.image = ros_msg
                
                # Send request
                future = self.get_sorting_prediction_client.call_async(req)
                
                try:
                    # Wait for result using await - non-blocking
                    result = await future
                    
                    if result is not None:
                        pred = result.prediction
                        predictions.append(pred)
                        self.get_logger().info(f"Got prediction: {pred}")
                    else:
                        self.get_logger().warn("Received null response from prediction service")
                        
                except Exception as e:
                    self.get_logger().error(f"Error waiting for prediction result: {str(e)}")
                    continue
                    
            except Exception as e:
                self.get_logger().error(f"Error in prediction request: {str(e)}")
                
        # Return average prediction
        if predictions:
            response.prediction = float(np.mean(predictions))
            self.get_logger().info(f"Final prediction (average of {len(predictions)} predictions): {response.prediction}")
        else:
            response.prediction = -1.0
            self.get_logger().warn("No predictions received")
            
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
        self.get_logger().info("Received update_markers request")
        new_markers = request.input_markers.markers
        marker_index_list = request.markers_to_update

        self.get_logger().info(f"Updating {len(marker_index_list)} markers")

        # update markers
        for i in range(len(marker_index_list)):
            index = marker_index_list[i]
            marker = new_markers[i]

            self.block_markers.markers[index] = marker

        # Return updated markers
        response.output_markers = self.block_markers
        self.get_logger().info("Sending response")

        return response
    
    def hand_detection_timer_callback(self):
        """Process the latest image for hand detection and publish results."""
        if self.last_img_msg is None:
            return
            
        try:
            # Convert ROS Image to CV2
            cv_image = self.ros2_image_to_cv2(self.last_img_msg, encoding='bgr8')
            if cv_image is None:
                return
                
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.hands.process(rgb_image)
            
            # Check if hands were detected
            hands_present = results.multi_hand_landmarks is not None
            
            # Only publish if state has changed
            if hands_present != self.last_hand_state:
                msg = Bool()
                msg.data = hands_present
                self.hands_detected_pub.publish(msg)
                self.last_hand_state = hands_present
                
                if hands_present:
                    self.get_logger().info('Hands detected')
                else:
                    self.get_logger().info('No hands detected')
                    
        except Exception as e:
            self.get_logger().error(f'Error in hand detection: {str(e)}')
    
    def destroy_node(self):
        """Clean up MediaPipe resources when shutting down."""
        if hasattr(self, 'hands'):
            self.hands.close()
        super().destroy_node()

    def segment_depth(self, rgb_image, depth_image):
        h, w = depth_image.shape
        
        # Convert depth image to 8-bit for processing
        depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=255.0 / (np.max(depth_image) - np.min(depth_image)))

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(depth_image_8bit, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blur, 30, 150)

        # Apply Hough Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 85)

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
        lower_color = np.array([0, 50, 50])  
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
        if self.last_img_msg is None:
            return
        
        markers, img_with_boxes = self.scan_overhead(request.input_markers, list(request.markers_to_update), request.update_scale)

        self.block_markers = markers
        response.output_markers = markers
        img_with_boxes_msg = self.cv2_to_ros2_image(img_with_boxes)
        self.img_msg_out = img_with_boxes_msg
        self.get_logger().info('Block scanning complete!')

        return response

    def scan_overhead(self, markers, markers_to_update, update_scale):
        """Function to scan table to determine approximate size and position of blocks."""
        # Get last image from realsense, convert to cv2
        image = self.ros2_image_to_cv2(self.last_img_msg, encoding='bgr8')
        depth = self.ros2_image_to_cv2(self.last_depth_msg, encoding='8UC1')

        img_table = self.segment_depth(image, depth)
        contours, img_mask = self.segment_image(img_table)

        # Create a copy of the original image to draw boxes on
        img_with_boxes = img_mask.copy()

        buffer = 5

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
            depth_cv_image = self.ros2_image_to_cv2(self.last_depth_msg)
            center_3d = self.get_position_from_image(center_x, center_y, depth_cv_image)

            # Get 3d points for box edges
            points_3d = []
            for point in box:
                pt_3d = self.get_position_from_image(point[0], point[1], depth_cv_image)
                points_3d.append(pt_3d)

            # Extract box side lengths
            side_lengths = []
            for i in range(len(points_3d)):
                length = np.sqrt((points_3d[i-1][0] - points_3d[i][0])**2 + (points_3d[i-1][1] - points_3d[i][1])**2)
                side_lengths.append(length)

            side1_length = (side_lengths[0] + side_lengths[2]) / 2
            side2_length = (side_lengths[1] + side_lengths[3]) / 2

            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.type = 1

            # Create PointStamped object for transformation
            point_stamped = PointStamped()
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.header.frame_id = 'd405_color_optical_frame'
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
                # If Z is very large, assume it is the same as Y
                if marker.scale.z > 2 * marker.scale.y:
                    marker.scale.z = marker.scale.y

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

        return markers, img_with_boxes

    def get_position_from_image(self, u, v, depth_cv_image):
        """Retrieve the 3D coordinates based on the depth image data."""
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            self.get_logger().warn('Camera info not yet received.')
            return

        # Convert the depth image to a Numpy array
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
    
    # Create multithreaded executor
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()