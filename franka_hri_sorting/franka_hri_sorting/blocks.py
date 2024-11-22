#!/usr/bin/env python3

"""
Blocks node for object detection and manipulation.

This node processes camera images to detect and track blocks. Network operations
are handled by the separate network_node.

PUBLISHERS:
    + img_out (sensor_msgs/Image): Processed image with detected blocks
    + blocks (visualization_msgs/MarkerArray): Detected block markers

SUBSCRIBERS:
    + /camera/d405/color/image_rect_raw (sensor_msgs/Image): Raw color image from camera
    + /camera/d405/aligned_depth_to_color/image_raw (sensor_msgs/Image): Aligned depth image
    + /camera/d405/aligned_depth_to_color/camera_info (sensor_msgs/CameraInfo): Camera information

SERVICES:
    + scan_overhead (franka_hri_interfaces/UpdateMarkers): Scans for blocks and updates markers
    + update_markers (franka_hri_interfaces/UpdateMarkers): Updates existing block markers
    + reset_blocks (std_srvs/Empty): Resets block data
"""

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
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, PointStamped, Quaternion
import tf_transformations
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
import torchvision.transforms as transforms

class Blocks(Node):
    def __init__(self):
        super().__init__('blocks')
        self._init_parameters()
        self._setup_subscribers()
        self._setup_publishers()
        self._setup_services()
        self._setup_timers()
        self._setup_transforms()
        self._init_yolo()
        self._init_bridge()

    def _init_parameters(self):
        """Initialize node parameters and variables."""
        self.last_img_msg = None
        self.last_depth_msg = None
        self.img_msg_out = Image()
        self.block_markers = MarkerArray()
        self.block_images = []
        self.fx = self.fy = self.cx = self.cy = None
        self.timer_count = 0.0

    def _setup_subscribers(self):
        """Set up ROS subscribers."""
        self.create_subscription(Image, '/camera/d405/color/image_rect_raw', self._image_callback, 10)
        self.create_subscription(Image, '/camera/d405/aligned_depth_to_color/image_raw', self._depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/d405/aligned_depth_to_color/camera_info', self._info_callback, 10)

    def _setup_publishers(self):
        """Set up ROS publishers."""
        self.img_out_pub = self.create_publisher(Image, 'img_out', 10)
        self.block_pub = self.create_publisher(MarkerArray, 'blocks', 10)

    def _setup_services(self):
        """Set up ROS services."""
        self.create_service(UpdateMarkers, 'scan_overhead', self._scan_overhead_callback)
        self.create_service(UpdateMarkers, 'update_markers', self._update_markers_callback)
        self.create_service(Empty, 'reset_blocks', self._reset_blocks_callback)
        
        # Create clients for the network services
        self.train_sorting_client = self.create_client(SortNet, 'train_sorting')
        self.get_sorting_prediction_client = self.create_client(SortNet, 'get_sorting_prediction')

    def _setup_timers(self):
        """Set up ROS timers."""
        self.create_timer(0.1, self._timer_callback)
        self.create_timer(0.1, self._marker_timer_callback)

    def _setup_transforms(self):
        """Set up TF2 transformations."""
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.listener = TransformListener(self.tf_buffer, self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self._make_static_transforms()

    def _init_yolo(self):
        """Initialize YOLO object detection model."""
        self.yolo = ultralytics.YOLO("yolov8x.pt")

    def _init_bridge(self):
        """Initialize CvBridge for image conversion."""
        self.bridge = CvBridge()

    def _make_static_transforms(self):
        """Publish static transforms for the camera and world frames."""
        t_cam = TransformStamped()
        t_cam.header.stamp = self.get_clock().now().to_msg()
        t_cam.header.frame_id = 'panda_hand'
        t_cam.child_frame_id = 'd405_link'
        t_cam.transform.translation.x, t_cam.transform.translation.y, t_cam.transform.translation.z = 0.07, 0.0, 0.05
        t_cam.transform.rotation.x, t_cam.transform.rotation.y = 0.0, 0.0
        t_cam.transform.rotation.z, t_cam.transform.rotation.w = 0.7071068, 0.7071068

        t_world = TransformStamped()
        t_world.header.stamp = self.get_clock().now().to_msg()
        t_world.header.frame_id = 'world'
        t_world.child_frame_id = 'panda_link0'
        t_world.transform.translation.x = t_world.transform.translation.y = t_world.transform.translation.z = 0.0
        t_world.transform.rotation.x = t_world.transform.rotation.y = t_world.transform.rotation.z = 0.0
        t_world.transform.rotation.w = 1.0

        self.tf_static_broadcaster.sendTransform([t_cam, t_world])
        self.get_logger().info("Published static transforms.")

    def _image_callback(self, msg):
        """Callback for receiving color images."""
        self.last_img_msg = msg

    def _depth_callback(self, msg):
        """Callback for receiving depth images."""
        self.last_depth_msg = msg

    def _info_callback(self, msg):
        """Callback for receiving camera information."""
        self.fx, self.fy, self.cx, self.cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]

    def _timer_callback(self):
        """Timer callback for general processing."""
        self.timer_count += 0.1

    def _marker_timer_callback(self):
        """Timer callback for publishing processed images and block markers."""
        if self.img_msg_out:
            self.img_out_pub.publish(self.img_msg_out)
            self.block_pub.publish(self.block_markers)

    def _scan_overhead_callback(self, request, response):
        """Service callback for scanning overhead and updating block markers."""
        if self.last_img_msg is None:
            return response
        
        markers, img_with_boxes = self._scan_overhead(request.input_markers, list(request.markers_to_update), request.update_scale)

        self.block_markers = markers
        response.output_markers = markers
        self.img_msg_out = self._cv2_to_ros2_image(img_with_boxes)
        self.get_logger().info('Block scanning complete!')

        return response

    def _update_markers_callback(self, request, response):
        """Service callback to update markers."""
        new_markers = request.input_markers.markers
        marker_index_list = request.markers_to_update

        for i, index in enumerate(marker_index_list):
            self.block_markers.markers[index] = new_markers[i]

        response.output_markers = self.block_markers
        return response

    def _reset_blocks_callback(self, request, response):
        """Reset block data."""
        self.block_markers = MarkerArray()
        self.block_images = []
        return response

    def _scan_overhead(self, markers, markers_to_update, update_scale):
        """
        Scan the overhead view to detect blocks.
        
        Args:
            markers (MarkerArray): Existing markers
            markers_to_update (list): Indices of markers to update
            update_scale (bool): Whether to update marker scale

        Returns:
            tuple: Updated markers and image with bounding boxes
        """
        image = self._ros2_image_to_cv2(self.last_img_msg, encoding='bgr8')
        depth = self._ros2_image_to_cv2(self.last_depth_msg, encoding='8UC1')

        img_table = self._segment_depth(image, depth)
        contours, img_mask = self._segment_image(img_table)

        img_with_boxes = img_mask.copy()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_with_boxes, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
            
            sub_image = self._extract_sub_image(img_table, contour, x, y, w, h)
            resized_image = self._process_sub_image(sub_image)

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_with_boxes, [box], 0, (0, 0, 255), 2)

            marker = self._create_marker(rect, depth, img_mask, contour)
            
            if marker:
                self._update_or_add_marker(marker, markers, markers_to_update, update_scale, resized_image)

        return markers, img_with_boxes

    def _segment_depth(self, rgb_image, depth_image):
        """
        Segment the depth image to isolate the table surface.

        Args:
            rgb_image (np.array): RGB image
            depth_image (np.array): Depth image

        Returns:
            np.array: Segmented table image
        """
        h, w = depth_image.shape
        depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=255.0 / (np.max(depth_image) - np.min(depth_image)))
        blur = cv2.GaussianBlur(depth_image_8bit, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 85)
        mask = np.zeros_like(depth_image_8bit)

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
                x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)

        inverted_mask = cv2.bitwise_not(mask)
        contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        table_mask = np.zeros_like(depth_image_8bit)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(table_mask, [largest_contour], -1, 255, -1)

        kernel = np.ones((11,11), np.uint8)
        closed_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel)
        table_segmented = cv2.bitwise_and(rgb_image, rgb_image, mask=closed_mask)

        return table_segmented

    def _segment_image(self, image):
        """
        Segment the image to isolate blocks.

        Args:
            image (np.array): Input image

        Returns:
            tuple: Contours and masked image
        """
        img_copy = image.copy()
        hsv_image = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
        lower_color = np.array([0, 100, 80])
        upper_color = np.array([180, 255, 255])
        mask = cv2.inRange(hsv_image, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_image = np.zeros_like(image)
        cv2.drawContours(mask_image, contours, -1, (255, 255, 255), -1)
        kernel = np.ones((7,7), np.uint8)
        open_mask = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel)
        masked_image = cv2.bitwise_and(image, open_mask)
        color_mask = cv2.cvtColor(open_mask, cv2.COLOR_BGR2GRAY)
        contours_out, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours_out, masked_image

    def _extract_sub_image(self, img_table, contour, x, y, w, h):
        """
        Extract a sub-image containing a single block.

        Args:
            img_table (np.array): Segmented table image
            contour (np.array): Contour of the block
            x, y, w, h (int): Bounding rectangle coordinates

        Returns:
            np.array: Extracted sub-image
        """
        buffer = 5
        contour_mask = np.zeros_like(img_table)
        cv2.drawContours(contour_mask, [contour], 0, (255, 255, 255), -1)
        sub_image = cv2.bitwise_and(img_table, contour_mask)
        return sub_image[max(0, y-buffer):min(y+h+buffer, sub_image.shape[0]),
                         max(0, x-buffer):min(x+w+buffer, sub_image.shape[1])]

    def _process_sub_image(self, sub_image):
        """
        Process the sub-image to a standard size.

        Args:
            sub_image (np.array): Input sub-image

        Returns:
            np.array: Processed sub-image
        """
        sub_height, sub_width, _ = sub_image.shape
        square_size = max(sub_height, sub_width)
        square_image = np.zeros((square_size, square_size, 3), dtype=np.uint8)
        y_pos = (square_size - sub_height) // 2
        x_pos = (square_size - sub_width) // 2
        square_image[y_pos:y_pos+sub_height, x_pos:x_pos+sub_width] = sub_image
        return cv2.resize(square_image, (128, 128))
    
    def _create_marker(self, rect, depth_image, img_mask, contour):
        """
        Create a marker for a detected block.

        Args:
            rect (tuple): Minimum area rectangle of the block
            depth_image (np.array): Depth image
            img_mask (np.array): Image mask
            contour (np.array): Contour of the block

        Returns:
            Marker: Created marker or None if creation fails
        """
        box_center, box_dimensions, box_angle = rect
        box_angle = -box_angle if box_dimensions[1] > box_dimensions[0] else 90 - box_angle
        box_angle_rad = box_angle / 180 * np.pi

        center_x, center_y = map(int, box_center)
        center_3d = self._get_position_from_image(center_x, center_y, depth_image)

        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = 1

        point_stamped = PointStamped()
        point_stamped.header.stamp = self.get_clock().now().to_msg()
        point_stamped.header.frame_id = 'd405_link'
        point_stamped.point.x, point_stamped.point.y, point_stamped.point.z = center_3d

        try:
            # Transform point from camera frame to world frame
            transform = self.tf_buffer.lookup_transform('world', point_stamped.header.frame_id, rclpy.time.Time())
            world_point = do_transform_point(point_stamped, transform)

            marker.pose.position.x = world_point.point.x
            marker.pose.position.y = world_point.point.y
            marker.pose.position.z = world_point.point.z / 2

            quaternion = tf_transformations.quaternion_from_euler(np.pi, 0, box_angle_rad)
            marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = quaternion

            marker.scale.x, marker.scale.y = max(box_dimensions), min(box_dimensions)
            marker.scale.z = world_point.point.z
            if marker.scale.z > 2 * marker.scale.y:
                marker.scale.z = marker.scale.y

            avg_color = self._get_average_color(img_mask, contour)
            marker.color.r, marker.color.g, marker.color.b = avg_color[2] / 255, avg_color[1] / 255, avg_color[0] / 255
            marker.color.a = 1.0

            return marker
        except Exception as e:
            self.get_logger().info(f"Failed to transform to 'world' frame: {str(e)}")
            return None

    def _update_or_add_marker(self, marker, markers, markers_to_update, update_scale, resized_image):
        """
        Update an existing marker or add a new one.
        [Previous docstring remains the same]
        """
        similar_marker_index = self._find_similar_marker(marker, markers)

        if not update_scale and similar_marker_index != -1:
            marker.scale = markers.markers[similar_marker_index].scale
            marker.pose.position.z = marker.pose.position.z * 2 - (marker.scale.z / 2)

        update_marker = not markers_to_update or (similar_marker_index in markers_to_update)

        if similar_marker_index == -1 and update_marker:
            marker.id = len(markers.markers)
            markers.markers.append(marker)
            self.block_images.append([resized_image])  # Store the CV2 image directly
        elif update_marker:
            marker.id = markers.markers[similar_marker_index].id
            markers.markers[similar_marker_index] = marker
            self.block_images[similar_marker_index].append(resized_image)  # Store the CV2 image directly

    def _find_similar_marker(self, single_marker, marker_array, distance_threshold=0.02, size_threshold=0.03):
        """
        Find a similar marker in the marker array.

        Args:
            single_marker (Marker): Marker to compare
            marker_array (MarkerArray): Array of markers to search
            distance_threshold (float): Maximum distance for similarity
            size_threshold (float): Maximum size difference for similarity

        Returns:
            int: Index of similar marker or -1 if not found
        """
        for i, marker in enumerate(marker_array.markers):
            if marker.type == single_marker.type and marker.action == single_marker.action:
                point1, point2 = single_marker.pose.position, marker.pose.position
                distance = ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2) ** 0.5
                if distance <= distance_threshold:
                    size_difference = self._calculate_size_difference(single_marker.scale, marker.scale)
                    if size_difference <= size_threshold:
                        return i
        return -1

    def _calculate_size_difference(self, scale1, scale2):
        """
        Calculate the size difference between two markers.

        Args:
            scale1, scale2 (Vector3): Scale of markers

        Returns:
            float: Maximum difference in any dimension
        """
        return max(abs(scale1.x - scale2.x), abs(scale1.y - scale2.y), abs(scale1.z - scale2.z))
    
    def _get_average_color(self, color_image, contour):
        """
        Get the average color of a contour in an image.

        Args:
            color_image (np.array): Color image
            contour (np.array): Contour to analyze

        Returns:
            np.array: Average color [B, G, R]
        """
        mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        masked_color = cv2.bitwise_and(color_image, color_image, mask=mask)
        color_values = masked_color[mask > 0]
        return np.mean(color_values, axis=0)

    def _get_position_from_image(self, u, v, depth_cv_image):
        """
        Get 3D position from image coordinates and depth.

        Args:
            u, v (int): Image coordinates
            depth_cv_image (np.array): Depth image

        Returns:
            list: 3D position [X, Y, Z]
        """
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            self.get_logger().warn('Camera info not yet received.')
            return

        depth_array = np.array(depth_cv_image, dtype=np.float32)
        depth_height, depth_width = depth_array.shape

        u = min(u, depth_width - 1)
        v = min(v, depth_height - 1)

        half_window = 11
        u_min, u_max = max(0, u - half_window), min(depth_array.shape[1], u + half_window + 1)
        v_min, v_max = max(0, v - half_window), min(depth_array.shape[0], v + half_window + 1)

        window = depth_array[v_min:v_max, u_min:u_max]
        valid_depths = window[np.logical_and(window > 0, ~np.isnan(window))]
        z = np.mean(valid_depths) / 1000.0  # Convert to meters

        X = (u - self.cx) * z / self.fx
        Y = (v - self.cy) * z / self.fy

        return [X, Y, z]

    async def get_sorting_prediction(self, index):
        """Get a prediction from the sorting network."""
        request = SortNet.Request()
        
        try:
            # Get the most recent image for this block
            image = self.block_images[index][-1]  # Get the most recent image
            
            # Convert to ROS Image message
            ros_msg = self.bridge.cv2_to_imgmsg(image, encoding='rgb8')
            request.image = ros_msg
            
            while not self.get_sorting_prediction_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn('Sorting prediction service not available...')
                
            response = await self.get_sorting_prediction_client.call_async(request)
            return response.prediction
            
        except Exception as e:
            self.get_logger().error(f'Failed to get sorting prediction: {str(e)}')
            return -1

    async def train_sorting_network(self, index, label):
        """Train the sorting network."""
        request = SortNet.Request()
        
        try:
            # Get the most recent image for this block
            image = self.block_images[index][-1]  # Get the most recent image
            
            # Convert to ROS Image message
            ros_msg = self.bridge.cv2_to_imgmsg(image, encoding='rgb8')
            request.image = ros_msg
            request.label = label
            
            while not self.train_sorting_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn('Sorting training service not available...')
                
            response = await self.train_sorting_client.call_async(request)
            self.get_logger().info('Trained sorting network')
            
        except Exception as e:
            self.get_logger().error(f'Failed to train sorting network: {str(e)}')

    def _preprocess_image(self, image):
        """
        Preprocess an image for the network.

        Args:
            image (np.array): Input image

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def _ros2_image_to_cv2(self, ros_image_msg, encoding='passthrough'):
        """
        Convert a ROS2 image message to an OpenCV image.

        Args:
            ros_image_msg (sensor_msgs.msg.Image): ROS2 image message
            encoding (str): Desired encoding for the output image

        Returns:
            np.array: OpenCV image
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding=encoding)
            return cv_image
        except CvBridgeError as e:
            self.get_logger().info(f"Error converting ROS2 image to OpenCV: {e}")
            return None

    def _cv2_to_ros2_image(self, cv_image, encoding="bgr8"):
        """
        Convert an OpenCV image to a ROS2 image message.

        Args:
            cv_image (np.array): OpenCV image
            encoding (str): Encoding of the input image

        Returns:
            sensor_msgs.msg.Image: ROS2 image message
        """
        try:
            ros_image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding=encoding)
            return ros_image_msg
        except CvBridgeError as e:
            self.get_logger().info(f"Error converting OpenCV image to ROS2 message: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = Blocks()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()