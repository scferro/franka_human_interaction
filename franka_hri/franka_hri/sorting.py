import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import sys
import tty
import termios

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import MarkerArray
from std_srvs.srv import Empty
from std_msgs.msg import String
from franka_hri_interfaces.action import PoseAction
from franka_msgs.action import Homing, Grasp
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from franka_hri_interfaces.srv import CreateBox
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, PointStamped

from geometry_msgs.msg import Quaternion
import tf_transformations

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point

class Sorting(Node):
    def __init__(self):
        super().__init__('sorting')

        # Subscriber to the Image topic
        self.img_sub = self.create_subscription(
            Image,
            '/camera/d405/color/image_rect_raw',
            self.image_callback,
            10)
        self.img_msg_out = Image()

        # Subscriber to the Depth image
        self.depth_sub = self.create_subscription(
            Image, '/camera/d405/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        # Subscriber to the Camera Info topic
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/d405/aligned_depth_to_color/camera_info', self.info_callback, 10)

        # Publisher for MarkerArray
        self.block_pub = self.create_publisher(MarkerArray, 'blocks', 10)

        # Publisher for image with contours
        self.img_out_pub = self.create_publisher(Image, 'img_out', 10)

        # Action client for robot pose
        self.move_to_pose_action_client = ActionClient(
            self, PoseAction, 'move_to_pose', callback_group=ReentrantCallbackGroup())

        # Action clients for gripper homing and grasping
        self.gripper_homing_client = ActionClient(
            self, Homing, 'panda_gripper/homing', callback_group=ReentrantCallbackGroup())
        self.gripper_grasping_client = ActionClient(
            self, Grasp, 'panda_gripper/grasp', callback_group=ReentrantCallbackGroup())

        # Service client to create collision objects
        self.create_box_client = self.create_client(CreateBox, 'create_box')

        # Create testing service
        self.sort_srv = self.create_service(Empty, 'sort_blocks', self.sort_srv_callback)

        # Create TF broadcaster and buffer
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.listener = TransformListener(self.tf_buffer, self)

        # Publish static transform for camera once at startup
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.make_static_transforms()

        # Create timer
        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.timer_count = 0.0

        # Other setup
        self.bridge = CvBridge()
        self.last_img_msg = None
        self.last_depth_msg = None
        self.block_tfs = []

    def timer_callback(self):
        self.timer_count += self.timer_period

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    async def send_pose_goal(self, pose, vel_factor=0.2, accel_factor=0.2):
        """Send a pose goal to the action server."""
        goal = PoseAction.Goal()
        goal.goal_pose = pose
        goal.vel_factor = vel_factor
        goal.accel_factor = accel_factor
        self.get_logger().info(f'Sending goal pose: {goal.goal_pose}')

        # Ensure the action server is ready
        while not self.move_to_pose_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for action server to become available...')

        # Send the goal
        future = await self.move_to_pose_action_client.send_goal_async(goal)
        result = await future.get_result_async()

        if result.result.success:
            self.get_logger().info('Goal succeeded.')
        else:
            self.get_logger().info('Goal failed.')

    async def home_gripper(self):
        """Home the gripper using the homing action client."""
        self.get_logger().info('Homing the gripper...')
        goal = Homing.Goal()

        # Ensure the action server is ready
        while not self.gripper_homing_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for homing action server to become available...')

        future = await self.gripper_homing_client.send_goal_async(goal)
        await future.get_result_async()

    async def grasp_block(self, width=0.03, speed=0.1, force=0.001):
        """Grasp the block using the grasp action client."""
        self.get_logger().info('Grasping the block...')
        goal = Grasp.Goal()
        goal.width = width
        goal.speed = speed
        goal.force = force
        goal.epsilon.inner = 0.05
        goal.epsilon.outer = 0.05

        # Ensure the action server is ready
        while not self.gripper_grasping_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for grasp action server to become available...')

        future = await self.gripper_grasping_client.send_goal_async(goal)
        result = await future.get_result_async()

        if result.result.success:
            self.get_logger().info('Grasp succeeded.')
        else:
            self.get_logger().info('Grasp failed.')

    async def call_create_box_service(self, pose, size, block_id):
        """Asynchronous service call to create a collision box."""
        request = CreateBox.Request()
        request.pose = pose
        request.size.x = size[0]
        request.size.y = size[1]
        request.size.z = size[2]

        string_msg = String()
        string_msg.data = block_id
        request.box_id = string_msg

        # Asynchronous service call
        self.future_box = self.create_box_client.call_async(request)
        self.get_logger().info('Box collision object successfully created')

    async def scan(self):
        """Asynchronous function to detect and process blocks."""
        if self.last_img_msg is None:
            return

        # Extract image from message
        image = self.ros2_image_to_cv2(self.last_img_msg)

        # Convert to grayscale, equalize
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray_img = cv2.equalizeHist(gray_img)

        # Canny edge detection
        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                    pt1, pt2 = simplified_approx[i - 1], simplified_approx[i]
                    distance = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
                    if distance < min_distance:
                        valid = False
                if valid:
                    square_vertices.append(simplified_approx)
                    cv2.drawContours(image_out, [simplified_approx], 0, (0, 255, 0), 3)

        self.img_msg_out = self.cv2_to_ros2_image(image_out)
        self.img_out_pub.publish(self.img_msg_out)

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

            side_lengths = []
            angles = []

            for i in range(4):
                x1_, y1_, z1_ = self.get_position(vert[i-1][0], vert[i-1][1])
                x2_, y2_, z2_ = self.get_position(vert[i][0], vert[i][1])
                dist = np.sqrt((x1_ - x2_)**2 + (y1_ - y2_)**2)
                side_lengths.append(dist)
                angle = np.arctan2((x1_ - x2_), (y1_ - y2_))
                while angle > np.pi/2:
                    angle += -np.pi
                while angle < -np.pi/2:
                    angle += np.pi
                angles.append(angle)

            sides1 = (side_lengths[0] + side_lengths[2]) / 2
            sides2 = (side_lengths[1] + side_lengths[3]) / 2
            angles1 = (angles[0] + angles[2]) / 2
            angles2 = (angles[1] + angles[3]) / 2

            block_angle = angles1

            if sides1 < sides2:
                block_angle = angles2

            block_id = 'block' + str(count)

            # Create PoseStamped object for transformation
            point_stamped = PointStamped()
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.header.frame_id = 'd405_link'
            point_stamped.point.x = x_3d
            point_stamped.point.y = y_3d
            point_stamped.point.z = z_3d

            # Convert to base link frame
            try:
                transform = self.tf_buffer.lookup_transform('world', point_stamped.header.frame_id, rclpy.time.Time())
                world_point = do_transform_point(point_stamped, transform)

                # Get quaternion
                quaternion = tf_transformations.quaternion_from_euler(np.pi, 0, -block_angle)
                quaternion_msg = Quaternion()
                quaternion_msg.x = quaternion[0]
                quaternion_msg.y = quaternion[1]
                quaternion_msg.z = quaternion[2]
                quaternion_msg.w = quaternion[3]

                # Get world transform
                t_world = TransformStamped()
                t_world.header.stamp = self.get_clock().now().to_msg()
                t_world.header.frame_id = 'world'
                t_world.child_frame_id = block_id
                t_world.transform.translation.x = world_point.point.x
                t_world.transform.translation.y = world_point.point.y
                t_world.transform.translation.z = world_point.point.z / 2
                t_world.transform.rotation = quaternion_msg

                # Save TF
                self.block_tfs.append(t_world)

                # Create a Pose object for the collision object
                pose = Pose()
                pose.position.x = world_point.point.x
                pose.position.y = world_point.point.y
                pose.position.z = world_point.point.z / 2
                pose.orientation = quaternion_msg

                # Define the size of the box
                if sides1 < sides2:
                    size = (sides1, sides2, world_point.point.z)
                else:
                    size = (sides2, sides1, world_point.point.z)

                # Call the service to add the collision object
                # await self.call_create_box_service(pose, size, block_id)

                count += 1

            except Exception as e:
                self.get_logger().warn(f"Failed to transform to 'world' frame: {str(e)}")
                continue

    async def sort_srv_callback(self, request, response):
        # Move to the scan pose
        scan_pose = Pose()
        scan_pose.position.x = 0.3
        scan_pose.position.y = 0.0
        scan_pose.position.z = 0.25
        scan_pose.orientation.x = 1.0
        scan_pose.orientation.y = 0.0
        scan_pose.orientation.z = 0.0
        scan_pose.orientation.w = 0.0

        await self.send_pose_goal(scan_pose, vel_factor=0.8, accel_factor=0.2)

        # Await the scan process
        await self.scan()

        if len(self.block_tfs) > 0:
            # Move to a block (example usage)
            grab_pose_1 = Pose()
            grab_pose_1.position.x = self.block_tfs[0].transform.translation.x
            grab_pose_1.position.y = self.block_tfs[0].transform.translation.y
            grab_pose_1.position.z = 0.15

            # Rotate grab angle by 90 degrees
            roll, pitch, yaw = tf_transformations.euler_from_quaternion(
                [self.block_tfs[0].transform.rotation.x, self.block_tfs[0].transform.rotation.y, self.block_tfs[0].transform.rotation.z, self.block_tfs[0].transform.rotation.w]
            )
            yaw += np.pi / 2
            if yaw > np.pi / 2:
                yaw += -np.pi
            elif yaw <-np.pi / 2:
                yaw += np.pi
            quaternion = tf_transformations.quaternion_from_euler(roll, pitch, yaw)

            grab_pose_1.orientation.x = quaternion[0]
            grab_pose_1.orientation.y = quaternion[1]
            grab_pose_1.orientation.z = quaternion[2]
            grab_pose_1.orientation.w = quaternion[3]
            
            # Hover above object
            await self.send_pose_goal(grab_pose_1, vel_factor=0.5, accel_factor=0.1)

            # Move in to grasp
            grab_pose_2 = grab_pose_1
            grab_pose_2.position.z = self.block_tfs[0].transform.translation.z + 0.04

            await self.send_pose_goal(grab_pose_2, vel_factor=0.5, accel_factor=0.1)

            # Grasp block
            await self.grasp_block(0.02)

            rand = np.random.random()

            # Create wait pose
            wait_pose = Pose()
            wait_pose.position.x = 0.3
            wait_pose.position.y = 0.2
            wait_pose.position.z = 0.3
            wait_pose.orientation.x = 1.0
            wait_pose.orientation.y = 0.0
            wait_pose.orientation.z = 0.0
            wait_pose.orientation.w = 0.0

            # Create drop pose
            drop_pose = Pose()
            drop_pose.position.x = 0.3
            drop_pose.position.y = 0.2
            drop_pose.position.z = 0.1
            drop_pose.orientation.x = 1.0
            drop_pose.orientation.y = 0.0
            drop_pose.orientation.z = 0.0
            drop_pose.orientation.w = 0.0

            if rand > 0.5:
                wait_pose.position.y = -wait_pose.position.y

            # Move to wait pose
            await self.send_pose_goal(wait_pose, vel_factor=0.8, accel_factor=0.1)
            
            # Wait for input from user
            self.timer_count = 0.0
            key = ''
            while (self.timer_count < 5.0) and (key != 'l') and (key != 'r'):
                key = self.get_key()

            if (key != 'l') and (key != 'r'):
                drop_pose.position.y = wait_pose.position.y
            elif key=='l':
                drop_pose.position.y = -0.2
            elif key=='r':
                drop_pose.position.y = 0.2

            # Move to drop pose
            await self.send_pose_goal(drop_pose, vel_factor=0.8, accel_factor=0.1)

            # Drop block
            await self.grasp_block(0.04)

        else:
            self.get_logger().warn("No blocks detected")

        await self.send_pose_goal(scan_pose, vel_factor=0.8, accel_factor=0.2)

        return response

    def get_position(self, u, v):
        """Retrieve the 3D coordinates based on the depth image data."""
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            self.get_logger().warn('Camera info not yet received.')
            return

        # Convert the depth image to a Numpy array
        cv_image = self.ros2_image_to_cv2(self.last_depth_msg)
        depth_array = np.array(cv_image, dtype=np.float32)
        z = depth_array[v, u] / 1000.0  # Depth in meters

        half_window = 5
        u_min = max(0, u - half_window)
        u_max = min(depth_array.shape[1], u + half_window + 1)
        v_min = max(0, v - half_window)
        v_max = min(depth_array.shape[0], v + half_window + 1)

        # Extract the window and calculate the average depth, ignoring NaNs and zeros
        window = depth_array[v_min:v_max, u_min:u_max]
        valid_depths = window[np.logical_and(window > 0, ~np.isnan(window))]

        z =  np.mean(valid_depths) / 1000.0  # Convert to meters

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

        t_cam.transform.translation.x = 0.05
        t_cam.transform.translation.y = 0.0
        t_cam.transform.translation.z = 0.04

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

def main(args=None):
    """The main function."""
    rclpy.init(args=args)
    node = Sorting()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
