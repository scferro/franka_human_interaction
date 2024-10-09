import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from franka_hri_interfaces.action import DoActionModel
from franka_hri_interfaces.srv import SetIncrement

import numpy as np
import cv2
from cv_bridge import CvBridge
import jax
from collections import deque
from threading import Lock

from octo.model.octo_model import OctoModel

class FrankaOcto(Node):
    def __init__(self):
        super().__init__('franka_octo')

        # Load the pre-trained OctoModel
        self.model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

        # Parameters
        self.declare_parameter('buffer_size', 5)
        self.declare_parameter('inference_frequency', 10.0)  # Hz
        self.declare_parameter('servo_frequency', 10.0)  # Hz

        self.buffer_size = self.get_parameter('buffer_size').value
        self.inference_frequency = self.get_parameter('inference_frequency').value
        self.servo_frequency = self.get_parameter('servo_frequency').value

        # Image buffer and lock
        self.image_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = Lock()

        # CV Bridge for converting between ROS and OpenCV images
        self.cv_bridge = CvBridge()

        # Create a callback group for the action server
        self.cb_group = ReentrantCallbackGroup()

        # Subscribers
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.camera_sub = self.create_subscription(
            Image,
            '/camera',
            self.camera_callback,
            qos_profile
        )

        # Action Server
        self._action_server = ActionServer(
            self,
            DoActionModel,
            '/do_action_octo',
            self.do_action_octo_callback,
            callback_group=self.cb_group
        )

        # Timer for periodic inference
        self.inference_timer = self.create_timer(
            1.0 / self.inference_frequency,
            self.inference_callback
        )

        # Service client for set_increment
        self.set_increment_client = self.create_client(SetIncrement, 'set_increment')

        # Timer for periodic servo updates
        self.servo_timer = self.create_timer(
            1.0 / self.servo_frequency,
            self.servo_callback
        )

        # Store the latest model output
        self.latest_action = None

        self.get_logger().info('OctoModel Node has been initialized')

    def camera_callback(self, msg):
        """Callback for the camera topic subscription."""
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        resized_image = cv2.resize(cv_image, (256, 256))
        with self.buffer_lock:
            self.image_buffer.append(resized_image)

    def inference_callback(self):
        """Callback for periodic inference."""
        with self.buffer_lock:
            if len(self.image_buffer) < 2:
                return  # Not enough images in the buffer

            # Prepare input for the model
            input_images = np.stack(list(self.image_buffer)[-2:])[None]

        observation = {
            'image_primary': input_images,
            'timestep_pad_mask': np.full((1, input_images.shape[1]), True, dtype=bool)
        }

        # If we have an active goal, use it for inference
        if hasattr(self, 'current_goal'):
            actions = self.model.sample_actions(
                observation,
                self.current_goal,
                unnormalization_statistics=self.model.dataset_statistics["bridge_dataset"]["action"],
                rng=jax.random.PRNGKey(0)
            )
            self.latest_action = actions[0]

            # Update the feedback
            self._feedback.progress += 0.1
            self._goal_handle.publish_feedback(self._feedback)

    def servo_callback(self):
        """Callback for periodic servo updates."""
        if self.latest_action is not None:
            request = SetIncrement.Request()
            request.linear.x = float(self.latest_action[0])
            request.linear.y = float(self.latest_action[1])
            request.linear.z = float(self.latest_action[2])
            request.angular.x = float(self.latest_action[3])
            request.angular.y = float(self.latest_action[4])
            request.angular.z = float(self.latest_action[5])

            future = self.set_increment_client.call_async(request)
            future.add_done_callback(self.servo_callback_done)

    def servo_callback_done(self, future):
        try:
            response = future.result()
            if not response.success:
                self.get_logger().warn(f"SetIncrement failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")

    def do_action_octo_callback(self, goal_handle):
        """Callback for the DoActionModel action server."""
        self.get_logger().info('Received action request')

        # Extract goal image and text command from the action goal
        goal_image = self.cv_bridge.imgmsg_to_cv2(goal_handle.request.goal_image, desired_encoding='bgr8')
        text_command = goal_handle.request.text_command

        # Resize goal image
        goal_image = cv2.resize(goal_image, (256, 256))

        # Create task for the model
        self.current_goal = self.model.create_tasks(
            goals={"image_primary": goal_image[None]},
            texts=[text_command]
        )

        # Initialize feedback
        self._feedback = DoActionModel.Feedback()
        self._feedback.progress = 0.0

        # Store the goal handle for publishing feedback
        self._goal_handle = goal_handle

        goal_handle.accept()

        # Run the action for a fixed duration or until a condition is met
        # For this example, we'll run for 10 seconds
        rate = self.create_rate(1)  # 1 Hz
        for _ in range(10):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return DoActionModel.Result(success=False, message='Action canceled')

            rate.sleep()

        # Action completed
        goal_handle.succeed()

        # Clear the current goal and latest action
        del self.current_goal
        self.latest_action = None

        result = DoActionModel.Result()
        result.success = True
        result.message = 'Action completed successfully'
        return result

def main(args=None):
    rclpy.init(args=args)
    franka_octo = FrankaOcto()
    
    # Use a MultiThreadedExecutor to enable processing callbacks from multiple nodes concurrently
    executor = MultiThreadedExecutor()
    executor.add_node(franka_octo)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        franka_octo.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()