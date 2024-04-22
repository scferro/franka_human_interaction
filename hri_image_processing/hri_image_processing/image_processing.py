import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool
from emotions import EmotionDetector

class ImageProcessing(Node):
    def __init__(self):
        super().__init__('image_processing')

        # Declare parameters
        self.declare_parameter('rate', 20.)
        self.declare_parameter('publish_emotions', True)

        # Initialize parameters
        self.rate = self.get_parameter('rate').get_parameter_value().double_value
        self.publish_emotions = self.get_parameter('publish_emotions').get_parameter_value().bool_value

        # Publisher
        self.emotions_pub = self.create_publisher(String, 'emotions_cam', 10)

        # Subscriber
        # self.subscription = self.create_subscription(
        #     String,
        #     'topic',
        #     self.sub_callback,
        #     10)

        # Service
        self.emotions_srv = self.create_service(SetBool, 'enable_emotions', self.emotions_srv_callback)
        
        # Timer
        timer_period = 1 / self.rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        if self.publish_emotions:
            msg = String()
            msg.data = 'Hello, world!'
            self.emotions_pub.publish(msg)
            self.get_logger().info('Publishing: "%s"' % msg.data)

    # def sub_callback(self, msg):
    #     self.get_logger().info('I heard: "%s"' % msg.data)

    def emotions_srv_callback(self, request, response):
        try:
            # Set emotions publishing according to the request
            if request.data:
                self.publish_emotions = True
            else:
                self.publish_emotions = False
            # Report success
            response.success = True
        except:
            # Report failure
            response.success = False
        return response

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessing()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
