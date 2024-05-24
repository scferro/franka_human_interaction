import rclpy
from rclpy.node import Node

from std_msgs.msg import Int8

#
# This class defines a node that reads a specific letter
# input from the keyboard, and publishes a corresponding int message
#
# Publishes a 0 for "N" (no) and a 1 for "Y" (yes)
#
class HumanInput(Node):

    # Initialize the node
    def __init__(self):   
        super().__init__('human_input')
        self.sorting_pub = self.create_publisher(Int8, 'human_sorting', 10)
        self.keyPressed = False
        
    # Method for getting key input
    def get_key(self):
        return input('Y for "yes", N for "no", and then Enter: ')
        
    # Spin node
    def spin(self):
        while True:
            self.keyPressed = False
            while rclpy.ok():
                key = self.get_key()
                self.get_logger().info('Pressed: %s' % key.upper()) # Print keypress
                
                msg = Int8()
                
                if key.upper() == 'Y':  # Convert to uppercase and compare key
                    msg.data = 1
                    self.keyPressed = True
                elif key.upper() == 'N':
                    msg.data = 0
                    self.keyPressed = True
                else:
                    print('"%s" is not a valid key' % key.upper()) # Print message data
                    continue
                    
                self.sorting_pub.publish(msg)
                self.get_logger().info('Published: "%d"' % msg.data)


def main(args=None):
    rclpy.init(args=args)
    node = HumanInput()
    node.spin()  
    # Destroy the node explicitly
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()