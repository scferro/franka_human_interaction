import rclpy
from rclpy.node import Node

from std_msgs.msg import Int8

#
# This class defines a node that reads a specific letter
# input from the keyboard, and publishes a corresponding int message
#
# Publishes a 0 for "L" (left) and a 1 for "R" (right)
#
class HumanInput(Node):

    # Initialize the node
    def __init__(self):   
        super().__init__('human_input')
        self.publisher_ = self.create_publisher(Int8, 'keyboard', 10)
        self.keyPressed = False
        
    # Method for getting key input
    def get_key(self):
        return input('R for Right, L for Left, and then Enter: ')
        
    # Spin node
    def spin(self):
        while rclpy.ok() and not self.keyPressed:
            key = self.get_key()
            self.get_logger().info('Pressed: %s' % key.upper()) # Print keypress
            
            msg = Int8()
            
            if key.upper() == 'R':  # Convert to uppercase and compare key
                msg.data = 1
                self.keyPressed = True
            elif key.upper() == 'L':
                msg.data = 0
                self.keyPressed = True
            else:
                print('"%s" is not a valid key' % key.upper()) # Print message data
                continue
                
            self.publisher_.publish(msg)
            self.get_logger().info('Published: "%d"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    key_publisher = HumanInput()

    key_publisher.spin()  

    # Destroy the node explicitly
    key_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()