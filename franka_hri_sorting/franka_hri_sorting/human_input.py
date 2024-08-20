"""
Human Input node for sorting decisions.

This node reads keyboard input from a human operator and publishes
corresponding integer messages for sorting decisions.

PUBLISHERS:
    + human_sorting (std_msgs/Int8) - Sorting decision (0 for "No", 1 for "Yes")

PARAMETERS:
    None
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8

class HumanInput(Node):
    """
    A node that reads keyboard input and publishes sorting decisions.

    This class defines a node that reads specific letter inputs from the keyboard
    and publishes corresponding integer messages for sorting decisions.
    It publishes 0 for "N" (no) and 1 for "Y" (yes).
    """

    def __init__(self):
        """Initialize the node, setting up the publisher and internal state."""
        super().__init__('human_input')
        self.sorting_pub = self.create_publisher(Int8, 'human_sorting', 10)
        self.keyPressed = False
        
    def get_key(self):
        """
        Get keyboard input from the user.

        Returns:
            str: The user's input string.
        """
        return input('Y for "yes", N for "no", and then Enter: ')
        
    def spin(self):
        """
        Main loop of the node.

        Continuously reads keyboard input, processes it, and publishes
        the corresponding sorting decision.
        """
        while True:
            self.keyPressed = False
            while rclpy.ok():
                key = self.get_key()
                self.get_logger().info('Pressed: %s' % key.upper())
                
                msg = Int8()
                
                if key.upper() == 'Y':
                    msg.data = 1
                    self.keyPressed = True
                elif key.upper() == 'N':
                    msg.data = 0
                    self.keyPressed = True
                else:
                    print('"%s" is not a valid key' % key.upper())
                    continue
                    
                self.sorting_pub.publish(msg)
                self.get_logger().info('Published: "%d"' % msg.data)

def main(args=None):
    """
    Main function to initialize and run the HumanInput node.

    Args:
        args: Command-line arguments (default: None)
    """
    rclpy.init(args=args)
    node = HumanInput()
    node.spin()  
    # Destroy the node explicitly
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()