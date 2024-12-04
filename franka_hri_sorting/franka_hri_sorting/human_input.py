"""
Enhanced Human Input node with GUI for sorting decisions and model saving.
Supports both binary (yes/no) and categorical (0-3) decisions.

PUBLISHERS:
    + human_sorting (std_msgs/Int8) - Decision value:
        - Binary mode: 0 (No) or 1 (Yes)
        - Categorical mode: 0-3 representing different categories

CLIENTS:
    + save_sorting_network (SaveModel) - Service to save sorting network
    + save_gesture_network (SaveModel) - Service to save gesture network
"""

import rclpy
from rclpy.node import Node
import tkinter as tk
from tkinter import ttk, messagebox
from std_msgs.msg import Int8
from franka_hri_interfaces.srv import SaveModel
import threading

class HumanInputGUI(Node):
    def __init__(self):
        """Initialize the node, setting up the publisher, services and GUI."""
        super().__init__('human_input')
        
        # Create publisher
        self.sorting_pub = self.create_publisher(Int8, 'human_sorting', 10)
        
        # Create service clients
        self.save_sorting_client = self.create_client(SaveModel, 'save_sorting_network')
        self.save_gesture_client = self.create_client(SaveModel, 'save_gesture_network')
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Human Input Control Panel")
        self.setup_gui()
        
        # Create a thread for ROS spinning
        self.ros_thread = threading.Thread(target=self.ros_spin, daemon=True)
        self.ros_thread.start()

    def setup_gui(self):
        """Set up the enhanced GUI with both binary and categorical inputs."""
        # Configure grid for main window
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        
        # Style configuration
        style = ttk.Style()
        style.configure('Correct.TButton', background='green')
        style.configure('Wrong.TButton', background='red')
        style.configure('Category.TButton', padding=10)
        
        # Create main frames
        binary_frame = ttk.LabelFrame(
            self.root, 
            text="Binary Decision (Yes/No)", 
            padding="10"
        )
        binary_frame.grid(
            row=0, column=0, columnspan=2, 
            padx=10, pady=5, sticky="nsew"
        )
        
        category_frame = ttk.LabelFrame(
            self.root, 
            text="Category Decision (0-3)", 
            padding="10"
        )
        category_frame.grid(
            row=1, column=0, columnspan=2, 
            padx=10, pady=5, sticky="nsew"
        )
        
        save_frame = ttk.LabelFrame(
            self.root, 
            text="Save Models", 
            padding="10"
        )
        save_frame.grid(
            row=2, column=0, columnspan=2, 
            padx=10, pady=5, sticky="nsew"
        )

        # Binary decision buttons
        yes_btn = ttk.Button(
            binary_frame, 
            text="Yes (1)", 
            style='Correct.TButton',
            command=lambda: self.publish_decision(1, 'binary')
        )
        yes_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        no_btn = ttk.Button(
            binary_frame, 
            text="No (0)", 
            style='Wrong.TButton',
            command=lambda: self.publish_decision(0, 'binary')
        )
        no_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Add explanatory label for binary mode
        binary_label = ttk.Label(
            binary_frame,
            text="Use for simple yes/no decisions",
            font=('TkDefaultFont', 9, 'italic')
        )
        binary_label.grid(row=1, column=0, columnspan=2, pady=(0, 5))

        # Category decision buttons
        for i in range(4):
            btn = ttk.Button(
                category_frame,
                text=f"Category {i}",
                style='Category.TButton',
                command=lambda x=i: self.publish_decision(x, 'category')
            )
            btn.grid(row=0, column=i, padx=5, pady=5, sticky="ew")

        # Add explanatory label for category mode
        category_label = ttk.Label(
            category_frame,
            text="Use for multi-category decisions",
            font=('TkDefaultFont', 9, 'italic')
        )
        category_label.grid(row=1, column=0, columnspan=4, pady=(0, 5))
        
        # Save model buttons
        save_sorting_btn = ttk.Button(
            save_frame,
            text="Save Sorting Network",
            command=self.save_sorting_network
        )
        save_sorting_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        save_gesture_btn = ttk.Button(
            save_frame,
            text="Save Gesture Network",
            command=self.save_gesture_network
        )
        save_gesture_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Status label with improved visibility
        self.status_label = ttk.Label(
            self.root, 
            text="Ready",
            font=('TkDefaultFont', 10, 'bold'),
            background='light gray',
            padding=5
        )
        self.status_label.grid(
            row=3, column=0, columnspan=2, 
            padx=5, pady=5, sticky="ew"
        )

    def publish_decision(self, decision, mode):
        """Publish decision with appropriate feedback based on mode."""
        msg = Int8()
        msg.data = decision
        self.sorting_pub.publish(msg)
        
        # Provide appropriate feedback based on mode
        if mode == 'binary':
            status_text = f"Published binary decision: {'Yes' if decision == 1 else 'No'} ({decision})"
        else:
            status_text = f"Published category decision: Category {decision}"
            
        self.status_label.config(text=status_text)
        self.get_logger().info(status_text)

    def save_sorting_network(self):
        """Call service to save sorting network."""
        if not self.save_sorting_client.wait_for_service(timeout_sec=1.0):
            messagebox.showerror(
                "Error", 
                "Save sorting network service not available"
            )
            return
            
        request = SaveModel.Request()
        future = self.save_sorting_client.call_async(request)
        future.add_done_callback(self.handle_save_sorting_response)
        self.status_label.config(text="Saving sorting network...")

    def save_gesture_network(self):
        """Call service to save gesture network."""
        if not self.save_gesture_client.wait_for_service(timeout_sec=1.0):
            messagebox.showerror(
                "Error", 
                "Save gesture network service not available"
            )
            return
            
        request = SaveModel.Request()
        future = self.save_gesture_client.call_async(request)
        future.add_done_callback(self.handle_save_gesture_response)
        self.status_label.config(text="Saving gesture network...")

    def handle_save_sorting_response(self, future):
        """Handle response from save sorting service."""
        try:
            response = future.result()
            if response.success:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Sorting network saved to: {response.filepath}"
                ))
            else:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", "Failed to save sorting network"
                ))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Error", f"Error saving sorting network: {str(e)}"
            ))

    def handle_save_gesture_response(self, future):
        """Handle response from save gesture service."""
        try:
            response = future.result()
            if response.success:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Gesture network saved to: {response.filepath}"
                ))
            else:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", "Failed to save gesture network"
                ))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Error", f"Error saving gesture network: {str(e)}"
            ))

    def ros_spin(self):
        """Spin ROS node in separate thread."""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

    def run(self):
        """Run the GUI."""
        try:
            self.root.mainloop()
        finally:
            # Clean up
            self.destroy_node()

def main(args=None):
    """Main function to initialize and run the GUI node."""
    rclpy.init(args=args)
    node = HumanInputGUI()
    
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure proper cleanup
        rclpy.shutdown()

if __name__ == '__main__':
    main()