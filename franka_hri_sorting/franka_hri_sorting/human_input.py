"""
GUI node for human input and corrections during robot learning.
Provides a graphical interface for binary and categorical decisions,
network corrections, and model saving functionality.

PUBLISHERS:
    + human_sorting (std_msgs/Int8) - Human decisions for sorting and classification

SERVICE CLIENTS:
    + save_sorting_network (franka_hri_interfaces/SaveModel) - Save sorting neural network model
    + save_gesture_network (franka_hri_interfaces/SaveModel) - Save gesture neural network model
    + save_complex_gesture_network (franka_hri_interfaces/SaveModel) - Save complex gesture neural network model
    + correct_sorting (franka_hri_interfaces/CorrectionService) - Correct sorting network predictions
    + correct_gesture (franka_hri_interfaces/CorrectionService) - Correct gesture network predictions
    + correct_complex_gesture (franka_hri_interfaces/CorrectionService) - Correct complex gesture network predictions

GUI COMPONENTS:
    + Binary Decision Panel - Buttons for Yes/No decisions
    + Category Decision Panel - Buttons for category selection (0-3)
    + Correction Panel - Interface for correcting network predictions
    + Save Panel - Controls for saving network models
"""


import rclpy
from rclpy.node import Node
import tkinter as tk
from tkinter import ttk, messagebox
from std_msgs.msg import Int8
from franka_hri_interfaces.srv import SaveModel, CorrectionService
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
        self.save_complex_gesture_client = self.create_client(SaveModel, 'save_complex_gesture_network')
        
        # Create correction service clients
        self.correct_sorting_client = self.create_client(CorrectionService, 'correct_sorting')
        self.correct_gesture_client = self.create_client(CorrectionService, 'correct_gesture')
        self.correct_complex_gesture_client = self.create_client(CorrectionService, 'correct_complex_gesture')
        
        # State tracking for corrections
        self.last_predictions = {
            'sorting': None,
            'gesture': None,
            'complex_gesture': None
        }
        
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
        binary_frame = self.create_binary_frame()
        category_frame = self.create_category_frame()
        correction_frame = self.create_correction_frame()
        save_frame = self.create_save_frame()
        
        # Add frames to main window
        binary_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        category_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        correction_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        save_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        
        # Status label with improved visibility
        self.status_label = ttk.Label(
            self.root, 
            text="Ready",
            font=('TkDefaultFont', 10, 'bold'),
            background='light gray',
            padding=5
        )
        self.status_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

    def create_binary_frame(self):
        """Create frame for binary decisions."""
        frame = ttk.LabelFrame(self.root, text="Binary Decision (Yes/No)", padding="10")
        
        yes_btn = ttk.Button(
            frame, 
            text="Yes (1)", 
            style='Correct.TButton',
            command=lambda: self.publish_decision(1, 'binary')
        )
        yes_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        no_btn = ttk.Button(
            frame, 
            text="No (0)", 
            style='Wrong.TButton',
            command=lambda: self.publish_decision(0, 'binary')
        )
        no_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(
            frame,
            text="Use for simple yes/no decisions",
            font=('TkDefaultFont', 9, 'italic')
        ).grid(row=1, column=0, columnspan=2, pady=(0, 5))
        
        return frame

    def create_category_frame(self):
        """Create frame for category decisions."""
        frame = ttk.LabelFrame(self.root, text="Category Decision (0-3)", padding="10")
        
        for i in range(4):
            btn = ttk.Button(
                frame,
                text=f"Category {i}",
                style='Category.TButton',
                command=lambda x=i: self.publish_decision(x, 'category')
            )
            btn.grid(row=0, column=i, padx=5, pady=5, sticky="ew")
        
        ttk.Label(
            frame,
            text="Use for multi-category decisions",
            font=('TkDefaultFont', 9, 'italic')
        ).grid(row=1, column=0, columnspan=4, pady=(0, 5))
        
        return frame

    def create_correction_frame(self):
        """Create frame for correction controls."""
        frame = ttk.LabelFrame(self.root, text="Corrections", padding="10")
        
        # Network selection
        ttk.Label(frame, text="Network:").grid(row=0, column=0, padx=5, pady=5)
        self.network_var = tk.StringVar(value="sorting")
        network_combo = ttk.Combobox(
            frame, 
            textvariable=self.network_var,
            values=["sorting", "gesture", "complex_gesture"],
            state="readonly"
        )
        network_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Old label entry
        ttk.Label(frame, text="Old Label:").grid(row=1, column=0, padx=5, pady=5)
        self.old_label_var = tk.StringVar()
        old_label_entry = ttk.Entry(
            frame, 
            textvariable=self.old_label_var,
            width=10
        )
        old_label_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # New label entry
        ttk.Label(frame, text="New Label:").grid(row=1, column=2, padx=5, pady=5)
        self.new_label_var = tk.StringVar()
        new_label_entry = ttk.Entry(
            frame, 
            textvariable=self.new_label_var,
            width=10
        )
        new_label_entry.grid(row=1, column=3, padx=5, pady=5)
        
        # Apply correction button
        apply_btn = ttk.Button(
            frame,
            text="Apply Correction",
            command=self.apply_correction
        )
        apply_btn.grid(row=2, column=0, columnspan=4, pady=10)
        
        return frame

    def create_save_frame(self):
        """Create frame for save controls."""
        frame = ttk.LabelFrame(self.root, text="Save Models", padding="10")
        
        save_sorting_btn = ttk.Button(
            frame,
            text="Save Sorting Network",
            command=self.save_sorting_network
        )
        save_sorting_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        save_gesture_btn = ttk.Button(
            frame,
            text="Save Gesture Network",
            command=self.save_gesture_network
        )
        save_gesture_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        save_complex_gesture_btn = ttk.Button(
            frame,
            text="Save Complex Gesture Network",
            command=self.save_complex_gesture_network
        )
        save_complex_gesture_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        return frame

    def apply_correction(self):
        """Apply correction to selected network."""
        try:
            network_type = self.network_var.get()
            
            # Get the new label from the entry field
            new_label = int(self.new_label_var.get())
            
            # Use the predicted label as the old label
            old_label = int(self.old_label_var.get())
            
            # Select appropriate service client
            if network_type == 'sorting':
                client = self.correct_sorting_client
            elif network_type == 'gesture':
                client = self.correct_gesture_client
            else:  # complex_gesture
                client = self.correct_complex_gesture_client
                
            # Create and send correction request
            request = CorrectionService.Request()
            request.old_label = old_label
            request.new_label = new_label
            
            if not client.wait_for_service(timeout_sec=1.0):
                self.status_label.config(text=f"Correction service for {network_type} not available")
                return
                
            future = client.call_async(request)
            future.add_done_callback(
                lambda f: self.handle_correction_response(f, network_type))
            
            self.status_label.config(text=f"Applying correction to {network_type} network...")
            
        except ValueError:
            messagebox.showerror("Error", "Labels must be valid integers")
        except Exception as e:
            messagebox.showerror("Error", f"Error applying correction: {str(e)}")

    def handle_correction_response(self, future, network_type):
        """Handle response from correction service."""
        try:
            response = future.result()
            if response.success:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Successfully corrected {network_type} network"
                ))
            else:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Failed to correct {network_type} network: {response.message}"
                ))
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(
                text=f"Error in correction: {str(e)}"
            ))

    def publish_decision(self, decision, mode):
        """Publish decision with appropriate feedback."""
        msg = Int8()
        msg.data = decision
        self.sorting_pub.publish(msg)
        
        if mode == 'binary':
            status_text = f"Published binary decision: {'Yes' if decision == 1 else 'No'} ({decision})"
        else:
            status_text = f"Published category decision: Category {decision}"
            
        self.status_label.config(text=status_text)
        self.get_logger().info(status_text)

    def save_sorting_network(self):
        """Save sorting network."""
        self.save_network('sorting')

    def save_gesture_network(self):
        """Save gesture network."""
        self.save_network('gesture')

    def save_complex_gesture_network(self):
        """Save complex gesture network."""
        self.save_network('complex_gesture')

    def save_network(self, network_type):
        """Generic network saving function."""
        client = getattr(self, f'save_{network_type}_client')
        
        if not client.wait_for_service(timeout_sec=1.0):
            messagebox.showerror(
                "Error", 
                f"Save {network_type} network service not available"
            )
            return
            
        request = SaveModel.Request()
        future = client.call_async(request)
        future.add_done_callback(
            lambda f: self.handle_save_response(f, network_type))
        self.status_label.config(text=f"Saving {network_type} network...")

    def handle_save_response(self, future, network_type):
        """Handle response from save service."""
        try:
            response = future.result()
            if response.success:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"{network_type.title()} network saved to: {response.filepath}"
                ))
            else:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to save {network_type} network"
                ))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Error", f"Error saving {network_type} network: {str(e)}"
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()