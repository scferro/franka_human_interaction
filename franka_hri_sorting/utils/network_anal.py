import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from datetime import datetime
import argparse

class NetworkAnalyzer:
    def __init__(self, log_file: str, network_type: str):
        self.log_file = Path(log_file)
        self.network_type = network_type
        
        # Plot style configuration
        plt.style.use('seaborn')
        self.colors = {
            'correct': '#2ecc71',    # Green for correct predictions
            'incorrect': '#e74c3c',   # Red for incorrect predictions
            'confidence': '#3498db',  # Blue for confidence line
            'average': '#f1c40f'      # Yellow for running average
        }

    def transform_confidence(self, confidence_values: np.ndarray) -> np.ndarray:
        """
        Transform confidence values to always be >0.5 for binary classification.
        For values below 0.5, returns 1-confidence.
        
        Args:
            confidence_values (np.ndarray): Array of original confidence values
            
        Returns:
            np.ndarray: Array of transformed confidence values
        """
        transformed = np.where(confidence_values < 0.5, 1 - confidence_values, confidence_values)
        return transformed

    def load_data(self):
        """Load and preprocess the prediction log data."""
        if not self.log_file.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_file}")
            
        # Load the CSV data
        self.data = pd.read_csv(self.log_file)
        
        # Add iteration column - simple index starting from 1
        self.data['iteration'] = np.arange(1, len(self.data) + 1)
        
        # Convert pandas DataFrame columns to numpy arrays for plotting
        self.iterations = self.data['iteration'].to_numpy()
        raw_confidence = self.data['confidence'].to_numpy()
        self.confidence = self.transform_confidence(raw_confidence)
        self.was_correct = self.data['was_correct'].to_numpy()

    def calculate_running_average(self, window_size=10):
        """
        Calculate running average of correct predictions using numpy.
        
        Args:
            window_size (int): Size of the sliding window for average calculation
            
        Returns:
            numpy.ndarray: Array of running averages
        """
        # Convert boolean values to integers for averaging
        correct_ints = self.was_correct.astype(int)
        
        # Create the running average using convolution
        kernel = np.ones(window_size) / window_size
        running_avg = np.convolve(correct_ints, kernel, mode='valid')
        
        # Pad the beginning to maintain array length
        # Use the first calculated average value for padding
        padding = np.full(window_size - 1, running_avg[0])
        return np.concatenate([padding, running_avg])

    def plot_prediction_accuracy(self, window_size=10):
        """
        Create a plot showing correct/incorrect predictions over iterations.
        
        Args:
            window_size (int): Size of the window for calculating running average
            
        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create boolean masks as numpy arrays
        correct_mask = self.was_correct
        incorrect_mask = ~correct_mask
        
        # Plot correct predictions
        ax.scatter(
            self.iterations[correct_mask],
            np.ones_like(self.iterations[correct_mask]),
            color=self.colors['correct'],
            label='Correct',
            alpha=0.6
        )
        
        # Plot incorrect predictions
        ax.scatter(
            self.iterations[incorrect_mask],
            np.zeros_like(self.iterations[incorrect_mask]),
            color=self.colors['incorrect'],
            label='Incorrect',
            alpha=0.6
        )
        
        # Calculate and plot running average
        running_avg = self.calculate_running_average(window_size)
        ax.plot(
            self.iterations,
            running_avg,
            color=self.colors['average'],
            label=f'Running Average (n={window_size})',
            linewidth=2
        )
        
        # Customize plot
        ax.set_xlabel('Training Iterations')
        ax.set_ylabel('Prediction Accuracy')
        ax.set_title(f'{self.network_type.title()} Network Prediction Accuracy by Iteration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits with padding
        ax.set_ylim(-0.1, 1.1)
        
        # Set x-axis to show whole numbers only
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        return fig

    def plot_prediction_confidence(self):
        """
        Create a plot showing prediction confidence over iterations.
        
        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create masks as numpy arrays
        correct_mask = self.was_correct
        incorrect_mask = ~correct_mask
        
        # Plot confidence line
        ax.plot(
            self.iterations,
            self.confidence,  # Now using transformed confidence values
            color=self.colors['confidence'],
            alpha=0.8,
            label='Prediction Confidence'
        )
        
        # Plot scatter points for correct/incorrect predictions
        ax.scatter(
            self.iterations[correct_mask],
            self.confidence[correct_mask],  # Using transformed confidence values
            color=self.colors['correct'],
            alpha=0.4,
            label='Correct Predictions'
        )
        ax.scatter(
            self.iterations[incorrect_mask],
            self.confidence[incorrect_mask],  # Using transformed confidence values
            color=self.colors['incorrect'],
            alpha=0.4,
            label='Incorrect Predictions'
        )
        
        # Customize plot
        ax.set_xlabel('Training Iterations')
        ax.set_ylabel('Confidence')
        ax.set_title(f'{self.network_type.title()} Network Prediction Confidence by Iteration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits with padding
        ax.set_ylim(0.5, 1.1)  # Updated to start at 0.5 since that's our minimum now
        
        # Set x-axis to show whole numbers only
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        return fig

def main():
    """Main function to handle command line arguments and run analysis."""
    parser = argparse.ArgumentParser(description='Analyze network prediction logs')
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='/home/scferro/Documents/final_project/hri_data/logs/colors_demo.csv',
        help='Path to the specific log file to analyze'
    )
    
    parser.add_argument(
        '--network-type',
        type=str,
        choices=['sorting', 'gesture', 'complex_gesture'],
        default='sorting',
        help='Type of network to analyze'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        default=10,
        help='Window size for calculating running average'
    )
    
    args = parser.parse_args()
    
    try:
        # Create analyzer and generate plots
        analyzer = NetworkAnalyzer(args.log_file, args.network_type)
        analyzer.load_data()
        
        # Create both plots
        acc_fig = analyzer.plot_prediction_accuracy(args.window_size)
        conf_fig = analyzer.plot_prediction_confidence()
        
        # Show plots
        plt.show()
        
    except Exception as e:
        print(f"Error analyzing network data: {str(e)}")
        return 1
        
    return 0

if __name__ == '__main__':
    exit(main())