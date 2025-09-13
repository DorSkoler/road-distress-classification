#!/usr/bin/env python3
"""
Simple test to debug visualization creation
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Test basic plotting
def test_basic_plot():
    # Create output directory with absolute path
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, "mlds_final_project_template", "images")
    print(f"Current directory: {current_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory created: {os.path.exists(output_dir)}")
    
    # Create a simple test plot
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.array([1, 2, 3])
    y = np.array([1, 4, 9])
    ax.plot(x, y, 'bo-')
    ax.set_title('Test Plot')
    
    # Save the plot
    test_file = os.path.join(output_dir, "test_plot.png")
    plt.savefig(test_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Test file saved: {os.path.exists(test_file)}")
    if os.path.exists(test_file):
        print(f"File size: {os.path.getsize(test_file)} bytes")

if __name__ == "__main__":
    test_basic_plot()
