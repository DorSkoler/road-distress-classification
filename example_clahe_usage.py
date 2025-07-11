#!/usr/bin/env python3
"""
Example usage of the CLAHE optimizer for road distress analysis

This script demonstrates how to use the CLAHEOptimizer class
both programmatically and from command line.
"""

import cv2
import numpy as np
from optimize_clahe_for_road_distress import CLAHEOptimizer
import os


def example_programmatic_usage():
    """Example of using CLAHEOptimizer programmatically"""
    # Example image path - replace with your actual image
    image_path = "your_road_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Please provide a valid image path. Current: {image_path}")
        return
    
    print("=== Programmatic CLAHE Optimization Example ===")
    
    # Initialize optimizer
    optimizer = CLAHEOptimizer(image_path)
    
    # Run optimization
    best_config = optimizer.optimize()
    
    # Apply best configuration to the same image
    enhanced_image = optimizer.apply_best_clahe()
    
    # Apply best configuration to a different image
    # enhanced_other = optimizer.apply_best_clahe("other_road_image.jpg")
    
    # Save results
    optimizer.save_results("clahe_results")
    
    print(f"\nBest CLAHE parameters found:")
    print(f"Clip Limit: {best_config['clip_limit']}")
    print(f"Tile Grid Size: {best_config['tile_grid_size']}")
    
    # Use the parameters in your own code
    clip_limit = best_config['clip_limit']
    tile_grid_size = best_config['tile_grid_size']
    
    # Example of applying to new images with found parameters
    def apply_optimized_clahe(input_image):
        """Apply optimized CLAHE to any image"""
        if isinstance(input_image, str):
            img = cv2.imread(input_image)
        else:
            img = input_image
            
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_l = clahe.apply(l_channel)
        
        lab[:, :, 0] = enhanced_l
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced
    
    print(f"\nYou can now use apply_optimized_clahe() function with your optimized parameters!")


def example_batch_processing():
    """Example of batch processing multiple images with optimized CLAHE"""
    # First, find optimal parameters using one representative image
    reference_image = "reference_road_image.jpg"
    
    if not os.path.exists(reference_image):
        print(f"Please provide a reference image: {reference_image}")
        return
    
    print("=== Batch Processing Example ===")
    
    # Optimize on reference image
    optimizer = CLAHEOptimizer(reference_image)
    best_config = optimizer.optimize()
    
    # Extract parameters
    clip_limit = best_config['clip_limit']
    tile_grid_size = best_config['tile_grid_size']
    
    print(f"Optimized parameters: clip={clip_limit}, grid={tile_grid_size}")
    
    # Apply to batch of images
    input_folder = "road_images/"
    output_folder = "enhanced_road_images/"
    
    if not os.path.exists(input_folder):
        print(f"Create folder with images: {input_folder}")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Process all images in folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"enhanced_{filename}")
            
            # Load and enhance image
            img = cv2.imread(input_path)
            if img is not None:
                enhanced = optimizer.apply_best_clahe(input_path)
                cv2.imwrite(output_path, enhanced)
                print(f"Enhanced: {filename}")


def create_sample_usage_command_line():
    """Show command line usage examples"""
    print("=== Command Line Usage Examples ===")
    print()
    print("Basic usage:")
    print("python optimize_clahe_for_road_distress.py --image road_image.jpg")
    print()
    print("Save detailed results and visualizations:")
    print("python optimize_clahe_for_road_distress.py --image road_image.jpg --save-results")
    print()
    print("Custom output directory:")
    print("python optimize_clahe_for_road_distress.py --image road_image.jpg --save-results --output-dir my_results")
    print()
    print("The script will output:")
    print("- Best CLAHE parameters for your specific image")
    print("- Detailed metrics analysis")
    print("- Visual comparisons (if --save-results)")
    print("- JSON file with all tested configurations")
    print("- Before/after comparison images")


if __name__ == "__main__":
    print("CLAHE Optimization for Road Distress Analysis - Examples")
    print("=" * 60)
    
    create_sample_usage_command_line()
    print()
    
    # Uncomment to run programmatic examples:
    # example_programmatic_usage()
    # example_batch_processing()
    
    print("\nTo test with your images:")
    print("1. Replace 'your_road_image.jpg' with your actual image path")
    print("2. Uncomment the example functions above")
    print("3. Run this script: python example_clahe_usage.py") 