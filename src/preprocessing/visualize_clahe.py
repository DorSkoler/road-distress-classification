import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from dataset_organization.loader import OrganizedDatasetLoader

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.
    
    Args:
        image: Input image (BGR format)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Preprocessed image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split into channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    # Merge channels
    lab_clahe = cv2.merge((cl, a, b))
    
    # Convert back to BGR
    bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return bgr_clahe, l, cl

def plot_histograms(original_l, enhanced_l, ax):
    """Plot histograms of original and enhanced L channel"""
    ax.hist(original_l.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.5, label='Original')
    ax.hist(enhanced_l.ravel(), bins=256, range=(0, 256), color='red', alpha=0.5, label='Enhanced')
    ax.set_title('L Channel Histogram')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.legend()

def get_annotation_info(annotation):
    """Extract relevant information from annotation JSON"""
    info = []
    for tag in annotation.get('tags', []):
        if tag['name'] in ['Damage', 'Occlusion', 'Crop']:
            info.append(f"{tag['name']}: {tag['value']}")
    return '\n'.join(info) if info else "No annotations"

def visualize_clahe_effect(images, annotations, num_samples=3):
    """
    Visualize the effect of CLAHE preprocessing on random images with different parameters.
    
    Args:
        images: List of images to sample from
        annotations: List of corresponding annotations
        num_samples: Number of random images to visualize
    """
    # Randomly select images and their annotations
    selected_indices = random.sample(range(len(images)), min(num_samples, len(images)))
    selected_images = [images[i] for i in selected_indices]
    selected_annotations = [annotations[i] for i in selected_indices]
    
    # Different CLAHE parameters to try
    clip_limits = [1.0, 2.0, 4.0, 8.0]
    grid_sizes = [(4, 4), (8, 8), (16, 16), (32, 32)]
    
    for img_idx, (img, annotation) in enumerate(zip(selected_images, selected_annotations)):
        # Create figure for each image
        fig = plt.figure(figsize=(15, 10))
        
        # Get annotation info
        annotation_text = get_annotation_info(annotation)
        
        # Set title with annotation info
        fig.suptitle(f'Image {img_idx + 1}\n{annotation_text}', fontsize=16, y=1.05)
        
        # Original image
        plt.subplot(3, 4, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')
        
        # Original histogram
        plt.subplot(3, 4, 2)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, _, _ = cv2.split(lab)
        plt.hist(l.ravel(), bins=256, range=(0, 256), color='blue')
        plt.title('Original L Channel Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        # Try different parameters
        for i, (clip_limit, grid_size) in enumerate(zip(clip_limits, grid_sizes)):
            # Apply CLAHE
            clahe_img, original_l, enhanced_l = apply_clahe(img, clip_limit, grid_size)
            
            # Plot CLAHE result
            plt.subplot(3, 4, 3 + i*2)
            plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
            plt.title(f'CLAHE (clip={clip_limit}, grid={grid_size})')
            plt.axis('off')
            
            # Plot histograms
            plt.subplot(3, 4, 4 + i*2)
            plot_histograms(original_l, enhanced_l, plt.gca())
        
        plt.tight_layout()
        plt.show()

def main():
    # Initialize dataset loader
    dataset_path = "../organized_dataset"
    loader = OrganizedDatasetLoader(dataset_path)
    
    # Load training images and annotations
    train_images, train_annotations = loader.load_split('train')
    
    # Visualize CLAHE effect with different parameters
    visualize_clahe_effect(train_images, train_annotations, num_samples=3)

if __name__ == '__main__':
    main() 