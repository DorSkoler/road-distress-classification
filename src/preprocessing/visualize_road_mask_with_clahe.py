import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from dataset_organization.loader import OrganizedDatasetLoader

def get_road_polygon(image, simplify_eps=5.0, margin_percent=0.1):
    """
    Detect the drivable road area using color and edge-based segmentation.
    Ignores the bottom-left corner where watermark is located.
    
    Args:
        image: Input image (BGR format)
        simplify_eps: Douglas–Peucker tolerance in pixels
        margin_percent: Percentage of image width/height to ignore at bottom-left corner
        
    Returns:
        poly: Polygon vertices in (x, y) image coordinates
        road_mask: Binary mask of the road area
    """
    # Convert to HSV color space for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for road detection (adjust these values based on your dataset)
    lower_road = np.array([0, 0, 100])  # Lower HSV values for road
    upper_road = np.array([180, 50, 255])  # Upper HSV values for road
    
    # Create mask based on color
    color_mask = cv2.inRange(hsv, lower_road, upper_road)
    
    # Create a mask to ignore the bottom-left corner
    height, width = image.shape[:2]
    margin_width = int(width * margin_percent)
    margin_height = int(height * margin_percent)
    ignore_mask = np.ones_like(color_mask) * 255
    ignore_mask[height-margin_height:, :margin_width] = 0
    
    # Apply the ignore mask
    color_mask = cv2.bitwise_and(color_mask, ignore_mask)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    road_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.empty((0, 2), np.int32), road_mask
    
    # Find the largest contour (assuming it's the road)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify the contour to get a polygon
    poly = cv2.approxPolyDP(largest_contour, epsilon=simplify_eps, closed=True).squeeze(1)
    
    # Create a clean mask from the largest contour
    road_mask = np.zeros_like(road_mask)
    cv2.drawContours(road_mask, [largest_contour], -1, 255, -1)
    
    return poly, road_mask

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

def visualize_road_clahe_effect(images, annotations, num_samples=3):
    """
    Visualize the effect of CLAHE preprocessing on random images with road detection.
    
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
    clip_limits = [1.0, 2.0, 4.0]
    grid_sizes = [(4, 4), (8, 8), (16, 16)]
    
    for img_idx, (img, annotation) in enumerate(zip(selected_images, selected_annotations)):
        # Get annotation info for this specific image
        annotation_text = get_annotation_info(annotation)
        
        # Print image info
        print(f"\nImage {img_idx + 1} Annotations:")
        print(annotation_text)
        print("-" * 50)
        
        # Create a large figure for all parameter combinations
        fig = plt.figure(figsize=(20, 15))
        
        # Set title with annotation info
        fig.suptitle(f'Image {img_idx + 1}\n{annotation_text}', fontsize=16, y=1.05)
        
        # Original image with road overlay
        plt.subplot(4, 3, 1)
        road_poly, road_mask = get_road_polygon(img)
        overlay = img.copy()
        if len(road_poly) > 0:
            cv2.drawContours(overlay, [road_poly], -1, (0, 255, 0), thickness=3)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title('Original with Road')
        plt.axis('off')
        
        # Road mask
        plt.subplot(4, 3, 2)
        plt.imshow(road_mask, cmap='gray')
        plt.title('Road Mask')
        plt.axis('off')
        
        # Original histogram
        plt.subplot(4, 3, 3)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, _, _ = cv2.split(lab)
        plt.hist(l.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.5)
        plt.title('Original L Channel Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        # Try different parameters
        for i, (clip_limit, grid_size) in enumerate(zip(clip_limits, grid_sizes)):
            # Apply CLAHE
            clahe_img, original_l, enhanced_l = apply_clahe(img, clip_limit, grid_size)
            
            # Get road polygon and mask from CLAHE-processed image
            road_poly_clahe, road_mask_clahe = get_road_polygon(clahe_img)
            
            # Plot CLAHE result with road overlay
            plt.subplot(4, 3, 4 + i*3)
            overlay = clahe_img.copy()
            if len(road_poly_clahe) > 0:
                cv2.drawContours(overlay, [road_poly_clahe], -1, (0, 255, 0), thickness=3)
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title(f'CLAHE (clip={clip_limit}, grid={grid_size})')
            plt.axis('off')
            
            # Plot road mask from CLAHE-processed image
            plt.subplot(4, 3, 5 + i*3)
            plt.imshow(road_mask_clahe, cmap='gray')
            plt.title(f'Road Mask (CLAHE)')
            plt.axis('off')
            
            # Plot histograms
            plt.subplot(4, 3, 6 + i*3)
            plot_histograms(original_l, enhanced_l, plt.gca())
            
            # Print parameter info to console
            print(f"Parameters: clip_limit={clip_limit}, grid_size={grid_size}")
            print("-" * 30)
        
        plt.tight_layout()
        plt.show()

def main():
    # Initialize dataset loader
    dataset_path = "../organized_dataset"
    loader = OrganizedDatasetLoader(dataset_path)
    
    # Load training images and annotations
    train_images, train_annotations = loader.load_split('train')
    
    # Visualize CLAHE effect with road detection
    visualize_road_clahe_effect(train_images, train_annotations, num_samples=3)

if __name__ == '__main__':
    main() 