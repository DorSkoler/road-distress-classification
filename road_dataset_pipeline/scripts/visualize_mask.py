import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

# Configuration
FILTERED_IMG_DIR = '../filtered'
FILTERED_MASK_DIR = '../filtered_masks'

def get_random_image_mask_pair():
    """Get a random image and its corresponding mask from any split"""
    splits = ['train', 'val', 'test']
    split = random.choice(splits)
    
    img_dir = os.path.join(FILTERED_IMG_DIR, split)
    mask_dir = os.path.join(FILTERED_MASK_DIR, split)
    
    # Get a random image
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        raise Exception(f"No images found in {img_dir}")
    
    img_name = random.choice(images)
    base_name = os.path.splitext(img_name)[0]
    mask_name = f"{base_name}_mask.png"
    
    return {
        'image_path': os.path.join(img_dir, img_name),
        'mask_path': os.path.join(mask_dir, mask_name),
        'split': split
    }

def visualize_mask_overlay():
    """Visualize an image with its mask overlay"""
    # Get random image-mask pair
    pair = get_random_image_mask_pair()
    
    # Read image and mask
    image = cv2.imread(pair['image_path'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    mask = cv2.imread(pair['mask_path'], cv2.IMREAD_GRAYSCALE)
    
    # Create overlay
    overlay = image.copy()
    overlay[mask > 0] = [255, 0, 0]  # Red overlay for road areas
    
    # Blend original and overlay
    alpha = 0.5
    blended = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    
    # Create figure with subplots
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Mask
    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Road Mask')
    plt.axis('off')
    
    # Overlay
    plt.subplot(133)
    plt.imshow(blended)
    plt.title('Overlay')
    plt.axis('off')
    
    # Add split information
    plt.suptitle(f'Split: {pair["split"]}\nImage: {os.path.basename(pair["image_path"])}', y=1.05)
    
    # Show plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_mask_overlay() 