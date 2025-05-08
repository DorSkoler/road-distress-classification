import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.dataset_organization.loader import OrganizedDatasetLoader


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.

    Args:
        image: Input image (BGR format)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        bgr_clahe: CLAHE-processed image
        original_l: Original L channel
        enhanced_l: Enhanced L channel
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l)
    lab_clahe = cv2.merge((enhanced_l, a, b))
    bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return bgr_clahe, l, enhanced_l


def get_road_polygon(image, simplify_eps=5.0, margin_percent=0.1):
   def get_road_polygon(image, simplify_eps=5.0, margin_percent=0.1):
    """
    Enhanced detection of roads and trails by merging color, texture, and shape cues.

    - Applies bilateral filtering to preserve edges while smoothing noise.
    - Uses multiple color ranges for paved, soil, and grassy paths.
    - Incorporates Sobel gradient magnitude instead of raw Canny to detect softer edges.
    - Dynamically sets morphological kernel size based on image dimensions.
    - Fills holes and removes small artifacts.
    """
    # Pre-filter to reduce noise but keep edges
    filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert to HSV for color segmentation
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

    # Color thresholds
    mask_paved = cv2.inRange(hsv, (0, 0, 120), (180, 60, 255))    # paved/gravel
    mask_soil  = cv2.inRange(hsv, (5, 50, 50), (30, 255, 200))     # dirt
    mask_grass = cv2.inRange(hsv, (25, 30, 30), (85, 255, 255))    # grassy trails
    color_mask = cv2.bitwise_or(mask_paved, mask_soil)
    color_mask = cv2.bitwise_or(color_mask, mask_grass)

    # Texture: Sobel gradient magnitude
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, texture_mask = cv2.threshold(grad_mag, 30, 255, cv2.THRESH_BINARY)

    # Combine color + texture
    combined = cv2.bitwise_or(color_mask, texture_mask)

    # Exclude bottom-left watermark
    h, w = combined.shape
    mw, mh = int(w * margin_percent), int(h * margin_percent)
    mask_ignore = np.ones_like(combined) * 255
    mask_ignore[h-mh:, :mw] = 0
    combined = cv2.bitwise_and(combined, mask_ignore)

    # Morphological operations
    k_size = max(3, int(min(w, h) * 0.005))  # 0.5% of smaller dim
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    clean = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Fill small holes
    contours, _ = cv2.findContours(clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = clean.copy()
    for i, cnt in enumerate(contours):
        cv2.drawContours(mask_filled, [cnt], i, 255, -1)

    # Find external contours for final shape
    ex_contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ex_contours:
        return np.empty((0, 2), np.int32), mask_filled

    # Filter by area
    min_area = w * h * 0.002
    ex_contours = [c for c in ex_contours if cv2.contourArea(c) > min_area]
    if not ex_contours:
        return np.empty((0, 2), np.int32), mask_filled

    # Largest region
    main_contour = max(ex_contours, key=cv2.contourArea)

    # Simplify contour
    approx = cv2.approxPolyDP(main_contour, simplify_eps, True)
    poly = approx.reshape(-1, 2)

    # Final filled mask
    final_mask = np.zeros_like(combined)
    cv2.drawContours(final_mask, [main_contour], -1, 255, -1)

    return poly, final_mask



def get_annotation_info(annotation):
    """
    Extract tags of interest from annotation for guidance.
    """
    info = {tag['name']: tag['value'] for tag in annotation.get('tags', [])}
    return info


def process_and_visualize(images, annotations, num_samples=4):
    """
    For a single image, visualize:
    1. Original image
    2. Road mask on original
    3. LAB histogram
    4. LAB image
    5. Road mask on LAB
    6. CLAHE image
    7. Road mask on CLAHE
    8. CLAHE+LAB image
    9. Road mask on CLAHE+LAB
    """
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    n_rows = 3
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    for img_i in indices:
        img = images[img_i]
        ann = annotations[img_i]
        
        # Get CLAHE parameters based on occlusion
        tags = get_annotation_info(ann)
        if tags.get('Occlusion') == 'Occluded':
            clip, grid = 1.0, (4, 4)
        else:
            clip, grid = 4.0, (16, 16)
            
        # Process images
        # Original and its mask
        poly_orig, mask_orig = get_road_polygon(img)
        overlay_orig = img.copy()
        if poly_orig.size > 0:
            cv2.drawContours(overlay_orig, [poly_orig], -1, (0, 255, 0), 2)
            
        # LAB image and its mask
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        poly_lab, mask_lab = get_road_polygon(lab_img)
        overlay_lab = lab_img.copy()
        if poly_lab.size > 0:
            cv2.drawContours(overlay_lab, [poly_lab], -1, (0, 255, 0), 2)
            
        # CLAHE image and its mask
        clahe_img, _, _ = apply_clahe(img, clip_limit=clip, tile_grid_size=grid)
        poly_clahe, mask_clahe = get_road_polygon(clahe_img)
        overlay_clahe = clahe_img.copy()
        if poly_clahe.size > 0:
            cv2.drawContours(overlay_clahe, [poly_clahe], -1, (0, 255, 0), 2)
            
        # CLAHE+LAB image and its mask
        clahe_lab_img, _, _ = apply_clahe(lab_img, clip_limit=clip, tile_grid_size=grid)
        poly_clahe_lab, mask_clahe_lab = get_road_polygon(clahe_lab_img)
        overlay_clahe_lab = clahe_lab_img.copy()
        if poly_clahe_lab.size > 0:
            cv2.drawContours(overlay_clahe_lab, [poly_clahe_lab], -1, (0, 255, 0), 2)

        # Plotting
        # 1. Original image
        ax = axes[0, 0]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('Original Image')
        ax.axis('off')

        # 2. Mask on original
        ax = axes[0, 1]
        ax.imshow(cv2.cvtColor(overlay_orig, cv2.COLOR_BGR2RGB))
        ax.set_title('Mask on Original')
        ax.axis('off')

        # 3. LAB histogram
        ax = axes[0, 2]
        ax.hist(l.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.5, label='L')
        ax.hist(a.ravel(), bins=256, range=[0, 256], color='red', alpha=0.5, label='A')
        ax.hist(b.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.5, label='B')
        ax.set_title('LAB Histograms')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.legend()

        # 4. LAB image
        ax = axes[1, 0]
        ax.imshow(cv2.cvtColor(lab_img, cv2.COLOR_BGR2RGB))
        ax.set_title('LAB Image')
        ax.axis('off')

        # 5. Mask on LAB
        ax = axes[1, 1]
        ax.imshow(cv2.cvtColor(overlay_lab, cv2.COLOR_BGR2RGB))
        ax.set_title('Mask on LAB')
        ax.axis('off')

        # 6. CLAHE image
        ax = axes[1, 2]
        ax.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'CLAHE Image\n(clip={clip}, grid={grid})')
        ax.axis('off')

        # 7. Mask on CLAHE
        ax = axes[2, 0]
        ax.imshow(cv2.cvtColor(overlay_clahe, cv2.COLOR_BGR2RGB))
        ax.set_title('Mask on CLAHE')
        ax.axis('off')

        # 8. CLAHE+LAB image
        ax = axes[2, 1]
        ax.imshow(cv2.cvtColor(clahe_lab_img, cv2.COLOR_BGR2RGB))
        ax.set_title('CLAHE+LAB Image')
        ax.axis('off')

        # 9. Mask on CLAHE+LAB
        ax = axes[2, 2]
        ax.imshow(cv2.cvtColor(overlay_clahe_lab, cv2.COLOR_BGR2RGB))
        ax.set_title('Mask on CLAHE+LAB')
        ax.axis('off')

        print(f"Image suggested CLAHE params: clip_limit={clip}, tile_grid_size={grid}")

    plt.tight_layout()
    plt.show()


def main():
    dataset_path = 'organized_dataset'
    loader = OrganizedDatasetLoader(dataset_path)
    images, annotations = loader.load_split('train')
    process_and_visualize(images, annotations, num_samples=1)


if __name__ == '__main__':
    main()
