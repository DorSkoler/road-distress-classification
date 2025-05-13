import os
import random
import shutil
from tqdm import tqdm
from collections import defaultdict

# Configuration
FILTERED_IMG_DIR = '../filtered'
FILTERED_MASK_DIR = '../filtered_masks'
ANNOTATIONS_DIR = '../tagged_json'
SPLIT_RATIOS = {
    'train': 0.7,  # 70% of roads for training
    'val': 0.15,   # 15% of roads for validation
    'test': 0.15   # 15% of roads for testing
}

def ensure_split_dirs():
    """Create train/val/test directories for images, masks, and annotations"""
    for split in SPLIT_RATIOS.keys():
        # Create directories for images and masks
        os.makedirs(os.path.join(FILTERED_IMG_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(FILTERED_MASK_DIR, split), exist_ok=True)
        # Create directory for annotations
        os.makedirs(os.path.join(ANNOTATIONS_DIR, split), exist_ok=True)

def get_road_folder(img_name):
    """Extract road folder name from image name"""
    # Format: road_X_image_name.png
    parts = img_name.split('_')
    if len(parts) >= 2:
        return f"road_{parts[1]}"  # Returns road_X
    return "unknown"

def organize_by_roads():
    """Organize images and their annotations by road folders"""
    img_files = [f for f in os.listdir(FILTERED_IMG_DIR) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    road_groups = defaultdict(lambda: {'images': [], 'annotations': []})
    
    for img_name in img_files:
        road_folder = get_road_folder(img_name)
        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}_mask.png"
        json_name = f"{base_name}.json"
        
        # Add image and mask if they exist
        if os.path.exists(os.path.join(FILTERED_MASK_DIR, mask_name)):
            road_groups[road_folder]['images'].append((img_name, mask_name))
        
        # Add annotation if it exists
        json_path = os.path.join(ANNOTATIONS_DIR, json_name)
        if os.path.exists(json_path):
            road_groups[road_folder]['annotations'].append(json_name)
    
    return road_groups

def assign_roads_randomly(road_groups):
    """Randomly assign roads to train/val/test splits"""
    roads = list(road_groups.keys())
    random.shuffle(roads)
    
    n_roads = len(roads)
    split_points = [
        int(SPLIT_RATIOS['train'] * n_roads),
        int((SPLIT_RATIOS['train'] + SPLIT_RATIOS['val']) * n_roads)
    ]
    
    splits = {
        'train': roads[:split_points[0]],
        'val': roads[split_points[0]:split_points[1]],
        'test': roads[split_points[1]:]
    }
    
    return splits

def main():
    # Setup
    ensure_split_dirs()
    
    # Get images and annotations organized by roads
    road_groups = organize_by_roads()
    
    # Print available roads
    print("\nAvailable roads:")
    for road in sorted(road_groups.keys()):
        n_images = len(road_groups[road]['images'])
        n_annotations = len(road_groups[road]['annotations'])
        print(f"{road}: {n_images} images, {n_annotations} annotations")
    
    # Randomly assign roads to splits
    splits = assign_roads_randomly(road_groups)
    
    # Print split assignments
    print("\nRoad assignments:")
    for split, roads in splits.items():
        print(f"\n{split}:")
        for road in sorted(roads):
            n_images = len(road_groups[road]['images'])
            n_annotations = len(road_groups[road]['annotations'])
            print(f"  {road}: {n_images} images, {n_annotations} annotations")
    
    # Move files to their respective directories
    print("\nMoving files to split directories...")
    for split, roads in splits.items():
        print(f"\nProcessing {split} split...")
        for road in roads:
            print(f"  Moving {road}...")
            
            # Move images and masks
            for img_name, mask_name in tqdm(road_groups[road]['images'], desc="  Images"):
                # Move image
                src_img = os.path.join(FILTERED_IMG_DIR, img_name)
                dst_img = os.path.join(FILTERED_IMG_DIR, split, img_name)
                shutil.move(src_img, dst_img)
                
                # Move mask
                src_mask = os.path.join(FILTERED_MASK_DIR, mask_name)
                dst_mask = os.path.join(FILTERED_MASK_DIR, split, mask_name)
                shutil.move(src_mask, dst_mask)
            
            # Move annotations
            for json_name in tqdm(road_groups[road]['annotations'], desc="  Annotations"):
                src_json = os.path.join(ANNOTATIONS_DIR, json_name)
                dst_json = os.path.join(ANNOTATIONS_DIR, split, json_name)
                shutil.move(src_json, dst_json)
    
    # Print final summary
    print("\nDataset split complete!")
    for split in SPLIT_RATIOS.keys():
        # Count images
        img_dir = os.path.join(FILTERED_IMG_DIR, split)
        n_images = len([f for f in os.listdir(img_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(img_dir) else 0
        
        # Count annotations
        ann_dir = os.path.join(ANNOTATIONS_DIR, split)
        n_annotations = len([f for f in os.listdir(ann_dir) 
                           if f.endswith('.json')]) if os.path.exists(ann_dir) else 0
        
        print(f"{split}: {n_images} images, {n_annotations} annotations")

if __name__ == '__main__':
    main() 