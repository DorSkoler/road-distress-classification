import os
import shutil
from tqdm import tqdm
from collections import defaultdict

# Configuration
FILTERED_IMG_DIR = '../filtered'
FILTERED_MASK_DIR = '../filtered_masks'
SPLITS = {
    "train": [f"road_{i}" for i in [
        261, 128, 214, 107, 332, 306, 137, 16, 198, 177, 230, 234, 316, 145, 262,
        250, 265, 321, 245, 251, 238, 48, 342, 272, 178, 12, 193, 331, 164, 216,
        264, 213, 197, 82, 232, 339, 131, 184, 153, 144, 303, 318, 246, 118, 180,
        333, 108, 138, 194, 241, 152, 340, 154, 114, 115, 263, 236, 357, 127, 360,
        147, 334
    ]],
    "val": [f"road_{i}" for i in [
        226, 146, 104, 56, 258, 4235, 247, 96, 155, 338, 281, 163, 160
    ]],
    "test": [f"road_{i}" for i in [
        344, 80, 113, 249, 337, 130, 320, 133, 139, 161, 176, 182, 111, 135, 355
    ]]
}

def ensure_split_dirs():
    """Create train/val/test directories for both images and masks"""
    for split in SPLITS.keys():
        os.makedirs(os.path.join(FILTERED_IMG_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(FILTERED_MASK_DIR, split), exist_ok=True)

def get_road_folder(img_name):
    """Extract road folder name from image name"""
    # Assuming format: road_X/image_name.png
    # Modify this function based on your actual naming convention
    parts = img_name.split('_')
    if len(parts) >= 2:
        return f"road_{parts[1]}"  # Adjust based on your naming pattern
    return "unknown"

def organize_by_roads():
    """Organize images by their road folders"""
    img_files = [f for f in os.listdir(FILTERED_IMG_DIR) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    road_groups = defaultdict(list)
    for img_name in img_files:
        road_folder = get_road_folder(img_name)
        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}_mask.png"
        
        if os.path.exists(os.path.join(FILTERED_MASK_DIR, mask_name)):
            road_groups[road_folder].append((img_name, mask_name))
    
    return road_groups

def main():
    # Setup
    ensure_split_dirs()
    
    # Get images organized by roads
    road_groups = organize_by_roads()
    
    # Print available roads
    print("\nAvailable roads:")
    for road in sorted(road_groups.keys()):
        print(f"{road}: {len(road_groups[road])} images")
    
    # Move files to their respective directories based on road assignments
    print("\nMoving files to split directories...")
    for split, roads in SPLITS.items():
        print(f"\nProcessing {split} split...")
        for road in roads:
            if road in road_groups:
                print(f"  Moving {road} ({len(road_groups[road])} images)...")
                for img_name, mask_name in tqdm(road_groups[road]):
                    # Move image
                    src_img = os.path.join(FILTERED_IMG_DIR, img_name)
                    dst_img = os.path.join(FILTERED_IMG_DIR, split, img_name)
                    shutil.move(src_img, dst_img)
                    
                    # Move mask
                    src_mask = os.path.join(FILTERED_MASK_DIR, mask_name)
                    dst_mask = os.path.join(FILTERED_MASK_DIR, split, mask_name)
                    shutil.move(src_mask, dst_mask)
            else:
                print(f"  Warning: {road} not found in dataset")
    
    # Print summary
    print("\nDataset split complete!")
    for split, roads in SPLITS.items():
        split_dir = os.path.join(FILTERED_IMG_DIR, split)
        if os.path.exists(split_dir):
            n_images = len([f for f in os.listdir(split_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"{split}: {n_images} images")

if __name__ == '__main__':
    main() 