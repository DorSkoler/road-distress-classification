import os
import shutil
from tqdm import tqdm

# Configuration
DATA_DIR = '../../data/coryell'
RAW_DIR = '../raw'
ANNOTATIONS_DIR = '../tagged_json'

def ensure_dirs():
    """Ensure required directories exist"""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

def copy_images_and_annotations():
    """Copy images and their JSON annotations"""
    ensure_dirs()
    
    # Get all road directories
    road_dirs = [d for d in os.listdir(DATA_DIR) 
                if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('Co Rd')]
    
    total_images = 0
    total_annotations = 0
    print(f"Found {len(road_dirs)} road directories")
    
    for road_dir in tqdm(road_dirs, desc="Processing roads"):
        road_path = os.path.join(DATA_DIR, road_dir)
        img_dir = os.path.join(road_path, 'img')
        ann_dir = os.path.join(road_path, 'ann')
        
        if not os.path.exists(img_dir):
            print(f"Warning: No img directory found in {road_dir}")
            continue
            
        # Get all images in the img subdirectory
        images = [f for f in os.listdir(img_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img in images:
            # Copy image
            src_img = os.path.join(img_dir, img)
            road_num = road_dir.replace('Co Rd ', '').replace(' ', '_')
            new_img_name = f"road_{road_num}_{img}"
            dst_img = os.path.join(RAW_DIR, new_img_name)
            shutil.copy2(src_img, dst_img)
            total_images += 1
            
            # Copy corresponding JSON annotation if it exists
            base_name = os.path.splitext(img)[0]
            json_name = f"{base_name}.json"
            src_json = os.path.join(ann_dir, json_name)
            if os.path.exists(src_json):
                dst_json = os.path.join(ANNOTATIONS_DIR, f"road_{road_num}_{json_name}")
                shutil.copy2(src_json, dst_json)
                total_annotations += 1
    
    print(f"\nCopied {total_images} images to {RAW_DIR}")
    print(f"Copied {total_annotations} JSON annotations to {ANNOTATIONS_DIR}")
    print("Files are named as: road_[ROAD_NUMBER]_[ORIGINAL_NAME]")

if __name__ == '__main__':
    copy_images_and_annotations() 