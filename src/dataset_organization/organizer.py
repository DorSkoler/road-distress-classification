import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm

def organize_dataset(source_path: str, output_path: str) -> dict:
    """
    Organize the dataset into a more structured format:
    
    output_path/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── annotations/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── metadata.json
    
    Args:
        source_path: Path to the source dataset
        output_path: Path where the organized dataset will be created
        
    Returns:
        dict: Metadata about the organized dataset
    """
    # Create directory structure
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_path, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'annotations', split), exist_ok=True)
    
    # Initialize metadata
    metadata = {
        'total_images': 0,
        'splits': {split: {'count': 0, 'damage_distribution': {}, 
                          'occlusion_distribution': {}, 'crop_distribution': {}} 
                 for split in splits}
    }
    
    # Process each road directory
    for root, dirs, files in os.walk(source_path):
        if "__pycache__" in root:
            continue
            
        if "img" in dirs and "ann" in dirs:
            img_dir = os.path.join(root, "img")
            ann_dir = os.path.join(root, "ann")
            
            # Get all image files
            img_files = [f for f in os.listdir(img_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
            
            for img_file in tqdm(img_files, desc=f"Processing {os.path.basename(root)}"):
                img_path = os.path.join(img_dir, img_file)
                ann_path = os.path.join(ann_dir, os.path.splitext(img_file)[0] + '.json')
                
                # Determine split (you can modify this logic based on your needs)
                # For now, using a simple modulo-based split
                split_idx = metadata['total_images'] % 10
                split = 'test' if split_idx == 0 else ('val' if split_idx == 1 else 'train')
                
                # Copy image
                new_img_path = os.path.join(output_path, 'images', split, img_file)
                shutil.copy2(img_path, new_img_path)
                
                # Copy and process annotation
                if os.path.exists(ann_path):
                    new_ann_path = os.path.join(output_path, 'annotations', split, 
                                              os.path.splitext(img_file)[0] + '.json')
                    shutil.copy2(ann_path, new_ann_path)
                    
                    # Update metadata
                    with open(ann_path, 'r') as f:
                        ann_data = json.load(f)
                        
                    for tag in ann_data['tags']:
                        if tag['name'] == 'Damage':
                            metadata['splits'][split]['damage_distribution'][tag['value']] = \
                                metadata['splits'][split]['damage_distribution'].get(tag['value'], 0) + 1
                        elif tag['name'] == 'Occlusion':
                            metadata['splits'][split]['occlusion_distribution'][tag['value']] = \
                                metadata['splits'][split]['occlusion_distribution'].get(tag['value'], 0) + 1
                        elif tag['name'] == 'Crop':
                            metadata['splits'][split]['crop_distribution'][tag['value']] = \
                                metadata['splits'][split]['crop_distribution'].get(tag['value'], 0) + 1
                
                metadata['splits'][split]['count'] += 1
                metadata['total_images'] += 1
    
    # Save metadata
    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return metadata 