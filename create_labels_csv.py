#!/usr/bin/env python3
"""
Create CSV labels files from JSON annotations for training
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

def create_labels_csv(images_txt_file: str, coryell_data_dir: str, output_csv: str):
    """
    Create CSV labels file from image list and JSON annotations
    
    Args:
        images_txt_file: Path to txt file containing image paths (e.g., train_images.txt)
        coryell_data_dir: Path to coryell data directory
        output_csv: Output CSV file path
    """
    
    # Read image paths from txt file
    with open(images_txt_file, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    labels_data = []
    
    for img_path in image_paths:
        # img_path format: "Co Rd 232/018_31.615684_-97.742088"
        road_name = img_path.split('/')[0]
        img_name = img_path.split('/')[1]
        
        # Construct full relative path to image file
        # This should be the path relative to the images_dir argument
        image_relative_path = f"{road_name}/img/{img_name}.png"
        ann_file = os.path.join(coryell_data_dir, road_name, 'ann', f"{img_name}.json")
        
        # Load annotation
        damage = 0
        occlusion = 0  
        crop = 0
        
        try:
            with open(ann_file, 'r') as f:
                ann_data = json.load(f)
            
            # Extract labels from tags
            for tag in ann_data.get('tags', []):
                tag_name = tag.get('name', '')
                tag_value = tag.get('value', '')
                
                if tag_name == 'Damage' and tag_value == 'Damaged':
                    damage = 1
                elif tag_name == 'Occlusion' and tag_value == 'Occluded':
                    occlusion = 1
                elif tag_name == 'Crop' and tag_value == 'Cropped':
                    crop = 1
                    
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load annotation for {img_path}: {e}")
        
        labels_data.append({
            'image_name': image_relative_path,  # Full relative path
            'damage': damage,
            'occlusion': occlusion,
            'crop': crop
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(labels_data)
    df.to_csv(output_csv, index=False)
    print(f"Created {output_csv} with {len(df)} samples")
    
    # Print label distribution
    print(f"Label distribution:")
    print(f"  Damage: {df['damage'].sum()}/{len(df)} ({df['damage'].mean():.2%})")
    print(f"  Occlusion: {df['occlusion'].sum()}/{len(df)} ({df['occlusion'].mean():.2%})")
    print(f"  Crop: {df['crop'].sum()}/{len(df)} ({df['crop'].mean():.2%})")

def main():
    # Paths
    coryell_dir = "data/coryell"
    splits_dir = "experiments/2025-07-05_hybrid_training/data/splits"
    
    # Create CSV files for each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_txt = os.path.join(splits_dir, f"{split}_images.txt")
        output_csv = f"{split}_labels.csv"
        
        if os.path.exists(images_txt):
            print(f"\nProcessing {split} split...")
            create_labels_csv(images_txt, coryell_dir, output_csv)
        else:
            print(f"Warning: {images_txt} not found")

if __name__ == "__main__":
    main() 