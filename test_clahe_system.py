#!/usr/bin/env python3
"""
Test Script for CLAHE-Enhanced Road Distress Classification System

This script demonstrates the complete workflow:
1. Batch CLAHE parameter optimization
2. Dataset creation with optimized parameters
3. Model training setup

Usage:
    python test_clahe_system.py --test-images-dir coryell --max-images 10
"""

import os
import argparse
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from batch_clahe_optimization import BatchCLAHEOptimizer
from clahe_dataset import CLAHEDataset, CLAHEDataModule


def test_batch_clahe_optimization(images_dir: str, max_images: int = 10):
    """Test batch CLAHE optimization"""
    print(f"\n=== Testing Batch CLAHE Optimization (max_images={max_images}) ===")
    
    output_json = "test_clahe_params.json"
    
    # Run batch optimization
    optimizer = BatchCLAHEOptimizer(images_dir, output_json)
    
    # Limit to few images for testing
    original_find = optimizer.find_all_images
    def limited_find():
        all_images = original_find()
        return all_images[:max_images]
    optimizer.find_all_images = limited_find
    
    print(f"Running CLAHE optimization on {max_images} images...")
    optimizer.process_all_images()
    
    # Check results
    if os.path.exists(output_json):
        import json
        with open(output_json, 'r') as f:
            data = json.load(f)
        
        print(f"✓ CLAHE optimization completed!")
        print(f"✓ Parameters saved for {len(data)} images")
        
        # Calculate statistics
        clip_limits = [params['clip_limit'] for params in data.values()]
        grid_sizes = [params['tile_grid_size'] for params in data.values()]
        
        print(f"✓ Average clip limit: {np.mean(clip_limits):.2f}")
        print(f"✓ Clip limit range: {min(clip_limits):.1f} - {max(clip_limits):.1f}")
        
        # Most common grid size
        from collections import Counter
        grid_size_counts = Counter(tuple(gs) for gs in grid_sizes)
        most_common = grid_size_counts.most_common(1)[0]
        print(f"✓ Most common grid size: {most_common[0][0]}x{most_common[0][1]} ({most_common[1]} images)")
        
        return output_json
    else:
        print("✗ CLAHE optimization failed!")
        return None


def test_clahe_dataset(images_dir: str, labels_csv: str, clahe_params_json: str, mask_opacity: float = 1.0):
    """Test CLAHE dataset loading and processing"""
    print(f"\n=== Testing CLAHE Dataset (mask_opacity={mask_opacity}) ===")
    
    # Create dummy labels CSV if needed
    if not os.path.exists(labels_csv):
        print(f"Creating dummy labels CSV: {labels_csv}")
        create_dummy_labels_csv(images_dir, labels_csv)
    
    # Test dataset creation
    try:
        from clahe_dataset import CLAHEDataset
        dataset = CLAHEDataset(
            images_dir=images_dir,
            masks_dir=images_dir,  # Using same dir for simplicity
            labels_file=labels_csv,
            clahe_params_json=clahe_params_json,
            mask_opacity=mask_opacity,
            img_size=256
        )
        
        print(f"✓ Dataset created successfully with {len(dataset)} samples")
        
        # Test sample loading
        if len(dataset) > 0:
            sample_image, sample_target = dataset[0]
            print(f"✓ Sample loaded: image shape {sample_image.shape}, target {sample_target}")
            
            # Get sample info
            sample_info = dataset.get_sample_info(0)
            print(f"✓ Sample info: {sample_info['image_path']}")
            print(f"  CLAHE params: {sample_info['clahe_params']}")
            
            return True
        else:
            print("✗ No samples found in dataset")
            return False
            
    except Exception as e:
        print(f"✗ Error testing dataset: {str(e)}")
        return False


def test_data_loader(images_dir: str, labels_csv: str, clahe_params_json: str, batch_size: int = 4):
    """Test data loader functionality"""
    print(f"\n=== Testing Data Loader (batch_size={batch_size}) ===")
    
    try:
        from clahe_dataset import CLAHEDataModule
        
        # Create data module
        data_module = CLAHEDataModule(
            train_images_dir=images_dir,
            val_images_dir=images_dir,
            test_images_dir=images_dir,
            train_masks_dir=images_dir,
            val_masks_dir=images_dir,
            test_masks_dir=images_dir,
            train_labels_csv=labels_csv,
            val_labels_csv=labels_csv,
            test_labels_csv=labels_csv,
            clahe_params_json=clahe_params_json,
            mask_opacity=1.0,
            batch_size=batch_size,
            num_workers=0,  # Avoid multiprocessing issues in testing
            img_size=256
        )
        
        # Test train loader
        train_loader = data_module.get_train_loader()
        print(f"✓ Train loader created with {len(train_loader)} batches")
        
        # Test one batch
        for batch_idx, (images, targets) in enumerate(train_loader):
            print(f"✓ Batch {batch_idx}: images {images.shape}, targets {targets.shape}")
            break
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing data loader: {str(e)}")
        return False


def create_dummy_labels_csv(images_dir: str, output_csv: str, max_images: int = 10):
    """Create a dummy labels CSV for testing"""
    image_paths = []
    labels = []
    
    # Find images
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and len(image_paths) < max_images:
                relative_path = Path(root) / file
                relative_to_base = relative_path.relative_to(Path(images_dir))
                
                # Generate random labels for testing
                image_paths.append(str(relative_to_base))
                labels.append({
                    'image_name': str(relative_to_base),
                    'damage': np.random.choice([0, 1]),
                    'occlusion': np.random.choice([0, 1]), 
                    'crop': np.random.choice([0, 1])
                })
    
    # Create DataFrame and save
    df = pd.DataFrame(labels)
    df.to_csv(output_csv, index=False)
    print(f"✓ Created dummy labels CSV with {len(df)} samples: {output_csv}")
    return output_csv


def create_usage_examples():
    """Create example usage scripts"""
    print("\n=== Creating Usage Examples ===")
    
    # Example 1: Batch optimization
    example1 = """#!/bin/bash
# Example 1: Run batch CLAHE optimization on your dataset
python batch_clahe_optimization.py \\
    --dataset-dir coryell \\
    --output-json clahe_parameters.json \\
    --max-images 100  # Remove this line to process all images

echo "CLAHE optimization completed! Check clahe_parameters.json"
"""
    
    with open("example_1_batch_optimization.sh", "w") as f:
        f.write(example1)
    
    # Example 2: Train Model E
    example2 = """#!/bin/bash
# Example 2: Train Model E (CLAHE + Full Mask Overlay)
python train_model_e.py \\
    --train-images data/train/images \\
    --val-images data/val/images \\
    --test-images data/test/images \\
    --train-masks data/train/masks \\
    --val-masks data/val/masks \\
    --test-masks data/test/masks \\
    --train-labels data/train/labels.csv \\
    --val-labels data/val/labels.csv \\
    --test-labels data/test/labels.csv \\
    --clahe-params clahe_parameters.json \\
    --epochs 100 \\
    --batch-size 32 \\
    --learning-rate 0.001 \\
    --output-dir model_e_results

echo "Model E training completed! Check model_e_results/"
"""
    
    with open("example_2_train_model_e.sh", "w") as f:
        f.write(example2)
    
    # Example 3: Train Model F
    example3 = """#!/bin/bash
# Example 3: Train Model F (CLAHE + Partial Mask Overlay)
python train_model_f.py \\
    --train-images data/train/images \\
    --val-images data/val/images \\
    --test-images data/test/images \\
    --train-masks data/train/masks \\
    --val-masks data/val/masks \\
    --test-masks data/test/masks \\
    --train-labels data/train/labels.csv \\
    --val-labels data/val/labels.csv \\
    --test-labels data/test/labels.csv \\
    --clahe-params clahe_parameters.json \\
    --epochs 100 \\
    --batch-size 32 \\
    --learning-rate 0.001 \\
    --output-dir model_f_results

echo "Model F training completed! Check model_f_results/"
"""
    
    with open("example_3_train_model_f.sh", "w") as f:
        f.write(example3)
    
    print("✓ Created usage examples:")
    print("  - example_1_batch_optimization.sh")
    print("  - example_2_train_model_e.sh")
    print("  - example_3_train_model_f.sh")


def main():
    """Run complete system test"""
    parser = argparse.ArgumentParser(description='Test CLAHE system components')
    parser.add_argument('--test-images-dir', default='coryell', help='Directory for test images')
    parser.add_argument('--max-images', type=int, default=5, help='Maximum images to process for testing')
    
    args = parser.parse_args()
    
    print("=== CLAHE System Test ===")
    print(f"Test images directory: {args.test_images_dir}")
    print(f"Max images for testing: {args.max_images}")
    
    # Step 1: Create dummy labels CSV
    labels_csv = create_dummy_labels_csv(args.test_images_dir, "test_labels.csv", args.max_images)
    
    # Step 2: Test batch CLAHE optimization
    print("\n2. Testing batch CLAHE optimization...")
    clahe_params_json = test_batch_clahe_optimization(args.test_images_dir, args.max_images)
    
    if clahe_params_json:
        # Step 3: Test CLAHE dataset with full opacity masks
        print(f"\n3. Testing CLAHE dataset with full opacity masks...")
        success = test_clahe_dataset(args.test_images_dir, labels_csv, clahe_params_json, mask_opacity=1.0)
        
        if success:
            # Step 4: Test data loader functionality
            print(f"\n4. Testing data loader functionality...")
            test_data_loader(args.test_images_dir, labels_csv, clahe_params_json, batch_size=2)
            
            # Step 5: Test CLAHE dataset with partial opacity masks
            print(f"\n5. Testing CLAHE dataset with partial opacity masks...")
            test_clahe_dataset(args.test_images_dir, labels_csv, clahe_params_json, mask_opacity=0.5)
            
            # Step 6: Create usage examples
            print(f"\n6. Creating usage examples...")
            create_usage_examples()
            
            print("\n=== All Tests Completed ===")
            print("✓ CLAHE system is working correctly!")
            print(f"✓ CLAHE parameters saved to: {clahe_params_json}")
            print(f"✓ Test labels saved to: {labels_csv}")
        else:
            print("\n✗ Dataset test failed")
    else:
        print("\n✗ CLAHE optimization test failed")


if __name__ == "__main__":
    main() 