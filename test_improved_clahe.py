#!/usr/bin/env python3
"""
Test script to verify improved CLAHE optimization produces diverse parameters
"""

import os
import pandas as pd
import numpy as np
from batch_clahe_optimization import BatchCLAHEOptimizer

def test_improved_optimization():
    """Test that the improved CLAHE optimization produces diverse parameters"""
    print("🧪 Testing Improved CLAHE Optimization")
    print("=" * 60)
    
    # Test with a few images from coryell
    dataset_dir = "coryell"
    output_csv = "test_improved_clahe_params.csv"
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    # Create optimizer
    optimizer = BatchCLAHEOptimizer(dataset_dir, output_csv)
    
    # Limit to 10 images for testing
    original_find = optimizer.find_all_images
    def limited_find():
        all_images = original_find()
        return all_images[:10]
    optimizer.find_all_images = limited_find
    
    print(f"🔍 Testing CLAHE optimization on 10 images...")
    print(f"📊 Parameter ranges:")
    print(f"   Clip limits: {optimizer.clip_limits}")
    print(f"   Grid sizes: {optimizer.tile_grid_sizes}")
    print(f"   Total combinations: {len(optimizer.clip_limits) * len(optimizer.tile_grid_sizes)}")
    print()
    
    # Run optimization
    optimizer.process_all_images()
    
    # Analyze results
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        print("📈 OPTIMIZATION RESULTS:")
        print(f"   Images processed: {len(df)}")
        print()
        
        # Analyze clip limit diversity
        clip_limits = df['clip_limit'].values
        unique_clips = df['clip_limit'].unique()
        print(f"📊 CLIP LIMIT ANALYSIS:")
        print(f"   Unique clip limits found: {len(unique_clips)} out of {len(optimizer.clip_limits)} possible")
        print(f"   Range: {clip_limits.min():.1f} - {clip_limits.max():.1f}")
        print(f"   Mean: {clip_limits.mean():.2f} ± {clip_limits.std():.2f}")
        print(f"   Distribution: {dict(zip(*np.unique(clip_limits, return_counts=True)))}")
        print()
        
        # Analyze grid size diversity
        grid_sizes = list(zip(df['tile_grid_x'], df['tile_grid_y']))
        unique_grids = len(set(grid_sizes))
        print(f"🔲 GRID SIZE ANALYSIS:")
        print(f"   Unique grid sizes found: {unique_grids} out of {len(optimizer.tile_grid_sizes)} possible")
        grid_counts = {}
        for grid in grid_sizes:
            grid_str = f"{grid[0]}x{grid[1]}"
            grid_counts[grid_str] = grid_counts.get(grid_str, 0) + 1
        print(f"   Distribution: {grid_counts}")
        print()
        
        # Check for diversity
        diversity_score = (len(unique_clips) / len(optimizer.clip_limits)) * (unique_grids / len(optimizer.tile_grid_sizes))
        print(f"🎯 DIVERSITY ASSESSMENT:")
        print(f"   Diversity score: {diversity_score:.2f} (0.0 = no diversity, 1.0 = full diversity)")
        
        if diversity_score > 0.3:
            print("   ✅ GOOD: Optimization is producing diverse parameters!")
        elif diversity_score > 0.1:
            print("   ⚠️  MODERATE: Some diversity, but could be improved")
        else:
            print("   ❌ POOR: Low diversity - optimization may not be working well")
        
        print()
        
        # Show detailed results
        print("📋 DETAILED RESULTS:")
        print(df[['image_path', 'clip_limit', 'tile_grid_x', 'tile_grid_y', 'composite_score']].head(10))
        
        # Check if all parameters are the same (the original problem)
        if len(unique_clips) == 1 and unique_grids == 1:
            print("\n❌ PROBLEM: All images got the same parameters!")
            print("This indicates the optimization is still not discriminating well.")
            return False
        else:
            print(f"\n✅ SUCCESS: Found {len(unique_clips)} different clip limits and {unique_grids} different grid sizes!")
            return True
    
    else:
        print("❌ ERROR: Optimization failed - no results file created")
        return False

def test_individual_image_differences():
    """Test individual images to see parameter differences"""
    print("\n" + "=" * 60)
    print("🔍 TESTING INDIVIDUAL IMAGE CHARACTERISTICS")
    print("=" * 60)
    
    from optimize_clahe_for_road_distress import CLAHEOptimizer
    import cv2
    
    # Test with a few different images
    test_images = [
        "coryell/Co Rd 161/img/042_31.463257_-98.086842.png",
        "coryell/Co Rd 161/img/050_31.464269_-98.088006.png", 
        "coryell/Co Rd 161/img/070_31.465499_-98.090865.png"
    ]
    
    results = []
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"📸 Analyzing: {os.path.basename(img_path)}")
            
            # Load image and analyze characteristics
            img = cv2.imread(img_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Image characteristics
                brightness = np.mean(gray)
                contrast = np.std(gray)
                texture = cv2.Laplacian(gray, cv2.CV_64F).var()
                dynamic_range = np.percentile(gray, 95) - np.percentile(gray, 5)
                
                print(f"   Brightness: {brightness:.1f}")
                print(f"   Contrast: {contrast:.1f}")
                print(f"   Texture: {texture:.1f}")
                print(f"   Dynamic Range: {dynamic_range:.1f}")
                
                # Quick CLAHE optimization
                try:
                    optimizer = CLAHEOptimizer(img_path)
                    best_config = optimizer.optimize()
                    
                    result = {
                        'image': os.path.basename(img_path),
                        'brightness': brightness,
                        'contrast': contrast, 
                        'texture': texture,
                        'dynamic_range': dynamic_range,
                        'best_clip': best_config['clip_limit'],
                        'best_grid': best_config['tile_grid_size'],
                        'score': best_config['composite_score']
                    }
                    results.append(result)
                    
                    print(f"   ➜ Best CLAHE: clip={best_config['clip_limit']}, grid={best_config['tile_grid_size']}")
                    print(f"   ➜ Score: {best_config['composite_score']:.4f}")
                    
                except Exception as e:
                    print(f"   ❌ Error optimizing: {str(e)}")
                
                print()
    
    # Compare results
    if len(results) >= 2:
        print("🔍 COMPARISON:")
        clip_limits = [r['best_clip'] for r in results]
        grid_sizes = [r['best_grid'] for r in results]
        
        if len(set(clip_limits)) > 1 or len(set(grid_sizes)) > 1:
            print("✅ SUCCESS: Different images got different optimal parameters!")
            for r in results:
                print(f"   {r['image']}: clip={r['best_clip']}, grid={r['best_grid']}")
        else:
            print("❌ ISSUE: All images still getting same parameters")

if __name__ == "__main__":
    print("🚀 Testing Improved CLAHE Optimization System")
    print()
    
    # Test 1: Batch optimization diversity
    success = test_improved_optimization()
    
    # Test 2: Individual image differences  
    test_individual_image_differences()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ IMPROVED OPTIMIZATION TEST PASSED!")
        print("The system now produces diverse CLAHE parameters for different images.")
    else:
        print("❌ OPTIMIZATION STILL NEEDS IMPROVEMENT")
        print("Consider further tuning the evaluation metrics.")
    print("=" * 60) 