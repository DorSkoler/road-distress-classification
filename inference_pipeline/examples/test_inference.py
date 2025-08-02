#!/usr/bin/env python3
"""
Test Script for Road Distress Inference Pipeline
Date: 2025-08-01

This script demonstrates how to use the inference pipeline programmatically
and provides examples for different use cases.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src import create_inference_engine, HeatmapGenerator


def test_single_image_inference():
    """Test inference on a single image."""
    print("Testing single image inference...")
    
    # Create a dummy test image (you can replace with real image path)
    test_image = np.random.randint(0, 255, (512, 768, 3), dtype=np.uint8)
    
    try:
        # Initialize inference engine
        engine = create_inference_engine()
        
        # Run inference
        results = engine.predict_single(test_image)
        
        print("Inference Results:")
        print(f"  Overall Confidence: {results['overall_confidence']:.3f}")
        print("  Class Predictions:")
        for class_name, class_result in results['class_results'].items():
            status = "✓" if class_result['prediction'] else "✗"
            print(f"    {status} {class_name.capitalize()}: {class_result['probability']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Single image inference failed: {e}")
        return False


def test_heatmap_generation():
    """Test heatmap generation."""
    print("\nTesting heatmap generation...")
    
    try:
        # Initialize components
        engine = create_inference_engine()
        heatmap_gen = HeatmapGenerator()
        
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Get prediction and confidence map
        results = engine.predict_single(test_image)
        confidence_map, _ = engine.get_damage_confidence_map(test_image)
        
        # Generate heatmap
        heatmap = heatmap_gen.create_damage_confidence_heatmap(
            test_image, confidence_map, results, title="Test Heatmap"
        )
        
        print(f"  Heatmap shape: {heatmap.shape}")
        print(f"  Confidence map range: [{confidence_map.min():.3f}, {confidence_map.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"Heatmap generation failed: {e}")
        return False


def test_regional_analysis():
    """Test regional analysis functionality."""
    print("\nTesting regional analysis...")
    
    try:
        # Initialize engine
        engine = create_inference_engine()
        
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Run regional analysis
        regional_results = engine.analyze_image_regions(test_image, grid_size=(2, 2))
        
        print(f"  Grid size: {regional_results['grid_size']}")
        print(f"  Total regions: {regional_results['total_regions']}")
        print(f"  Damage regions: {regional_results['damage_regions']}")
        print(f"  Damage percentage: {regional_results['damage_percentage']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Regional analysis failed: {e}")
        return False


def test_visualization_types():
    """Test different visualization types."""
    print("\nTesting visualization types...")
    
    try:
        # Initialize components
        engine = create_inference_engine()
        heatmap_gen = HeatmapGenerator()
        
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Get results
        results = engine.predict_single(test_image)
        confidence_map, _ = engine.get_damage_confidence_map(test_image)
        regional_results = engine.analyze_image_regions(test_image, grid_size=(2, 2))
        
        # Test different visualizations
        viz_types = []
        
        # 1. Damage heatmap
        damage_viz = heatmap_gen.create_damage_confidence_heatmap(
            test_image, confidence_map, results
        )
        viz_types.append(("Damage Heatmap", damage_viz.shape))
        
        # 2. Multi-class visualization
        multiclass_viz = heatmap_gen.create_multi_class_visualization(
            test_image, results
        )
        viz_types.append(("Multi-class", multiclass_viz.shape))
        
        # 3. Regional heatmap
        regional_viz = heatmap_gen.create_regional_heatmap(
            test_image, regional_results, focus_class='damage'
        )
        viz_types.append(("Regional", regional_viz.shape))
        
        # 4. Comparison grid
        grid_viz = heatmap_gen.create_comparison_grid(
            test_image, results, confidence_map
        )
        viz_types.append(("Comparison Grid", grid_viz.shape))
        
        print("  Generated visualizations:")
        for viz_name, viz_shape in viz_types:
            print(f"    {viz_name}: {viz_shape}")
        
        return True
        
    except Exception as e:
        print(f"Visualization testing failed: {e}")
        return False


def test_batch_processing():
    """Test batch processing simulation."""
    print("\nTesting batch processing simulation...")
    
    try:
        # Initialize engine
        engine = create_inference_engine()
        
        # Create multiple test images
        test_images = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        # Process batch
        batch_results = engine.predict_batch(test_images)
        
        print(f"  Processed {len(batch_results)} images")
        print(f"  Successful predictions: {sum(1 for r in batch_results if r is not None)}")
        
        # Show summary statistics
        if batch_results and batch_results[0] is not None:
            damage_count = sum(1 for r in batch_results 
                             if r and r['class_results']['damage']['prediction'])
            print(f"  Images with damage: {damage_count}")
        
        return True
        
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("ROAD DISTRESS INFERENCE PIPELINE - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Single Image Inference", test_single_image_inference),
        ("Heatmap Generation", test_heatmap_generation),
        ("Regional Analysis", test_regional_analysis),
        ("Visualization Types", test_visualization_types),
        ("Batch Processing", test_batch_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✓ {test_name} - PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} - FAILED")
        except Exception as e:
            print(f"✗ {test_name} - ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The inference pipeline is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)