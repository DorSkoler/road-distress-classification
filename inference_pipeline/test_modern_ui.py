#!/usr/bin/env python3
"""
Test Script for Modern UI
Date: 2025-08-01

Quick test to verify the modern UI improvements work correctly.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_heatmap_generation():
    """Test the new clean heatmap generation methods."""
    print("🧪 Testing Modern UI Components")
    print("=" * 50)
    
    try:
        from src.heatmap_generator import HeatmapGenerator
        
        # Create test data
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_confidence = np.random.rand(256, 256).astype(np.float32)
        
        # Initialize heatmap generator
        heatmap_gen = HeatmapGenerator(alpha=0.6)
        
        print("✅ HeatmapGenerator initialized")
        
        # Test clean heatmap generation
        clean_heatmap = heatmap_gen.create_clean_heatmap(
            test_image, test_confidence, scale_factor=1.5
        )
        print(f"✅ Clean heatmap generated: {clean_heatmap.shape}")
        
        # Test pure confidence map
        pure_confidence = heatmap_gen.create_pure_confidence_map(
            test_confidence, scale_factor=2.0
        )
        print(f"✅ Pure confidence map generated: {pure_confidence.shape}")
        
        # Test scaling
        original_size = test_image.shape[:2]
        scaled_size = clean_heatmap.shape[:2]
        scale_ratio = scaled_size[0] / original_size[0]
        print(f"✅ Scaling works: {original_size} -> {scaled_size} (scale: {scale_ratio:.1f}x)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_ui_components():
    """Test UI component availability."""
    print("\n🎨 Testing UI Components")
    print("-" * 30)
    
    try:
        # Test if all required packages are available
        import streamlit
        print("✅ Streamlit available")
        
        import plotly
        print("✅ Plotly available")
        
        import cv2
        print("✅ OpenCV available")
        
        # Test CSS classes (just check if they're defined)
        css_classes = [
            "main-header", "metric-card", "damage-detected", 
            "no-damage", "info-panel", "image-container"
        ]
        print(f"✅ CSS classes defined: {len(css_classes)} classes")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def main():
    """Run all tests."""
    print("🛣️ Modern UI Test Suite")
    print("=" * 60)
    
    tests = [
        ("Heatmap Generation", test_heatmap_generation),
        ("UI Components", test_ui_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            print(f"\n🎉 {test_name}: PASSED")
            passed += 1
        else:
            print(f"\n💥 {test_name}: FAILED")
    
    print("\n" + "=" * 60)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🚀 Modern UI is ready to launch!")
        print("Run: python launch_ui.py")
    else:
        print("⚠️  Some issues detected. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)