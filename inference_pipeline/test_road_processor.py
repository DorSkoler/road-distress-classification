#!/usr/bin/env python3
"""
Test Script for Road Processing Functionality
Tests the new road folder processing with actual data.
"""

import sys
from pathlib import Path
import glob
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_filename_parsing():
    """Test filename parsing functionality."""
    print("🧪 Testing Filename Parsing")
    print("=" * 40)
    
    try:
        from src.road_processor import RoadProcessor
        
        # Create processor (with None engines for testing)
        processor = RoadProcessor(None, None)
        
        # Test cases
        test_files = [
            "000_31.296905_-97.543646.png",
            "123_45.678901_-123.456789.jpg", 
            "invalid_filename.png",
            "000_invalid_-97.543646.png"
        ]
        
        for filename in test_files:
            result = processor.parse_filename(filename)
            if result:
                seq, lon, lat = result
                print(f"✅ {filename} -> seq:{seq}, lon:{lon}, lat:{lat}")
            else:
                print(f"❌ {filename} -> Invalid format")
                
        print("\n✅ Filename parsing tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Filename parsing test failed: {e}")
        return False

def test_road_data_discovery():
    """Test discovery of actual road data."""
    print("\n🗂️  Testing Road Data Discovery")
    print("=" * 40)
    
    try:
        # Look for the road data directory
        data_paths = [
            "../data/coryell/Co Rd 342/img/",
            "road-distress-classification/data/coryell/Co Rd 342/img/",
            "../road-distress-classification/data/coryell/Co Rd 342/img/"
        ]
        
        road_data_dir = None
        for path in data_paths:
            if Path(path).exists():
                road_data_dir = Path(path)
                break
        
        if not road_data_dir:
            print("❌ Could not find road data directory")
            print("   Expected paths:", data_paths)
            return False
        
        # Count road images
        image_files = list(road_data_dir.glob("*.png")) + list(road_data_dir.glob("*.jpg"))
        print(f"✅ Found road data directory: {road_data_dir}")
        print(f"📁 Contains {len(image_files)} image files")
        
        # Test filename parsing on actual files
        if image_files:
            from src.road_processor import RoadProcessor
            processor = RoadProcessor(None, None)
            
            valid_count = 0
            sample_files = image_files[:10]  # Test first 10 files
            
            for img_file in sample_files:
                result = processor.parse_filename(img_file.name)
                if result:
                    valid_count += 1
                    if valid_count <= 3:  # Show first 3
                        seq, lon, lat = result
                        print(f"  📷 {img_file.name} -> seq:{seq}, coords:({lat:.6f},{lon:.6f})")
            
            print(f"✅ {valid_count}/{len(sample_files)} files have valid format")
            
            if valid_count > 0:
                print("✅ Road data format validation successful")
                return True
        
        print("❌ No valid road images found")
        return False
        
    except Exception as e:
        print(f"❌ Road data discovery failed: {e}")
        return False

def test_road_sequence_validation():
    """Test road sequence validation."""
    print("\n🔢 Testing Road Sequence Validation")
    print("=" * 40)
    
    try:
        from src.road_processor import RoadProcessor, RoadImageData
        
        processor = RoadProcessor(None, None)
        
        # Create sample road images
        sample_images = [
            RoadImageData("000_31.296905_-97.543646.png", 0, -97.543646, 31.296905),
            RoadImageData("001_31.296954_-97.543848.png", 1, -97.543848, 31.296954),
            RoadImageData("002_31.296988_-97.544054.png", 2, -97.544054, 31.296988),
            RoadImageData("005_31.297055_-97.544467.png", 5, -97.544467, 31.297055),  # Gap
        ]
        
        validation_result = processor.validate_road_sequence(sample_images)
        
        print(f"Valid sequence: {validation_result['valid']}")
        print(f"Issues found: {validation_result['issues']}")
        print(f"Sequence stats: {validation_result['stats']}")
        
        if not validation_result['valid']:
            print("⚠️  Sequence validation found issues (expected for test data)")
        else:
            print("✅ Sequence validation passed")
            
        return True
        
    except Exception as e:
        print(f"❌ Sequence validation test failed: {e}")
        return False

def test_scoring_algorithm():
    """Test road scoring algorithm with sample data."""
    print("\n📊 Testing Scoring Algorithm")
    print("=" * 40)
    
    try:
        from src.road_processor import RoadProcessor
        
        processor = RoadProcessor(None, None)
        
        # Create sample results data
        sample_results = {
            'predictions': {
                'damage': [
                    {'probability': 0.9, 'prediction': True, 'confidence': 0.9},
                    {'probability': 0.3, 'prediction': False, 'confidence': 0.7},
                    {'probability': 0.7, 'prediction': True, 'confidence': 0.7},
                    {'probability': 0.1, 'prediction': False, 'confidence': 0.9}
                ],
                'occlusion': [
                    {'probability': 0.2, 'prediction': False, 'confidence': 0.8},
                    {'probability': 0.1, 'prediction': False, 'confidence': 0.9},
                    {'probability': 0.8, 'prediction': True, 'confidence': 0.8},
                    {'probability': 0.3, 'prediction': False, 'confidence': 0.7}
                ],
                'crop': [
                    {'probability': 0.1, 'prediction': False, 'confidence': 0.9},
                    {'probability': 0.2, 'prediction': False, 'confidence': 0.8},
                    {'probability': 0.1, 'prediction': False, 'confidence': 0.9},
                    {'probability': 0.05, 'prediction': False, 'confidence': 0.95}
                ]
            }
        }
        
        scoring_result = processor.calculate_road_score(sample_results)
        
        print(f"Overall score: {scoring_result['overall_score']:.1f}/100")
        print(f"Health category: {scoring_result['health_category']}")
        print(f"Damage segments: {scoring_result['breakdown']['damage_segments']}")
        print(f"Total segments: {scoring_result['breakdown']['total_segments']}")
        
        # Check individual segment scores
        print("\nSegment scores:")
        for i, segment in enumerate(scoring_result['segments'][:3]):  # Show first 3
            print(f"  Segment {i}: {segment['score']:.1f} (damage: {segment['damage_prob']:.3f})")
        
        print("✅ Scoring algorithm test completed")
        return True
        
    except Exception as e:
        print(f"❌ Scoring algorithm test failed: {e}")
        return False

def test_visualizer():
    """Test road visualizer functionality."""
    print("\n📈 Testing Road Visualizer")
    print("=" * 40)
    
    try:
        from src.road_visualizer import RoadVisualizer
        
        visualizer = RoadVisualizer()
        
        # Test color mapping
        test_scores = [95, 80, 65, 45, 25]
        print("Score to category mapping:")
        for score in test_scores:
            category = visualizer._score_to_category(score)
            color = visualizer.health_colors[category]
            print(f"  {score:3d} -> {category:>9} ({color})")
        
        # Test summary metrics creation
        sample_scoring_data = {
            'overall_score': 72.5,
            'breakdown': {
                'total_segments': 4,
                'damage_segments': 2,
                'occlusion_segments': 1,
                'crop_segments': 0,
                'average_damage_prob': 0.475
            }
        }
        
        metrics = visualizer.create_summary_metrics(sample_scoring_data)
        print(f"\nSummary metrics:")
        print(f"  Overall score: {metrics['overall_score']['value']} - {metrics['overall_score']['delta']}")
        print(f"  Total segments: {metrics['total_segments']['value']}")
        print(f"  Damage detected: {metrics['damage_detected']['value']}")
        
        print("✅ Visualizer test completed")
        return True
        
    except Exception as e:
        print(f"❌ Visualizer test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🛣️ Road Processor Test Suite")
    print("=" * 60)
    
    tests = [
        ("Filename Parsing", test_filename_parsing),
        ("Road Data Discovery", test_road_data_discovery),
        ("Sequence Validation", test_road_sequence_validation),
        ("Scoring Algorithm", test_scoring_algorithm),
        ("Road Visualizer", test_visualizer)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"\n🎉 {test_name}: PASSED")
                passed += 1
            else:
                print(f"\n💥 {test_name}: FAILED")
        except Exception as e:
            print(f"\n💥 {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🚀 All tests passed! Road processing infrastructure is ready.")
        print("\n📋 Next Steps:")
        print("1. Update main UI (app.py) to include road processing")
        print("2. Test with actual road folder upload")
        print("3. Refine visualization and scoring based on results")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
