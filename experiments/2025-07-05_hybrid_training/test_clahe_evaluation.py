#!/usr/bin/env python3
"""
Test script to verify CLAHE evaluation is working correctly for models G and H.
"""

import sys
from pathlib import Path
import yaml
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from evaluation.model_evaluator import ModelEvaluator
from data.dataset import create_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_clahe_integration():
    """Test CLAHE integration for models G and H."""
    
    print("🧪 Testing CLAHE Evaluation Integration")
    print("=" * 60)
    
    # Load config
    config_path = "config/base_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test 1: Dataset CLAHE Configuration
    print("\n1️⃣ Testing Dataset CLAHE Configuration:")
    
    for variant in ['model_g', 'model_h']:
        try:
            dataset = create_dataset('test', config, variant, use_augmented=False)
            
            # Enable dynamic CLAHE
            dataset.use_dynamic_clahe_optimization = True
            
            print(f"   ✅ {variant}:")
            print(f"      - Use CLAHE: {dataset.use_clahe}")
            print(f"      - Use Masks: {dataset.use_masks}")
            print(f"      - Dynamic CLAHE: {getattr(dataset, 'use_dynamic_clahe_optimization', False)}")
            print(f"      - Samples: {len(dataset)}")
            
        except Exception as e:
            print(f"   ❌ {variant}: {e}")
    
    # Test 2: Model Evaluator Configuration
    print("\n2️⃣ Testing Model Evaluator Configuration:")
    
    try:
        evaluator = ModelEvaluator(config_path, "results")
        print(f"   ✅ ModelEvaluator initialized")
        print(f"      - CLAHE variants: {evaluator.clahe_variants}")
        print(f"      - Device: {evaluator.device}")
        
    except Exception as e:
        print(f"   ❌ ModelEvaluator: {e}")
    
    # Test 3: CLAHE Optimization Import
    print("\n3️⃣ Testing CLAHE Optimization Import:")
    
    try:
        # Test import path
        batch_clahe_path = Path(__file__).parent.parent.parent / "batch_clahe_optimization.py"
        print(f"   📁 Batch CLAHE path: {batch_clahe_path}")
        print(f"   📁 File exists: {batch_clahe_path.exists()}")
        
        if batch_clahe_path.exists():
            sys.path.insert(0, str(batch_clahe_path.parent))
            from batch_clahe_optimization import SimpleCLAHEOptimizer
            print(f"   ✅ SimpleCLAHEOptimizer imported successfully")
            
            # Test with dummy image
            import numpy as np
            dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            optimizer = SimpleCLAHEOptimizer(dummy_image)
            print(f"   ✅ SimpleCLAHEOptimizer created with dummy image")
            
        else:
            print(f"   ❌ batch_clahe_optimization.py not found")
            
    except Exception as e:
        print(f"   ❌ CLAHE import: {e}")
    
    # Test 4: Model Directory Check
    print("\n4️⃣ Testing Model Directory Structure:")
    
    results_dir = Path("results")
    for variant in ['model_g', 'model_h']:
        model_dir = results_dir / variant
        checkpoint_dir = model_dir / "checkpoints"
        checkpoint_file = checkpoint_dir / "best_model.pth"
        
        print(f"   📁 {variant}:")
        print(f"      - Model dir exists: {model_dir.exists()}")
        print(f"      - Checkpoint dir exists: {checkpoint_dir.exists()}")
        print(f"      - Best model exists: {checkpoint_file.exists()}")
    
    print("\n🎯 Test Summary:")
    print("   - If all tests pass, CLAHE evaluation should work correctly")
    print("   - Models G & H should use dynamic CLAHE optimization")
    print("   - Each test image will be optimized individually during evaluation")
    
    return True

def test_single_model_evaluation():
    """Test evaluation of a single CLAHE model."""
    
    print("\n🔬 Testing Single Model Evaluation")
    print("=" * 60)
    
    try:
        evaluator = ModelEvaluator("config/base_config.yaml", "results")
        
        # Test model H (CLAHE + partial masks + augmentation)
        print("Testing model_h evaluation...")
        
        # This would run the actual evaluation - commented out to avoid long runtime
        # results = evaluator.evaluate_model('model_h')
        # print(f"✅ Model H evaluation completed")
        # print(f"   - Macro F1: {results['test_metrics']['macro_f1']:.3f}")
        
        print("✅ Evaluation setup successful (actual evaluation commented out)")
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")

if __name__ == "__main__":
    test_clahe_integration()
    test_single_model_evaluation()
    
    print("\n" + "=" * 60)
    print("🎉 CLAHE Evaluation Test Complete!")
    print("=" * 60)