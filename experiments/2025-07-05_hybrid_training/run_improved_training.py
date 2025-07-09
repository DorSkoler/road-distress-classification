#!/usr/bin/env python3
"""
Run improved training with fixes for class imbalance and overfitting.

This script uses the improved configuration that addresses:
1. Class imbalance with proper class weights
2. Overfitting with stronger regularization  
3. More honest evaluation with macro F1
4. Reduced augmentation to prevent overfitting
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from training.trainer import HybridTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('improved_training.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run improved training for all model variants."""
    
    print("🚀 Starting IMPROVED training with class balance and regularization fixes")
    print("=" * 70)
    
    # Configuration with improvements
    config_path = "config/improved_config.yaml"
    
    # Model variants to train
    variants = ['model_a', 'model_b', 'model_c', 'model_d']
    
    results = {}
    total_start_time = time.time()
    
    for i, variant in enumerate(variants, 1):
        print(f"\n[{i}/{len(variants)}] Training {variant}")
        print("-" * 50)
        
        try:
            # Create trainer with improved config
            trainer = HybridTrainer(config_path, variant)
            
            # Run training
            variant_results = trainer.train()
            results[variant] = variant_results
            
            print(f"✅ {variant} completed:")
            print(f"   Best macro F1: {variant_results['best_metric']:.4f}")
            print(f"   Best epoch: {variant_results['best_epoch']+1}")
            print(f"   Time: {variant_results['total_time']/3600:.2f} hours")
            
        except KeyboardInterrupt:
            print(f"\n🛑 Training interrupted for {variant}")
            print("Partial results saved. You can resume later.")
            break
            
        except Exception as e:
            logger.error(f"Error training {variant}: {e}")
            print(f"❌ {variant} failed: {e}")
            continue
    
    # Final summary
    total_time = time.time() - total_start_time
    print("\n" + "=" * 70)
    print("🏁 IMPROVED TRAINING SUMMARY")
    print("=" * 70)
    
    if results:
        print("\nResults with improved configuration:")
        print("(Class weights, stronger regularization, macro F1 evaluation)")
        print()
        
        for variant, result in results.items():
            print(f"{variant:8}: Best macro F1 = {result['best_metric']:.4f} "
                  f"(epoch {result['best_epoch']+1}, {result['total_time']/3600:.1f}h)")
        
        # Find best model
        best_variant = max(results.items(), key=lambda x: x[1]['best_metric'])
        print(f"\n🏆 Best model: {best_variant[0]} with macro F1 = {best_variant[1]['best_metric']:.4f}")
        
    else:
        print("❌ No models completed training")
    
    print(f"\nTotal training time: {total_time/3600:.2f} hours")
    print("\n📊 Key improvements implemented:")
    print("  ✓ Class weights: [1.0, 2.2, 9.3] to address severe imbalance")
    print("  ✓ Stronger regularization: dropout 0.7, weight_decay 0.05")
    print("  ✓ Macro F1 evaluation: More honest with imbalanced classes")
    print("  ✓ Reduced augmentation: Prevent overfitting")
    print("  ✓ Early stopping: 5 epochs instead of 10")
    print("  ✓ Label smoothing: 0.1 for additional regularization")
    
    print("\n🔍 Expected improvements:")
    print("  • Lower overall F1 scores (more honest evaluation)")
    print("  • Better performance on minority classes (crop, occlusion)")
    print("  • Reduced train/val gap (less overfitting)")
    print("  • More stable training curves")

if __name__ == "__main__":
    main() 