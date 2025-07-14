#!/usr/bin/env python3
"""
Run Training for Models E, F, G, H
Date: 2025-07-05

This script demonstrates how to train the new CLAHE-enhanced model variants
using the existing trainer architecture.

Models:
- Model E: CLAHE enhanced images + full masks (no augmentation)
- Model F: CLAHE enhanced images + partial masks (no augmentation)  
- Model G: CLAHE enhanced images + full masks + augmentation
- Model H: CLAHE enhanced images + partial masks + augmentation
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from training.trainer import HybridTrainer

def run_model_training(model_variant: str, config_path: str = None):
    """
    Run training for a specific model variant.
    
    Args:
        model_variant: Model variant to train ('model_e', 'model_f', 'model_g', 'model_h')
        config_path: Optional path to configuration file
    """
    print(f"🚀 Starting training for {model_variant.upper()}")
    
    # Set default config path if not provided
    if config_path is None:
        config_path = f"config/{model_variant}_config.yaml"
    
    # Verify config file exists
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"❌ Configuration file not found: {config_file}")
        print("Available configs:")
        config_dir = Path("config")
        if config_dir.exists():
            for cfg in config_dir.glob("model_*.yaml"):
                print(f"  - {cfg}")
        return False
    
    try:
        # Initialize trainer
        trainer = HybridTrainer(
            config_path=str(config_file),
            variant=model_variant
        )
        
        # Start training
        trainer.train()
        
        print(f"✅ Training completed successfully for {model_variant.upper()}")
        return True
        
    except Exception as e:
        print(f"❌ Training failed for {model_variant.upper()}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train CLAHE-enhanced model variants')
    parser.add_argument(
        '--model', 
        choices=['model_e', 'model_f', 'model_g', 'model_h', 'all'],
        required=True,
        help='Model variant to train'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (optional)'
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run models sequentially when training all (default: False)'
    )
    
    args = parser.parse_args()
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("🎯 CLAHE-Enhanced Model Training")
    print("=" * 50)
    
    if args.model == 'all':
        # Train all CLAHE models
        models = ['model_e', 'model_f', 'model_g', 'model_h']
        success_count = 0
        
        for model in models:
            print(f"\n📦 Training {model.upper()}")
            print("-" * 30)
            
            success = run_model_training(model, args.config)
            if success:
                success_count += 1
            
            if not args.sequential and not success:
                print(f"⚠️  Stopping due to failure in {model}")
                break
        
        print(f"\n🏁 Training Summary:")
        print(f"   Successful: {success_count}/{len(models)}")
        
        if success_count == len(models):
            print("🎉 All models trained successfully!")
        else:
            print("⚠️  Some models failed to train")
            
    else:
        # Train single model
        success = run_model_training(args.model, args.config)
        if success:
            print("🎉 Training completed successfully!")
        else:
            print("❌ Training failed")

if __name__ == "__main__":
    main() 