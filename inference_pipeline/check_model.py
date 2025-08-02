#!/usr/bin/env python3
"""
Model Checkpoint Checker
Date: 2025-08-01

Check if the Model B checkpoint exists and is loadable.
"""

import sys
from pathlib import Path
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_checkpoint(experiments_path: str = "../experiments/2025-07-05_hybrid_training"):
    """Check if the model checkpoint exists and is loadable."""
    checkpoint_path = Path(experiments_path) / "results" / "model_b" / "checkpoints" / "best_model.pth"
    
    print("🔍 Model Checkpoint Checker")
    print("=" * 50)
    print(f"Checking: {checkpoint_path}")
    
    # Check if file exists
    if not checkpoint_path.exists():
        print("❌ CHECKPOINT NOT FOUND")
        print(f"Expected location: {checkpoint_path.absolute()}")
        print("\n🔧 Possible solutions:")
        print("1. Check if the experiments path is correct")
        print("2. Ensure the model training completed successfully")
        print("3. Look for alternative checkpoint files:")
        
        # Check for alternative checkpoint files
        checkpoints_dir = checkpoint_path.parent
        if checkpoints_dir.exists():
            checkpoint_files = list(checkpoints_dir.glob("*.pth"))
            if checkpoint_files:
                print("   Available checkpoints:")
                for cp in sorted(checkpoint_files):
                    print(f"   - {cp.name}")
            else:
                print("   No .pth files found in checkpoints directory")
        else:
            print("   Checkpoints directory does not exist")
        
        return False
    
    print("✅ CHECKPOINT FILE EXISTS")
    print(f"File size: {checkpoint_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Try to load the checkpoint
    print("\n🔄 Testing checkpoint loading...")
    
    try:
        # Try with weights_only=True first (secure)
        print("Attempting secure load (weights_only=True)...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        print("✅ Loaded successfully with weights_only=True")
        load_method = "secure"
    except Exception as e:
        print(f"⚠️  Secure load failed: {e}")
        try:
            # Fall back to weights_only=False
            print("Attempting fallback load (weights_only=False)...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            print("✅ Loaded successfully with weights_only=False")
            load_method = "fallback"
        except Exception as e2:
            print(f"❌ Both loading methods failed: {e2}")
            return False
    
    # Analyze checkpoint contents
    print(f"\n📊 Checkpoint Analysis (loaded via {load_method}):")
    
    if isinstance(checkpoint, dict):
        print("Checkpoint keys:")
        for key in checkpoint.keys():
            if hasattr(checkpoint[key], 'shape'):
                print(f"  - {key}: {checkpoint[key].shape}")
            elif isinstance(checkpoint[key], (int, float, str)):
                print(f"  - {key}: {checkpoint[key]}")
            else:
                print(f"  - {key}: {type(checkpoint[key])}")
    else:
        print(f"Checkpoint type: {type(checkpoint)}")
    
    # Check for expected keys
    expected_keys = ['model_state_dict', 'epoch', 'best_accuracy', 'best_f1']
    if isinstance(checkpoint, dict):
        missing_keys = [key for key in expected_keys if key not in checkpoint]
        if missing_keys:
            print(f"\n⚠️  Missing expected keys: {missing_keys}")
        else:
            print("\n✅ All expected keys present")
    
    print("\n🎉 CHECKPOINT IS LOADABLE")
    return True

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check Model B checkpoint")
    parser.add_argument('--experiments-path', type=str, 
                       default='../experiments/2025-07-05_hybrid_training',
                       help='Path to experiments directory')
    
    args = parser.parse_args()
    
    success = check_model_checkpoint(args.experiments_path)
    
    if success:
        print("\n✅ Model checkpoint is ready for inference!")
        print("You can now run the UI: python launch_ui.py")
    else:
        print("\n❌ Model checkpoint issues detected.")
        print("Please resolve the issues above before running inference.")
        sys.exit(1)

if __name__ == "__main__":
    main()