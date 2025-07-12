#!/usr/bin/env python3
"""
Wrapper script to run training with suppressed warnings
"""

import os
import sys
import subprocess

# Set environment variables to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Import warnings suppression
import warnings
warnings.filterwarnings('ignore')

def check_dependencies():
    """Check and install required dependencies"""
    try:
        import requests
        print("✓ requests library is available")
    except ImportError:
        print("Installing requests library...")
        subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)
        print("✓ requests library installed")

def main():
    """Run training with the specified arguments"""
    
    # Check dependencies first
    check_dependencies()
    
    # Default training command
    cmd = [
        sys.executable, "train_model_e.py",
        "--train-images", "data/coryell",
        "--val-images", "data/coryell", 
        "--train-labels", "train_labels.csv",
        "--val-labels", "val_labels.csv",
        "--clahe-params", "clahe_params.json",
        "--output-dir", "experiments/model_e",
        "--batch-size", "32",
        "--epochs", "50", 
        "--lr", "1e-3",
        "--img-size", "256"
    ]
    
    print("Starting Model E training with suppressed warnings...")
    print("Command:", " ".join(cmd))
    print("-" * 50)
    
    # Run the training
    try:
        result = subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 