#!/usr/bin/env python3
"""
Cross-platform setup script for 2025-07-05 Hybrid Training Experiment
Automatically detects platform and sets up the environment accordingly
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CrossPlatformSetup:
    """Cross-platform setup manager for the hybrid training experiment."""
    
    def __init__(self):
        self.platform_info = self._detect_platform()
        self.project_root = Path(__file__).parent.parent
        self.python_exe = sys.executable
        
    def _detect_platform(self):
        """Detect current platform."""
        system = platform.system().lower()
        
        platform_map = {
            'windows': 'windows',
            'darwin': 'mac', 
            'linux': 'linux'
        }
        
        detected_platform = platform_map.get(system, 'linux')
        
        info = {
            'os': detected_platform,
            'system': system,
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'is_windows': system == 'windows',
            'is_mac': system == 'darwin',
            'is_linux': system == 'linux'
        }
        
        return info
    
    def check_python_version(self):
        """Check if Python version is compatible."""
        major, minor = sys.version_info[:2]
        
        if major < 3 or (major == 3 and minor < 8):
            logger.error(f"Python 3.8+ required, found {major}.{minor}")
            return False
        
        logger.info(f"Python version: {major}.{minor} ✓")
        return True
    
    def check_gpu_support(self):
        """Check for GPU support on current platform."""
        try:
            import torch
            
            if self.platform_info['is_mac']:
                # Check for Apple Silicon MPS
                if torch.backends.mps.is_available():
                    logger.info("Apple Silicon MPS support detected ✓")
                    return 'mps'
                else:
                    logger.info("MPS not available, using CPU")
                    return 'cpu'
            else:
                # Check for CUDA
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    logger.info(f"CUDA GPU detected: {device_name} ✓")
                    return 'cuda'
                else:
                    logger.info("CUDA not available, using CPU")
                    return 'cpu'
        except ImportError:
            logger.info("PyTorch not installed yet, will check after installation")
            return 'unknown'
    
    def install_requirements(self):
        """Install requirements using pip."""
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            logger.error(f"Requirements file not found: {requirements_file}")
            return False
        
        logger.info("Installing Python packages...")
        
        try:
            # Install base requirements
            cmd = [self.python_exe, "-m", "pip", "install", "-r", str(requirements_file)]
            
            if self.platform_info['is_windows']:
                # Windows might need additional flags
                cmd.extend(["--user"])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to install requirements: {result.stderr}")
                return False
            
            logger.info("Requirements installed successfully ✓")
            return True
            
        except Exception as e:
            logger.error(f"Error installing requirements: {e}")
            return False
    
    def create_directories(self):
        """Create necessary directories."""
        directories = [
            "data/splits",
            "data/masks", 
            "data/augmented",
            "results/model_a",
            "results/model_b",
            "results/model_c", 
            "results/model_d",
            "logs",
            "checkpoints"
        ]
        
        logger.info("Creating directories...")
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                
                # Set permissions on Unix-like systems
                if not self.platform_info['is_windows']:
                    full_path.chmod(0o755)
                    
            except Exception as e:
                logger.error(f"Failed to create directory {full_path}: {e}")
                return False
        
        logger.info("Directories created successfully ✓")
        return True
    
    def setup_environment_variables(self):
        """Setup platform-specific environment variables."""
        env_vars = {
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1'
        }
        
        logger.info("Setting up environment variables...")
        
        for var, value in env_vars.items():
            os.environ[var] = value
            logger.info(f"Set {var}={value}")
        
        # Platform-specific optimizations
        if self.platform_info['is_mac']:
            # Apple Silicon optimizations
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            logger.info("Set Apple Silicon MPS fallback")
        
        logger.info("Environment variables configured ✓")
        return True
    
    def check_data_availability(self):
        """Check if coryell data is available."""
        coryell_path = self.project_root / "../../data/coryell"
        
        if not coryell_path.exists():
            logger.warning(f"Coryell data not found at {coryell_path}")
            logger.warning("Please ensure the data directory exists before running experiments")
            return False
        
        # Count road directories
        road_dirs = [d for d in coryell_path.iterdir() if d.is_dir() and d.name.startswith('Co Rd')]
        logger.info(f"Found {len(road_dirs)} road directories in coryell data ✓")
        return True
    
    def create_sample_config(self):
        """Create a sample configuration file if it doesn't exist."""
        config_file = self.project_root / "config" / "user_config.yaml"
        
        if config_file.exists():
            logger.info("User config already exists")
            return True
        
        # Create user-specific config based on platform
        sample_config = f"""# User-specific configuration for {self.platform_info['os']}
# Copy from base_config.yaml and modify as needed

# Override platform detection if needed
platform:
  os: "{self.platform_info['os']}"

# Override paths if needed
dataset:
  coryell_path: "../../data/coryell"  # Adjust if your data is elsewhere

# Override hardware settings if needed
hardware:
  device: "auto"  # Will auto-detect best device
  
# Override number of workers based on your system
dataset:
  num_workers: null  # Will auto-detect optimal number
"""
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(sample_config)
            logger.info(f"Created sample config: {config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to create sample config: {e}")
            return False
    
    def verify_installation(self):
        """Verify that everything is installed correctly."""
        logger.info("Verifying installation...")
        
        try:
            # Test imports
            import torch
            import torchvision
            import segmentation_models_pytorch as smp
            import cv2
            import yaml
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            
            logger.info("All required packages imported successfully ✓")
            
            # Test GPU support
            device = self.check_gpu_support()
            logger.info(f"Using device: {device}")
            
            # Test basic operations
            x = torch.randn(1, 3, 256, 256)
            if device == 'cuda':
                x = x.cuda()
            elif device == 'mps':
                x = x.to('mps')
            
            logger.info("Basic tensor operations working ✓")
            
            return True
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
    
    def run_setup(self):
        """Run the complete setup process."""
        logger.info("="*60)
        logger.info("2025-07-05 Hybrid Training Experiment Setup")
        logger.info("="*60)
        logger.info(f"Platform: {self.platform_info['os']} ({self.platform_info['system']})")
        logger.info(f"Architecture: {self.platform_info['architecture']}")
        logger.info(f"Python: {self.platform_info['python_version']}")
        logger.info("="*60)
        
        steps = [
            ("Check Python version", self.check_python_version),
            ("Create directories", self.create_directories),
            ("Install requirements", self.install_requirements),
            ("Setup environment", self.setup_environment_variables),
            ("Create sample config", self.create_sample_config),
            ("Check data availability", self.check_data_availability),
            ("Verify installation", self.verify_installation)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n--- {step_name} ---")
            if not step_func():
                logger.error(f"Setup failed at step: {step_name}")
                return False
        
        logger.info("\n" + "="*60)
        logger.info("Setup completed successfully! 🎉")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Review config/base_config.yaml")
        logger.info("2. Adjust config/user_config.yaml if needed")
        logger.info("3. Run: python src/data/smart_splitter.py")
        logger.info("4. Train models using the training scripts")
        logger.info("="*60)
        
        return True

def main():
    """Main setup function."""
    setup = CrossPlatformSetup()
    success = setup.run_setup()
    
    if not success:
        logger.error("Setup failed. Please check the errors above.")
        sys.exit(1)
    
    logger.info("Setup successful!")

if __name__ == "__main__":
    main() 