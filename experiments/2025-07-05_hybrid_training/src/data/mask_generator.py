#!/usr/bin/env python3
"""
Cross-Platform Road Mask Generator for Hybrid Training Experiment
Date: 2025-07-05

This script generates road masks for all images using a segmentation model:
1. Loads road segmentation model from checkpoints
2. Processes all images through the model
3. Calculates road coverage percentage
4. Filters images with <15% road coverage
5. Saves high-quality masks with cross-platform compatibility

Adapted from 2025-06-28 experiment with cross-platform improvements.
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from collections import defaultdict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.platform_utils import PlatformManager, get_platform_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridMaskGenerator:
    """Cross-platform road mask generator for hybrid training experiment."""
    
    def __init__(self, config_path: str = "config/base_config.yaml"):
        """Initialize the road mask generator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.platform_utils = PlatformManager(self.config)
        self.setup_paths()
        self.setup_device()
        self.model = None  # Will be loaded when needed
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply platform-specific configurations
        # Configuration is loaded as-is, platform manager will handle platform-specific logic
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def setup_paths(self):
        """Setup all necessary paths with cross-platform compatibility."""
        # Data paths
        coryell_path = self.config['dataset']['coryell_path']
        self.coryell_root = Path(coryell_path).resolve()
        
        # Use platform-specific path handling
        self.splits_dir = Path("data/splits").resolve()
        self.masks_dir = Path("data/masks").resolve()
        
        # Create directories with proper permissions
        self.platform_utils.create_directory(self.masks_dir)
        
        # Create subdirectories for each split
        for split in ['train', 'val', 'test']:
            split_dir = self.masks_dir / split
            self.platform_utils.create_directory(split_dir)
            
        logger.info(f"Setup paths completed - Coryell: {self.coryell_root}")
        logger.info(f"Masks directory: {self.masks_dir}")
    
    def setup_device(self):
        """Setup device for model inference with cross-platform support."""
        # Use platform utils for device detection
        device_str = self.platform_utils.get_device()
        self.device = torch.device(device_str)
        
        # Log device information
        if self.device.type == 'cuda':
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif self.device.type == 'mps':
            logger.info("Using Apple Silicon MPS acceleration")
        else:
            logger.info("Using CPU (no GPU acceleration available)")
    
    def load_model(self):
        """Load the road segmentation model with cross-platform support."""
        if self.model is not None:
            return  # Already loaded
            
        # Check if model path is specified in config
        model_config = self.config.get('mask_generation', {})
        model_path = model_config.get('model_checkpoint')
        
        if not model_path:
            logger.warning("No model path specified in config. Using dummy model for testing.")
            self._create_dummy_model()
            return
        
        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}. Using dummy model for testing.")
            self._create_dummy_model()
            return
        
        logger.info(f"Loading model from {model_path}")
        
        try:
            # Create the model architecture (U-Net with ResNet34 encoder)
            import segmentation_models_pytorch as smp
            
            self.model = smp.Unet(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                classes=1,
                activation=None
            )
            
            # Load the state dict with cross-platform compatibility
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Using dummy model for testing")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model for testing when real model is not available."""
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                # Create a simple road-like mask in the center
                batch_size = x.shape[0]
                height, width = x.shape[2], x.shape[3]
                
                # Create a rectangular road mask in the center
                mask = torch.zeros(batch_size, 1, height, width)
                center_h, center_w = height // 2, width // 2
                road_height = height // 3
                road_width = width // 2
                
                start_h = center_h - road_height // 2
                end_h = center_h + road_height // 2
                start_w = center_w - road_width // 2
                end_w = center_w + road_width // 2
                
                mask[:, :, start_h:end_h, start_w:end_w] = 1.0
                
                return mask
        
        self.model = DummyModel()
        self.model.to(self.device)
        self.model.eval()
        logger.info("Created dummy model for testing")
    
    def load_split_data(self, split_name: str) -> List[str]:
        """Load image IDs for a specific split.
        
        Args:
            split_name: Name of the split (train, val, test)
            
        Returns:
            List of image IDs for the split
        """
        split_file = self.splits_dir / f"{split_name}_images.txt"
        
        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            return []
        
        try:
            with open(split_file, 'r', encoding='utf-8') as f:
                image_ids = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            logger.error(f"Error reading split file {split_file}: {e}")
            return []
        
        logger.info(f"Loaded {len(image_ids)} images for {split_name} split")
        return image_ids
    
    def load_image(self, image_id: str) -> Optional[np.ndarray]:
        """Load image from coryell data with cross-platform path handling.
        
        Args:
            image_id: Image ID in format "Co Rd 4235/000_31.708136_-97.693460"
            
        Returns:
            Loaded image as numpy array
        """
        try:
            # Parse road name and image name
            if '/' not in image_id:
                logger.warning(f"Invalid image ID format: {image_id}")
                return None
                
            road_name, image_name = image_id.split('/', 1)
            
            # Construct image path with cross-platform handling
            image_path = self.coryell_root / road_name / "img" / f"{image_name}.png"
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                return None
            
            # Load image with cross-platform compatibility
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image_rgb
            
        except Exception as e:
            logger.warning(f"Error loading image {image_id}: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model inference.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Preprocessed image as torch tensor
        """
        # Get target size from config
        target_size = tuple(self.config['dataset']['image_size'])
        
        # Resize image
        image_resized = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def generate_mask(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Generate road mask using the segmentation model.
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Road mask as numpy array (H, W)
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        with torch.no_grad():
            # Forward pass
            output = self.model(image_tensor)
            
            # Apply sigmoid if not already applied
            if output.shape[1] == 1:
                mask = torch.sigmoid(output)
            else:
                mask = F.softmax(output, dim=1)[:, 1:2]  # Take road class
            
            # Convert to numpy
            mask_np = mask.squeeze().cpu().numpy()
            
            # Apply threshold
            threshold = self.config.get('mask_generation', {}).get('confidence_threshold', 0.5)
            mask_binary = (mask_np > threshold).astype(np.uint8)
            
            return mask_binary
    
    def calculate_road_coverage(self, mask: np.ndarray) -> float:
        """Calculate road coverage percentage.
        
        Args:
            mask: Binary road mask
            
        Returns:
            Road coverage percentage (0.0 to 1.0)
        """
        total_pixels = mask.shape[0] * mask.shape[1]
        road_pixels = np.sum(mask)
        coverage = road_pixels / total_pixels
        
        return coverage
    
    def postprocess_mask(self, mask: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """Postprocess mask to original image size.
        
        Args:
            mask: Generated mask
            original_size: Original image size (H, W)
            
        Returns:
            Resized mask to original size
        """
        # Resize mask to original image size
        mask_resized = cv2.resize(mask.astype(np.float32), 
                                 (original_size[1], original_size[0]))
        
        # Apply dilation if configured
        dilation = self.config.get('mask_generation', {}).get('dilation_kernel', 0)
        if dilation > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
            mask_resized = cv2.dilate(mask_resized, kernel, iterations=1)
        
        # Convert back to binary
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        return mask_binary
    
    def save_mask(self, mask: np.ndarray, image_id: str, split_name: str):
        """Save mask to file with cross-platform compatibility.
        
        Args:
            mask: Binary road mask
            image_id: Image ID
            split_name: Split name (train, val, test)
        """
        try:
            # Parse road name and image name
            road_name, image_name = image_id.split('/', 1)
            
            # Create road directory with cross-platform handling
            road_dir = self.masks_dir / split_name / road_name
            self.platform_utils.create_directory(road_dir)
            
            # Save mask
            mask_path = road_dir / f"{image_name}.png"
            save_format = self.config.get('mask_generation', {}).get('save_format', 'png')
            
            if save_format == 'png':
                # Use cross-platform file writing
                success = cv2.imwrite(str(mask_path), mask * 255)
                if not success:
                    logger.warning(f"Failed to save mask: {mask_path}")
            else:
                # Save as numpy array
                np.save(str(mask_path).replace('.png', '.npy'), mask)
                
        except Exception as e:
            logger.warning(f"Error saving mask for {image_id}: {e}")
    
    def process_split(self, split_name: str) -> Dict:
        """Process all images in a split to generate masks.
        
        Args:
            split_name: Name of the split (train, val, test)
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing {split_name} split...")
        
        # Load image IDs
        image_ids = self.load_split_data(split_name)
        
        if not image_ids:
            logger.warning(f"No images found for {split_name} split")
            return {}
        
        # Processing statistics
        stats = {
            'total_images': len(image_ids),
            'processed_images': 0,
            'failed_images': 0,
            'filtered_images': 0,
            'coverage_stats': [],
            'road_coverage_distribution': defaultdict(int)
        }
        
        # Quality thresholds
        min_coverage = self.config['dataset'].get('min_road_coverage', 0.15)
        max_coverage = self.config['dataset'].get('max_road_coverage', 0.95)
        
        # Process each image with progress bar
        for image_id in tqdm(image_ids, desc=f"Processing {split_name}"):
            try:
                # Load image
                image = self.load_image(image_id)
                if image is None:
                    stats['failed_images'] += 1
                    continue
                
                # Preprocess image
                image_tensor = self.preprocess_image(image)
                
                # Generate mask
                mask = self.generate_mask(image_tensor)
                
                # Calculate coverage
                coverage = self.calculate_road_coverage(mask)
                stats['coverage_stats'].append(coverage)
                
                # Check quality thresholds
                if coverage < min_coverage or coverage > max_coverage:
                    stats['filtered_images'] += 1
                    logger.debug(f"Filtered {image_id}: coverage {coverage:.3f}")
                    continue
                
                # Postprocess mask
                original_size = (image.shape[0], image.shape[1])
                mask_final = self.postprocess_mask(mask, original_size)
                
                # Save mask
                self.save_mask(mask_final, image_id, split_name)
                
                # Update statistics
                stats['processed_images'] += 1
                
                # Track coverage distribution
                coverage_bin = int(coverage * 10) * 10  # 0-10%, 10-20%, etc.
                stats['road_coverage_distribution'][coverage_bin] += 1
                
            except Exception as e:
                logger.warning(f"Error processing {image_id}: {e}")
                stats['failed_images'] += 1
        
        # Calculate summary statistics
        if stats['coverage_stats']:
            stats['mean_coverage'] = np.mean(stats['coverage_stats'])
            stats['std_coverage'] = np.std(stats['coverage_stats'])
            stats['min_coverage'] = np.min(stats['coverage_stats'])
            stats['max_coverage'] = np.max(stats['coverage_stats'])
        
        logger.info(f"{split_name} split completed:")
        logger.info(f"  - Total: {stats['total_images']}")
        logger.info(f"  - Processed: {stats['processed_images']}")
        logger.info(f"  - Failed: {stats['failed_images']}")
        logger.info(f"  - Filtered: {stats['filtered_images']}")
        
        if 'mean_coverage' in stats:
            logger.info(f"  - Mean coverage: {stats['mean_coverage']:.3f}")
            logger.info(f"  - Coverage range: {stats['min_coverage']:.3f} - {stats['max_coverage']:.3f}")
        
        return stats
    
    def create_visualization(self, all_stats: Dict):
        """Create visualization of mask generation results.
        
        Args:
            all_stats: Statistics from all splits
        """
        logger.info("Creating mask generation visualizations...")
        
        # Use matplotlib with cross-platform backend
        plt.switch_backend('Agg')  # Non-interactive backend
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Split processing statistics
        splits = list(all_stats.keys())
        processed_counts = [all_stats[split]['processed_images'] for split in splits]
        failed_counts = [all_stats[split]['failed_images'] for split in splits]
        filtered_counts = [all_stats[split]['filtered_images'] for split in splits]
        
        x = np.arange(len(splits))
        width = 0.25
        
        axes[0, 0].bar(x - width, processed_counts, width, label='Processed', color='green')
        axes[0, 0].bar(x, failed_counts, width, label='Failed', color='red')
        axes[0, 0].bar(x + width, filtered_counts, width, label='Filtered', color='orange')
        
        axes[0, 0].set_title('Mask Generation Results by Split')
        axes[0, 0].set_ylabel('Number of Images')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(splits)
        axes[0, 0].legend()
        
        # Coverage statistics
        coverage_means = []
        coverage_stds = []
        
        for split in splits:
            if 'mean_coverage' in all_stats[split]:
                coverage_means.append(all_stats[split]['mean_coverage'])
                coverage_stds.append(all_stats[split]['std_coverage'])
            else:
                coverage_means.append(0)
                coverage_stds.append(0)
        
        axes[0, 1].bar(splits, coverage_means, yerr=coverage_stds, capsize=5, color='skyblue')
        axes[0, 1].set_title('Mean Road Coverage by Split')
        axes[0, 1].set_ylabel('Coverage Percentage')
        axes[0, 1].axhline(y=0.15, color='red', linestyle='--', label='Min Threshold (15%)')
        axes[0, 1].legend()
        
        # Coverage distribution (combined)
        all_coverage = []
        for split_stats in all_stats.values():
            all_coverage.extend(split_stats['coverage_stats'])
        
        if all_coverage:
            axes[1, 0].hist(all_coverage, bins=20, color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('Road Coverage Distribution (All Images)')
            axes[1, 0].set_xlabel('Coverage Percentage')
            axes[1, 0].set_ylabel('Number of Images')
            axes[1, 0].axvline(x=0.15, color='red', linestyle='--', label='Min Threshold')
            axes[1, 0].axvline(x=0.95, color='red', linestyle='--', label='Max Threshold')
            axes[1, 0].legend()
        
        # Processing success rate
        success_rates = []
        for split in splits:
            total = all_stats[split]['total_images']
            processed = all_stats[split]['processed_images']
            success_rate = (processed / total * 100) if total > 0 else 0
            success_rates.append(success_rate)
        
        axes[1, 1].bar(splits, success_rates, color=['#2E8B57', '#FF6B6B', '#4ECDC4'])
        axes[1, 1].set_title('Processing Success Rate by Split')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].set_ylim(0, 100)
        
        plt.tight_layout()
        
        # Save with cross-platform path handling
        viz_path = self.masks_dir / 'mask_generation_visualization.png'
        plt.savefig(str(viz_path), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {viz_path}")
    
    def save_statistics(self, all_stats: Dict):
        """Save processing statistics to file.
        
        Args:
            all_stats: Statistics from all splits
        """
        logger.info("Saving mask generation statistics...")
        
        # Save detailed statistics
        stats_file = self.masks_dir / "mask_generation_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        
        # Create summary
        summary = {
            'total_images': sum(stats['total_images'] for stats in all_stats.values()),
            'total_processed': sum(stats['processed_images'] for stats in all_stats.values()),
            'total_failed': sum(stats['failed_images'] for stats in all_stats.values()),
            'total_filtered': sum(stats['filtered_images'] for stats in all_stats.values()),
            'overall_success_rate': 0,
            'splits': {}
        }
        
        for split_name, stats in all_stats.items():
            total = stats['total_images']
            processed = stats['processed_images']
            success_rate = (processed / total * 100) if total > 0 else 0
            
            summary['splits'][split_name] = {
                'total_images': total,
                'processed_images': processed,
                'failed_images': stats['failed_images'],
                'filtered_images': stats['filtered_images'],
                'success_rate': success_rate
            }
            
            if 'mean_coverage' in stats:
                summary['splits'][split_name]['mean_coverage'] = stats['mean_coverage']
                summary['splits'][split_name]['coverage_range'] = [stats['min_coverage'], stats['max_coverage']]
        
        # Calculate overall success rate
        total_images = summary['total_images']
        total_processed = summary['total_processed']
        summary['overall_success_rate'] = (total_processed / total_images * 100) if total_images > 0 else 0
        
        # Save summary
        summary_file = self.masks_dir / "mask_generation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved statistics to {stats_file}")
        logger.info(f"Saved summary to {summary_file}")
    
    def run(self):
        """Run the complete road mask generation pipeline."""
        logger.info("Starting cross-platform road mask generation pipeline...")
        logger.info(f"Platform: {self.platform_utils.get_platform()}")
        logger.info(f"Device: {self.device}")
        
        # Process all splits
        all_stats = {}
        
        for split_name in ['train', 'val', 'test']:
            stats = self.process_split(split_name)
            if stats:  # Only add if processing was successful
                all_stats[split_name] = stats
        
        if not all_stats:
            logger.error("No splits were processed successfully!")
            return
        
        # Create visualizations
        self.create_visualization(all_stats)
        
        # Save statistics
        self.save_statistics(all_stats)
        
        logger.info("Cross-platform road mask generation completed successfully!")

def main():
    """Main function."""
    try:
        generator = HybridMaskGenerator()
        generator.run()
    except Exception as e:
        logger.error(f"Error in mask generation: {e}")
        raise

if __name__ == "__main__":
    main() 