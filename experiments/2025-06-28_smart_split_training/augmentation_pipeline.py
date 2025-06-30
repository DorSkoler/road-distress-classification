#!/usr/bin/env python3
"""
Augmentation Pipeline for Road Distress Classification
Date: 2025-06-28

This script creates diverse augmented versions of images:
1. Loads images and masks from smart splits
2. Applies comprehensive augmentation strategies
3. Maintains label consistency across augmentations
4. Generates 3-5 versions per image
5. Saves augmented images and corresponding masks
"""

import os
import json
import yaml
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from collections import defaultdict
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AugmentationPipeline:
    """Comprehensive augmentation pipeline for road distress classification."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the augmentation pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.setup_paths()
        self.setup_random_seed()
        self.create_augmentation_transforms()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def setup_paths(self):
        """Setup all necessary paths."""
        # Data paths
        self.coryell_root = Path(self.config['dataset']['coryell_path'])
        self.splits_dir = Path("splits")
        self.masks_dir = Path("masks")
        
        # Output paths
        self.augmented_dir = Path("augmented")
        self.augmented_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each split
        for split in ['train', 'val', 'test']:
            (self.augmented_dir / split).mkdir(exist_ok=True)
            (self.augmented_dir / split / "images").mkdir(exist_ok=True)
            (self.augmented_dir / split / "masks").mkdir(exist_ok=True)
            
        logger.info("Setup paths completed")
    
    def setup_random_seed(self):
        """Setup random seed for reproducibility."""
        random.seed(self.config['system']['random_seed'])
        np.random.seed(self.config['system']['random_seed'])
        logger.info(f"Set random seed to {self.config['system']['random_seed']}")
    
    def create_augmentation_transforms(self):
        """Create conservative, road-specific augmentation transforms."""
        logger.info("Creating conservative augmentation transforms...")
        
        aug_config = self.config['augmentation']
        samples_per_image = aug_config['samples_per_image']
        
        # Create conservative augmentation strategies
        self.augmentation_strategies = []
        
        # Strategy 1: Light geometric (minimal rotation, flip)
        geometric_transform = A.Compose([
            A.Rotate(limit=5, p=0.3),  # Very small rotation
            A.HorizontalFlip(p=0.5),   # Flip is safe for roads
        ])
        self.augmentation_strategies.append(("geometric", geometric_transform))
        
        # Strategy 2: Light color adjustments (realistic lighting changes)
        color_transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.1,  # Very small brightness change
                contrast_limit=0.1,    # Very small contrast change
                p=0.5
            ),
        ])
        self.augmentation_strategies.append(("color", color_transform))
        
        # Strategy 3: Minimal noise (realistic camera noise)
        noise_transform = A.Compose([
            A.GaussNoise(
                var_limit=(5.0, 15.0),  # Very light noise
                p=0.3
            ),
        ])
        self.augmentation_strategies.append(("noise", noise_transform))
        
        # Strategy 4: Combined light augmentations
        combined_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.4),
            A.GaussNoise(var_limit=(3.0, 10.0), p=0.2),
        ])
        self.augmentation_strategies.append(("combined", combined_transform))
        
        # Strategy 5: No augmentation (original image)
        no_aug_transform = A.Compose([])
        self.augmentation_strategies.append(("original", no_aug_transform))
        
        logger.info(f"Created {len(self.augmentation_strategies)} conservative augmentation strategies")
    
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
        
        with open(split_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]
        
        logger.info(f"Loaded {len(image_ids)} images for {split_name} split")
        return image_ids
    
    def load_image_and_mask(self, image_id: str, split_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load image and corresponding mask.
        
        Args:
            image_id: Image ID in format "Co Rd 4235/000_31.708136_-97.693460"
            split_name: Split name (train, val, test)
            
        Returns:
            Tuple of (image, mask) as numpy arrays
        """
        try:
            # Parse road name and image name
            road_name, image_name = image_id.split('/', 1)
            
            # Load image
            image_path = self.coryell_root / road_name / "img" / f"{image_name}.png"
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                return None, None
            
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                return None, None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load mask
            mask_path = self.masks_dir / split_name / road_name / f"{image_name}.png"
            if not mask_path.exists():
                logger.warning(f"Mask not found: {mask_path}")
                return image_rgb, None
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Failed to load mask: {mask_path}")
                return image_rgb, None
            
            return image_rgb, mask
            
        except Exception as e:
            logger.warning(f"Error loading image/mask {image_id}: {e}")
            return None, None
    
    def apply_augmentation(self, image: np.ndarray, mask: Optional[np.ndarray], 
                          strategy_name: str, transform) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply augmentation to image and mask.
        
        Args:
            image: Input image
            mask: Input mask (optional)
            strategy_name: Name of augmentation strategy
            transform: Albumentations transform
            
        Returns:
            Tuple of (augmented_image, augmented_mask)
        """
        try:
            if mask is not None:
                # Apply transform to both image and mask
                augmented = transform(image=image, mask=mask)
                return augmented['image'], augmented['mask']
            else:
                # Apply transform to image only
                augmented = transform(image=image)
                return augmented['image'], None
                
        except Exception as e:
            logger.warning(f"Error applying {strategy_name} augmentation: {e}")
            return image, mask
    
    def save_augmented_data(self, image: np.ndarray, mask: Optional[np.ndarray], 
                           image_id: str, split_name: str, aug_index: int, strategy_name: str):
        """Save augmented image and mask.
        
        Args:
            image: Augmented image
            mask: Augmented mask (optional)
            image_id: Original image ID
            split_name: Split name
            aug_index: Augmentation index
            strategy_name: Strategy name
        """
        try:
            # Parse road name and image name
            road_name, image_name = image_id.split('/', 1)
            
            # Create road directories
            image_road_dir = self.augmented_dir / split_name / "images" / road_name
            mask_road_dir = self.augmented_dir / split_name / "masks" / road_name
            image_road_dir.mkdir(exist_ok=True)
            mask_road_dir.mkdir(exist_ok=True)
            
            # Create augmented filename
            aug_filename = f"{image_name}_aug{aug_index:02d}_{strategy_name}.png"
            
            # Save augmented image
            image_path = image_road_dir / aug_filename
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_path), image_bgr)
            
            # Save augmented mask if available
            if mask is not None:
                mask_path = mask_road_dir / aug_filename
                cv2.imwrite(str(mask_path), mask)
            
        except Exception as e:
            logger.warning(f"Error saving augmented data for {image_id}: {e}")
    
    def process_split(self, split_name: str) -> Dict:
        """Process all images in a split to create augmented versions.
        
        Args:
            split_name: Name of the split (train, val, test)
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing {split_name} split for augmentation...")
        
        # Load image IDs
        image_ids = self.load_split_data(split_name)
        
        if not image_ids:
            logger.warning(f"No images found for {split_name} split")
            return {}
        
        # Processing statistics
        stats = {
            'total_images': len(image_ids),
            'processed_images': 0,
            'skipped_no_mask': 0,
            'skipped_no_image': 0,
            'failed_images': 0,
            'total_augmented': 0,
            'augmentation_stats': defaultdict(int)
        }
        
        samples_per_image = self.config['augmentation']['samples_per_image']
        
        # Process each image
        for image_id in tqdm(image_ids, desc=f"Augmenting {split_name}"):
            try:
                # Load image and mask
                image, mask = self.load_image_and_mask(image_id, split_name)
                
                # Skip if no image
                if image is None:
                    stats['skipped_no_image'] += 1
                    continue
                
                # Skip if no mask (don't augment images without masks)
                if mask is None:
                    stats['skipped_no_mask'] += 1
                    continue
                
                # Apply different augmentation strategies
                for aug_index, (strategy_name, transform) in enumerate(self.augmentation_strategies):
                    if aug_index >= samples_per_image:
                        break
                    
                    # Apply augmentation
                    aug_image, aug_mask = self.apply_augmentation(image, mask, strategy_name, transform)
                    
                    # Save augmented data
                    self.save_augmented_data(aug_image, aug_mask, image_id, split_name, aug_index, strategy_name)
                    
                    # Update statistics
                    stats['total_augmented'] += 1
                    stats['augmentation_stats'][strategy_name] += 1
                
                stats['processed_images'] += 1
                
            except Exception as e:
                logger.warning(f"Error processing {image_id}: {e}")
                stats['failed_images'] += 1
        
        logger.info(f"{split_name} split augmentation completed:")
        logger.info(f"  - Total images in split: {stats['total_images']}")
        logger.info(f"  - Processed (with mask): {stats['processed_images']}")
        logger.info(f"  - Skipped (no mask): {stats['skipped_no_mask']}")
        logger.info(f"  - Skipped (no image): {stats['skipped_no_image']}")
        logger.info(f"  - Failed: {stats['failed_images']}")
        logger.info(f"  - Total Augmented: {stats['total_augmented']}")
        logger.info(f"  - Augmentation Distribution: {dict(stats['augmentation_stats'])}")
        
        return stats
    
    def create_visualization(self, all_stats: Dict):
        """Create visualization of augmentation results.
        
        Args:
            all_stats: Statistics from all splits
        """
        logger.info("Creating augmentation visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Split processing statistics
        splits = list(all_stats.keys())
        processed_counts = [all_stats[split]['processed_images'] for split in splits]
        failed_counts = [all_stats[split]['failed_images'] for split in splits]
        augmented_counts = [all_stats[split]['total_augmented'] for split in splits]
        
        x = np.arange(len(splits))
        width = 0.25
        
        axes[0, 0].bar(x - width, processed_counts, width, label='Processed', color='green')
        axes[0, 0].bar(x, failed_counts, width, label='Failed', color='red')
        axes[0, 0].bar(x + width, augmented_counts, width, label='Augmented', color='blue')
        
        axes[0, 0].set_title('Augmentation Results by Split')
        axes[0, 0].set_ylabel('Number of Images')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(splits)
        axes[0, 0].legend()
        
        # Augmentation strategy distribution (combined)
        all_aug_stats = defaultdict(int)
        for split_stats in all_stats.values():
            for strategy, count in split_stats['augmentation_stats'].items():
                all_aug_stats[strategy] += count
        
        if all_aug_stats:
            strategies = list(all_aug_stats.keys())
            counts = list(all_aug_stats.values())
            
            axes[0, 1].bar(strategies, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
            axes[0, 1].set_title('Augmentation Strategy Distribution')
            axes[0, 1].set_ylabel('Number of Augmented Images')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Augmentation ratio by split
        augmentation_ratios = []
        for split in splits:
            total = all_stats[split]['total_images']
            augmented = all_stats[split]['total_augmented']
            ratio = (augmented / total) if total > 0 else 0
            augmentation_ratios.append(ratio)
        
        axes[1, 0].bar(splits, augmentation_ratios, color=['#2E8B57', '#FF6B6B', '#4ECDC4'])
        axes[1, 0].set_title('Augmentation Ratio by Split')
        axes[1, 0].set_ylabel('Augmented Images per Original Image')
        
        # Success rate by split
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
        plt.savefig(self.augmented_dir / 'augmentation_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {self.augmented_dir / 'augmentation_visualization.png'}")
    
    def save_statistics(self, all_stats: Dict):
        """Save processing statistics to file.
        
        Args:
            all_stats: Statistics from all splits
        """
        logger.info("Saving augmentation statistics...")
        
        # Save detailed statistics
        stats_file = self.augmented_dir / "augmentation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        # Create summary
        summary = {
            'total_images': sum(stats['total_images'] for stats in all_stats.values()),
            'total_processed': sum(stats['processed_images'] for stats in all_stats.values()),
            'total_failed': sum(stats['failed_images'] for stats in all_stats.values()),
            'total_augmented': sum(stats['total_augmented'] for stats in all_stats.values()),
            'overall_success_rate': 0,
            'augmentation_ratio': 0,
            'splits': {}
        }
        
        for split_name, stats in all_stats.items():
            total = stats['total_images']
            processed = stats['processed_images']
            augmented = stats['total_augmented']
            success_rate = (processed / total * 100) if total > 0 else 0
            aug_ratio = (augmented / total) if total > 0 else 0
            
            summary['splits'][split_name] = {
                'total_images': total,
                'processed_images': processed,
                'failed_images': stats['failed_images'],
                'total_augmented': augmented,
                'success_rate': success_rate,
                'augmentation_ratio': aug_ratio,
                'augmentation_distribution': dict(stats['augmentation_stats'])
            }
        
        # Calculate overall statistics
        total_images = summary['total_images']
        total_processed = summary['total_processed']
        total_augmented = summary['total_augmented']
        
        summary['overall_success_rate'] = (total_processed / total_images * 100) if total_images > 0 else 0
        summary['augmentation_ratio'] = (total_augmented / total_images) if total_images > 0 else 0
        
        # Save summary
        summary_file = self.augmented_dir / "augmentation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved statistics to {stats_file}")
        logger.info(f"Saved summary to {summary_file}")
    
    def run(self):
        """Run the complete augmentation pipeline."""
        logger.info("Starting augmentation pipeline...")
        
        # Process all splits
        all_stats = {}
        
        for split_name in ['train', 'val', 'test']:
            stats = self.process_split(split_name)
            all_stats[split_name] = stats
        
        # Create visualizations
        self.create_visualization(all_stats)
        
        # Save statistics
        self.save_statistics(all_stats)
        
        logger.info("Augmentation pipeline completed successfully!")

def main():
    """Main function."""
    pipeline = AugmentationPipeline()
    pipeline.run()

if __name__ == "__main__":
    main() 