#!/usr/bin/env python3
"""
Clean Split Files
Date: 2025-06-28

This script cleans up the split files to only include images that have both:
1. Original image file
2. Corresponding mask file

This ensures the augmentation pipeline only processes valid image-mask pairs.
"""

import os
import json
from pathlib import Path
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SplitCleaner:
    """Clean split files to only include valid image-mask pairs."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the split cleaner."""
        self.config = self._load_config(config_path)
        self.setup_paths()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def setup_paths(self):
        """Setup all necessary paths."""
        self.coryell_root = Path(self.config['dataset']['coryell_path'])
        self.splits_dir = Path("splits")
        self.masks_dir = Path("masks")
        
        # Create backup directory
        self.backup_dir = Path("splits_backup")
        self.backup_dir.mkdir(exist_ok=True)
        
        logger.info("Setup paths completed")
    
    def validate_image_mask_pair(self, image_id: str, split_name: str) -> bool:
        """Validate that both image and mask exist for a given image ID.
        
        Args:
            image_id: Image ID in format "Co Rd 4235/000_31.708136_-97.693460"
            split_name: Split name (train, val, test)
            
        Returns:
            True if both image and mask exist, False otherwise
        """
        try:
            # Parse road name and image name
            road_name, image_name = image_id.split('/', 1)
            
            # Check if image exists
            image_path = self.coryell_root / road_name / "img" / f"{image_name}.png"
            if not image_path.exists():
                return False
            
            # Check if mask exists
            mask_path = self.masks_dir / split_name / road_name / f"{image_name}.png"
            if not mask_path.exists():
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating {image_id}: {e}")
            return False
    
    def clean_split_file(self, split_name: str) -> Dict:
        """Clean a single split file.
        
        Args:
            split_name: Name of the split (train, val, test)
            
        Returns:
            Dictionary with cleaning statistics
        """
        split_file = self.splits_dir / f"{split_name}_images.txt"
        
        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            return {}
        
        # Read original image IDs
        with open(split_file, 'r') as f:
            original_image_ids = [line.strip() for line in f.readlines()]
        
        logger.info(f"Processing {split_name} split: {len(original_image_ids)} images")
        
        # Validate each image-mask pair
        valid_image_ids = []
        invalid_image_ids = []
        
        for image_id in original_image_ids:
            if self.validate_image_mask_pair(image_id, split_name):
                valid_image_ids.append(image_id)
            else:
                invalid_image_ids.append(image_id)
        
        # Create backup of original file
        backup_file = self.backup_dir / f"{split_name}_images_backup.txt"
        with open(backup_file, 'w') as f:
            for image_id in original_image_ids:
                f.write(f"{image_id}\n")
        
        # Write cleaned file
        with open(split_file, 'w') as f:
            for image_id in valid_image_ids:
                f.write(f"{image_id}\n")
        
        # Statistics
        stats = {
            'original_count': len(original_image_ids),
            'valid_count': len(valid_image_ids),
            'invalid_count': len(invalid_image_ids),
            'removed_count': len(invalid_image_ids),
            'removal_rate': (len(invalid_image_ids) / len(original_image_ids) * 100) if original_image_ids else 0
        }
        
        logger.info(f"{split_name} split cleaning completed:")
        logger.info(f"  - Original images: {stats['original_count']}")
        logger.info(f"  - Valid images: {stats['valid_count']}")
        logger.info(f"  - Invalid images: {stats['invalid_count']}")
        logger.info(f"  - Removal rate: {stats['removal_rate']:.2f}%")
        
        # Log some examples of invalid images
        if invalid_image_ids:
            logger.info(f"  - Examples of removed images: {invalid_image_ids[:5]}")
        
        return stats
    
    def run(self):
        """Run the complete split cleaning process."""
        logger.info("Starting split cleaning process...")
        
        all_stats = {}
        
        # Clean all splits
        for split_name in ['train', 'val', 'test']:
            stats = self.clean_split_file(split_name)
            all_stats[split_name] = stats
        
        # Save overall statistics
        total_original = sum(stats['original_count'] for stats in all_stats.values())
        total_valid = sum(stats['valid_count'] for stats in all_stats.values())
        total_removed = sum(stats['removed_count'] for stats in all_stats.values())
        
        overall_stats = {
            'total_original': total_original,
            'total_valid': total_valid,
            'total_removed': total_removed,
            'overall_removal_rate': (total_removed / total_original * 100) if total_original > 0 else 0,
            'splits': all_stats
        }
        
        # Save statistics
        stats_file = self.splits_dir / "split_cleaning_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        
        logger.info("Split cleaning completed!")
        logger.info(f"Overall statistics:")
        logger.info(f"  - Total original images: {total_original}")
        logger.info(f"  - Total valid images: {total_valid}")
        logger.info(f"  - Total removed images: {total_removed}")
        logger.info(f"  - Overall removal rate: {overall_stats['overall_removal_rate']:.2f}%")
        logger.info(f"  - Statistics saved to: {stats_file}")

def main():
    """Main function."""
    cleaner = SplitCleaner()
    cleaner.run()

if __name__ == "__main__":
    main() 