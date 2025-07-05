#!/usr/bin/env python3
"""
Cross-Platform Dataset for Hybrid Training Experiment
Date: 2025-07-05

This module implements a flexible dataset class that supports all 4 model variants:
- Model A: Pictures + masks
- Model B: Pictures + augmentation (no masks)
- Model C: Pictures + augmentation + masks
- Model D: Pictures + augmentation + weighted masks

Adapted from successful architectures with cross-platform compatibility.
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from PIL import Image
import random

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.platform_utils import PlatformUtils

logger = logging.getLogger(__name__)

class HybridRoadDataset(Dataset):
    """
    Cross-platform dataset for hybrid road distress classification.
    
    Supports all 4 model variants with different input strategies:
    - Model A: Original images + masks
    - Model B: Original + augmented images (no masks)
    - Model C: Original + augmented images + masks
    - Model D: Original + augmented images + weighted masks
    """
    
    def __init__(self, 
                 split_name: str,
                 config: Dict,
                 variant: str = 'model_a',
                 transform=None,
                 use_augmented: bool = None,
                 use_masks: bool = None):
        """
        Initialize the dataset.
        
        Args:
            split_name: Name of the split (train, val, test)
            config: Configuration dictionary
            variant: Model variant name ('model_a', 'model_b', 'model_c', 'model_d')
            transform: Optional transforms to apply
            use_augmented: Override augmentation usage
            use_masks: Override mask usage
        """
        self.split_name = split_name
        self.config = config
        self.variant = variant
        self.transform = transform
        self.platform_utils = PlatformUtils()
        
        # Setup paths
        self.setup_paths()
        
        # Determine variant configuration
        self.setup_variant_config(use_augmented, use_masks)
        
        # Load data
        self.load_data()
        
        logger.info(f"Initialized HybridRoadDataset:")
        logger.info(f"  - Split: {split_name}")
        logger.info(f"  - Variant: {variant}")
        logger.info(f"  - Use augmented: {self.use_augmented}")
        logger.info(f"  - Use masks: {self.use_masks}")
        logger.info(f"  - Total samples: {len(self.samples)}")
    
    def setup_paths(self):
        """Setup all necessary paths with cross-platform compatibility."""
        # Data paths
        coryell_path = self.config['dataset']['coryell_path']
        self.coryell_root = Path(coryell_path).resolve()
        
        # Use platform-specific path handling
        self.splits_dir = self.platform_utils.get_data_path("splits")
        self.masks_dir = self.platform_utils.get_data_path("masks")
        self.augmented_dir = self.platform_utils.get_data_path("augmented")
        
        logger.debug(f"Dataset paths setup - Coryell: {self.coryell_root}")
    
    def setup_variant_config(self, use_augmented: Optional[bool], use_masks: Optional[bool]):
        """Setup configuration based on model variant."""
        # Default configurations for each variant
        variant_configs = {
            'model_a': {'use_augmented': False, 'use_masks': True},
            'model_b': {'use_augmented': True, 'use_masks': False},
            'model_c': {'use_augmented': True, 'use_masks': True},
            'model_d': {'use_augmented': True, 'use_masks': True}
        }
        
        if self.variant not in variant_configs:
            raise ValueError(f"Unknown variant: {self.variant}. Available: {list(variant_configs.keys())}")
        
        # Use override values if provided, otherwise use variant defaults
        config = variant_configs[self.variant]
        self.use_augmented = use_augmented if use_augmented is not None else config['use_augmented']
        self.use_masks = use_masks if use_masks is not None else config['use_masks']
    
    def load_data(self):
        """Load data samples based on configuration."""
        # Load split data
        split_file = self.splits_dir / f"{self.split_name}_images.txt"
        
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r', encoding='utf-8') as f:
            original_image_ids = [line.strip() for line in f.readlines() if line.strip()]
        
        self.samples = []
        
        # Add original images
        for image_id in original_image_ids:
            self.samples.append({
                'image_id': image_id,
                'is_augmented': False,
                'aug_suffix': None
            })
        
        # Add augmented images if needed
        if self.use_augmented:
            self.add_augmented_samples(original_image_ids)
        
        logger.info(f"Loaded {len(self.samples)} samples ({len(original_image_ids)} original)")
    
    def add_augmented_samples(self, original_image_ids: List[str]):
        """Add augmented samples to the dataset."""
        augmented_images_dir = self.augmented_dir / self.split_name / "images"
        
        if not augmented_images_dir.exists():
            logger.warning(f"Augmented images directory not found: {augmented_images_dir}")
            return
        
        # Find augmented versions for each original image
        for image_id in original_image_ids:
            try:
                road_name, image_name = image_id.split('/', 1)
                road_dir = augmented_images_dir / road_name
                
                if not road_dir.exists():
                    continue
                
                # Find all augmented versions of this image
                aug_pattern = f"{image_name}_aug"
                for aug_file in road_dir.glob(f"{aug_pattern}*.png"):
                    # Extract augmentation suffix
                    aug_suffix = aug_file.stem.replace(f"{image_name}_", "")
                    
                    self.samples.append({
                        'image_id': image_id,
                        'is_augmented': True,
                        'aug_suffix': aug_suffix
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing augmented samples for {image_id}: {e}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                           Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            If use_masks=True: (image, mask, labels)
            If use_masks=False: (image, labels)
        """
        sample = self.samples[idx]
        
        # Load image
        image = self.load_image(sample)
        if image is None:
            # Return dummy data if image loading fails
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Load mask if needed
        mask = None
        if self.use_masks:
            mask = self.load_mask(sample)
            if mask is None:
                # Return dummy mask if mask loading fails
                mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Load labels
        labels = self.load_labels(sample)
        
        # Apply transforms
        if self.transform:
            if mask is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            else:
                transformed = self.transform(image=image)
                image = transformed['image']
        else:
            # Default preprocessing
            image = self.preprocess_image(image)
            if mask is not None:
                mask = self.preprocess_mask(mask)
        
        # Return appropriate format
        if self.use_masks:
            return image, mask, labels
        else:
            return image, labels
    
    def load_image(self, sample: Dict) -> Optional[np.ndarray]:
        """Load image from sample information."""
        try:
            image_id = sample['image_id']
            road_name, image_name = image_id.split('/', 1)
            
            if sample['is_augmented']:
                # Load augmented image
                aug_filename = f"{image_name}_{sample['aug_suffix']}.png"
                image_path = self.augmented_dir / self.split_name / "images" / road_name / aug_filename
            else:
                # Load original image
                image_path = self.coryell_root / road_name / "img" / f"{image_name}.png"
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                return None
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image_rgb
            
        except Exception as e:
            logger.warning(f"Error loading image for sample {sample}: {e}")
            return None
    
    def load_mask(self, sample: Dict) -> Optional[np.ndarray]:
        """Load mask from sample information."""
        try:
            image_id = sample['image_id']
            road_name, image_name = image_id.split('/', 1)
            
            if sample['is_augmented']:
                # Load augmented mask
                aug_filename = f"{image_name}_{sample['aug_suffix']}.png"
                mask_path = self.augmented_dir / self.split_name / "masks" / road_name / aug_filename
            else:
                # Load original mask
                mask_path = self.masks_dir / self.split_name / road_name / f"{image_name}.png"
            
            if not mask_path.exists():
                logger.warning(f"Mask not found: {mask_path}")
                return None
            
            # Load mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Failed to load mask: {mask_path}")
                return None
            
            return mask
            
        except Exception as e:
            logger.warning(f"Error loading mask for sample {sample}: {e}")
            return None
    
    def load_labels(self, sample: Dict) -> torch.Tensor:
        """Load labels from sample information."""
        try:
            image_id = sample['image_id']
            road_name, image_name = image_id.split('/', 1)
            
            # Labels are always from the original image
            label_path = self.coryell_root / road_name / "img" / f"{image_name}.png.json"
            
            if not label_path.exists():
                logger.warning(f"Label file not found: {label_path}")
                return torch.zeros(3, dtype=torch.float32)
            
            with open(label_path, 'r', encoding='utf-8') as f:
                img_data = json.load(f)
            
            # Extract labels (same as successful 10/05 approach)
            labels = torch.zeros(3, dtype=torch.float32)
            
            for tag in img_data.get('tags', []):
                if tag['name'] == 'Damage':
                    labels[0] = 1.0 if tag['value'] == 'Damaged' else 0.0
                elif tag['name'] == 'Occlusion':
                    labels[1] = 1.0 if tag['value'] == 'Occluded' else 0.0
                elif tag['name'] == 'Crop':
                    labels[2] = 1.0 if tag['value'] == 'Cropped' else 0.0
            
            return labels
            
        except Exception as e:
            logger.warning(f"Error loading labels for sample {sample}: {e}")
            return torch.zeros(3, dtype=torch.float32)
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image to tensor."""
        # Get target size from config
        target_size = tuple(self.config['dataset']['image_size'])
        
        # Resize image
        image_resized = cv2.resize(image, target_size)
        
        # Normalize to [0, 1] and convert to tensor
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)
        
        return image_tensor
    
    def preprocess_mask(self, mask: np.ndarray) -> torch.Tensor:
        """Preprocess mask to tensor."""
        # Get target size from config
        target_size = tuple(self.config['dataset']['image_size'])
        
        # Resize mask
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize to [0, 1] and convert to tensor
        mask_normalized = mask_resized.astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_normalized).unsqueeze(0)
        
        return mask_tensor
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution in the dataset."""
        class_counts = {'damage': 0, 'occlusion': 0, 'crop': 0}
        
        for i in range(len(self.samples)):
            try:
                labels = self.load_labels(self.samples[i])
                if labels[0] > 0.5:
                    class_counts['damage'] += 1
                if labels[1] > 0.5:
                    class_counts['occlusion'] += 1
                if labels[2] > 0.5:
                    class_counts['crop'] += 1
            except Exception as e:
                logger.warning(f"Error getting class distribution for sample {i}: {e}")
        
        return class_counts
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a specific sample."""
        if idx >= len(self.samples):
            raise IndexError(f"Sample index {idx} out of range")
        
        sample = self.samples[idx]
        return {
            'index': idx,
            'image_id': sample['image_id'],
            'is_augmented': sample['is_augmented'],
            'aug_suffix': sample['aug_suffix'],
            'variant': self.variant,
            'use_masks': self.use_masks,
            'use_augmented': self.use_augmented
        }


def create_dataset(split_name: str, config: Dict, variant: str = 'model_a', **kwargs) -> HybridRoadDataset:
    """
    Convenience function to create a dataset for a specific variant.
    
    Args:
        split_name: Name of the split (train, val, test)
        config: Configuration dictionary
        variant: Model variant name
        **kwargs: Additional arguments for dataset
        
    Returns:
        HybridRoadDataset instance
    """
    return HybridRoadDataset(split_name, config, variant, **kwargs)


def get_dataset_stats(dataset: HybridRoadDataset) -> Dict:
    """
    Get comprehensive statistics about a dataset.
    
    Args:
        dataset: HybridRoadDataset instance
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_samples': len(dataset),
        'variant': dataset.variant,
        'split': dataset.split_name,
        'use_masks': dataset.use_masks,
        'use_augmented': dataset.use_augmented,
        'class_distribution': dataset.get_class_distribution()
    }
    
    # Calculate augmentation ratio
    original_count = sum(1 for s in dataset.samples if not s['is_augmented'])
    augmented_count = sum(1 for s in dataset.samples if s['is_augmented'])
    
    stats['original_samples'] = original_count
    stats['augmented_samples'] = augmented_count
    stats['augmentation_ratio'] = augmented_count / original_count if original_count > 0 else 0
    
    return stats


if __name__ == "__main__":
    # Test dataset creation
    print("Testing dataset variants...")
    
    # Mock configuration
    config = {
        'dataset': {
            'coryell_path': '../../data/coryell',
            'image_size': [256, 256]
        }
    }
    
    variants = ['model_a', 'model_b', 'model_c', 'model_d']
    
    for variant in variants:
        print(f"\n{variant.upper()}:")
        try:
            dataset = create_dataset('train', config, variant)
            stats = get_dataset_stats(dataset)
            
            print(f"  - Total samples: {stats['total_samples']}")
            print(f"  - Use masks: {stats['use_masks']}")
            print(f"  - Use augmented: {stats['use_augmented']}")
            print(f"  - Original samples: {stats['original_samples']}")
            print(f"  - Augmented samples: {stats['augmented_samples']}")
            print(f"  - Augmentation ratio: {stats['augmentation_ratio']:.2f}")
            print(f"  - ✓ Dataset created successfully")
            
        except Exception as e:
            print(f"  - ✗ Error: {e}")
    
    print("\n✓ Dataset testing completed!") 