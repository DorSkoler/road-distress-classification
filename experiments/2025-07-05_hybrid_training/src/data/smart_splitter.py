#!/usr/bin/env python3
"""
Smart Data Splitter for Road Distress Classification - 2025-07-05 Hybrid Training
Adapted from the successful 2025-06-28 experiment
Cross-platform compatible for Mac and Windows

This script implements intelligent data splitting that:
1. Works with raw coryell data organized by roads
2. Splits by actual roads (no road crosses train/val/test boundaries) 
3. Maintains balanced label distribution across splits
4. Creates file lists for the hybrid training pipeline
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from collections import defaultdict
import sys

# Add utils to path for cross-platform support
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from platform_utils import PlatformManager, setup_cross_platform_logging

# Configure logging
logger = logging.getLogger(__name__)

class HybridSmartDataSplitter:
    """Intelligent data splitter for hybrid training experiment with cross-platform support."""
    
    def __init__(self, config_path: str = "config/base_config.yaml"):
        """Initialize the smart data splitter.
        
        Args:
            config_path: Path to configuration file
        """
        self.platform_manager = PlatformManager()
        self.config = self.platform_manager.load_config(config_path)
        self.setup_platform_specific_settings()
        self.setup_paths()
        self.setup_random_seed()
        
        # Setup cross-platform logging
        setup_cross_platform_logging(self.config)
        
    def setup_platform_specific_settings(self):
        """Setup platform-specific settings."""
        # Update dataset config with platform-specific workers
        self.config['dataset']['num_workers'] = self.platform_manager.get_num_workers()
        
        # Update hardware config with platform-specific device
        self.config['hardware']['device'] = self.platform_manager.get_device()
        
        logger.info(f"Platform: {self.platform_manager.platform_info['os']}")
        logger.info(f"Device: {self.config['hardware']['device']}")
        logger.info(f"Workers: {self.config['dataset']['num_workers']}")
        
    def setup_paths(self):
        """Setup all necessary paths using cross-platform utilities."""
        # Use cross-platform path handling
        coryell_path = self.config['dataset']['coryell_path']
        self.coryell_root = self.platform_manager.normalize_path(coryell_path)
        
        # Create output directories
        self.splits_dir = self.platform_manager.create_directory("data/splits")
        
        logger.info(f"Coryell data path: {self.coryell_root}")
        logger.info(f"Splits output directory: {self.splits_dir}")
    
    def setup_random_seed(self):
        """Setup random seed for reproducibility."""
        seed = self.config.get('system', {}).get('random_seed', 42)
        np.random.seed(seed)
        logger.info(f"Set random seed to {seed}")
    
    def load_coryell_data(self) -> Tuple[Dict[str, List[str]], Dict[str, Dict]]:
        """Load data from coryell directory structure using cross-platform file handling.
        
        Returns:
            Tuple of (road_to_images, image_annotations)
        """
        logger.info("Loading coryell data...")
        
        road_to_images = {}
        image_annotations = {}
        
        # Get all road directories using cross-platform approach
        if not self.coryell_root.exists():
            logger.error(f"Coryell data directory not found: {self.coryell_root}")
            raise FileNotFoundError(f"Coryell data directory not found: {self.coryell_root}")
        
        road_dirs = [d for d in self.coryell_root.iterdir() if d.is_dir() and d.name != '__pycache__']
        road_dirs = [d for d in road_dirs if d.name.startswith('Co Rd')]
        
        logger.info(f"Found {len(road_dirs)} roads")
        
        for road_dir in tqdm(road_dirs, desc="Processing roads"):
            road_name = road_dir.name
            img_dir = road_dir / "img"
            ann_dir = road_dir / "ann"
            
            if not img_dir.exists() or not ann_dir.exists():
                logger.warning(f"Missing img or ann directory for {road_name}")
                continue
            
            # Get all images for this road using cross-platform file extensions
            image_files = self.platform_manager.find_files(img_dir, 'image')
            road_images = []
            
            for img_file in image_files:
                # Get corresponding annotation file
                ann_file = ann_dir / f"{img_file.stem}.json"
                
                if not ann_file.exists():
                    logger.warning(f"Missing annotation for {img_file}")
                    continue
                
                # Load annotation
                try:
                    with open(ann_file, 'r', encoding='utf-8') as f:
                        ann_data = json.load(f)
                    
                    # Extract labels from tags
                    labels = self._extract_labels_from_annotation(ann_data)
                    
                    # Create unique image ID (road_name + image_name)
                    image_id = f"{road_name}/{img_file.stem}"
                    
                    # Store annotation
                    image_annotations[image_id] = labels
                    road_images.append(image_id)
                    
                except Exception as e:
                    logger.warning(f"Error loading annotation {ann_file}: {e}")
                    continue
            
            if road_images:
                road_to_images[road_name] = road_images
                logger.debug(f"Road {road_name}: {len(road_images)} images")
        
        logger.info(f"Loaded {len(image_annotations)} images from {len(road_to_images)} roads")
        return road_to_images, image_annotations
    
    def _extract_labels_from_annotation(self, ann_data: Dict) -> Dict[str, bool]:
        """Extract binary labels from annotation tags.
        
        Args:
            ann_data: Annotation data from JSON file
            
        Returns:
            Dictionary with binary labels
        """
        labels = {
            'damaged': False,
            'occlusion': False,
            'cropped': False
        }
        
        if 'tags' not in ann_data:
            return labels
        
        for tag in ann_data['tags']:
            name = tag.get('name', '').lower()
            value = tag.get('value', '').lower()
            
            if name == 'damage':
                labels['damaged'] = value == 'damaged'
            elif name == 'occlusion':
                labels['occlusion'] = value == 'occluded'
            elif name == 'crop':
                labels['cropped'] = value == 'cropped'
        
        return labels
    
    def calculate_road_statistics(self, road_to_images: Dict[str, List[str]], 
                                image_annotations: Dict[str, Dict]) -> Dict:
        """Calculate statistics for each road.
        
        Args:
            road_to_images: Mapping of road names to image IDs
            image_annotations: Mapping of image IDs to labels
            
        Returns:
            Dictionary with road statistics
        """
        logger.info("Calculating road statistics...")
        
        road_stats = {}
        labels = ['damaged', 'occlusion', 'cropped']
        
        for road_name, image_ids in road_to_images.items():
            # Count images per label for this road
            label_counts = {label: 0 for label in labels}
            total_images = 0
            
            for image_id in image_ids:
                if image_id in image_annotations:
                    total_images += 1
                    for label in labels:
                        if image_annotations[image_id].get(label, False):
                            label_counts[label] += 1
            
            # Calculate percentages
            label_percentages = {}
            for label in labels:
                label_percentages[label] = label_counts[label] / max(total_images, 1)
            
            road_stats[road_name] = {
                'total_images': total_images,
                'label_counts': label_counts,
                'label_percentages': label_percentages
            }
        
        logger.info("Road statistics calculated")
        return road_stats
    
    def create_balanced_road_splits(self, road_to_images: Dict[str, List[str]], 
                                   image_annotations: Dict[str, Dict],
                                   road_stats: Dict) -> Dict[str, List[str]]:
        """Create balanced splits by road ensuring no road appears in multiple splits.
        
        Args:
            road_to_images: Mapping of road names to image IDs
            image_annotations: Mapping of image IDs to labels
            road_stats: Statistics for each road
            
        Returns:
            Dictionary with splits
        """
        logger.info("Creating balanced road splits...")
        
        # Get split ratios from config
        ratios = self.config['splitting']
        train_ratio = ratios['train_ratio']
        val_ratio = ratios['val_ratio']
        test_ratio = ratios['test_ratio']
        
        # Calculate target counts
        total_images = sum(len(images) for images in road_to_images.values())
        target_counts = {
            'train': int(total_images * train_ratio),
            'val': int(total_images * val_ratio),
            'test': int(total_images * test_ratio)
        }
        
        # Initialize splits
        splits = {'train': [], 'val': [], 'test': []}
        split_distributions = {'train': defaultdict(int), 'val': defaultdict(int), 'test': defaultdict(int)}
        
        # Sort roads by total images (process larger roads first)
        sorted_roads = sorted(road_to_images.keys(), 
                            key=lambda x: road_stats[x]['total_images'], 
                            reverse=True)
        
        # Assign each road to a split
        for road_name in sorted_roads:
            image_ids = road_to_images[road_name]
            
            # Find best split for this road
            best_split = self._find_best_split_for_road(
                road_name, image_ids, image_annotations, 
                splits, split_distributions, target_counts
            )
            
            # Assign road to best split
            splits[best_split].extend(image_ids)
            
            # Update distributions
            for image_id in image_ids:
                if image_id in image_annotations:
                    for label in ['damaged', 'occlusion', 'cropped']:
                        if image_annotations[image_id].get(label, False):
                            split_distributions[best_split][label] += 1
        
        # Validate splits
        self._validate_road_splits(splits, image_annotations, target_counts)
        
        logger.info("Balanced road splits created")
        return splits
    
    def _find_best_split_for_road(self, road_name: str, image_ids: List[str],
                                 image_annotations: Dict[str, Dict],
                                 splits: Dict[str, List[str]],
                                 split_distributions: Dict[str, Dict[str, int]],
                                 target_counts: Dict[str, int]) -> str:
        """Find the best split for a road based on current distributions.
        
        Args:
            road_name: Name of the road
            image_ids: List of image IDs for this road
            image_annotations: Mapping of image IDs to labels
            splits: Current splits
            split_distributions: Current label distributions per split
            target_counts: Target image counts per split
            
        Returns:
            Best split name for this road
        """
        # Calculate current split sizes
        current_sizes = {split: len(images) for split, images in splits.items()}
        
        # Calculate how far each split is from target
        size_scores = {}
        for split in ['train', 'val', 'test']:
            if current_sizes[split] < target_counts[split]:
                size_scores[split] = target_counts[split] - current_sizes[split]
            else:
                size_scores[split] = 0
        
        # If no split needs more images, choose the one with smallest current size
        if all(score == 0 for score in size_scores.values()):
            return min(current_sizes.keys(), key=lambda x: current_sizes[x])
        
        # Choose split with highest need for more images
        return max(size_scores.keys(), key=lambda x: size_scores[x])
    
    def _validate_road_splits(self, splits: Dict[str, List[str]], 
                            image_annotations: Dict[str, Dict],
                            target_counts: Dict[str, int]):
        """Validate that splits are reasonable.
        
        Args:
            splits: Dictionary with splits
            image_annotations: Mapping of image IDs to labels
            target_counts: Target image counts per split
        """
        logger.info("Validating road splits...")
        
        for split_name, image_ids in splits.items():
            actual_count = len(image_ids)
            target_count = target_counts[split_name]
            
            logger.info(f"{split_name}: {actual_count} images (target: {target_count})")
            
            # Check for reasonable balance
            if actual_count == 0:
                logger.warning(f"Split {split_name} has no images!")
        
        # Check for overlap (this should never happen with road-based splitting)
        all_images = set()
        for image_ids in splits.values():
            for image_id in image_ids:
                if image_id in all_images:
                    logger.error(f"Image {image_id} appears in multiple splits!")
                all_images.add(image_id)
        
        logger.info("Split validation completed")
    
    def save_splits(self, splits: Dict[str, List[str]], 
                   image_annotations: Dict[str, Dict],
                   road_stats: Dict):
        """Save splits to text files for the hybrid training pipeline.
        
        Args:
            splits: Dictionary with splits
            image_annotations: Mapping of image IDs to labels
            road_stats: Statistics for each road
        """
        logger.info("Saving splits...")
        
        # Save image lists for each split using cross-platform text saving
        for split_name, image_ids in splits.items():
            # Create content with platform-appropriate line endings
            content = '\n'.join(image_ids)
            
            # Save using cross-platform method
            file_path = self.splits_dir / f"{split_name}_images.txt"
            self.platform_manager.save_text_file(content, file_path)
            
            logger.info(f"Saved {len(image_ids)} images to {split_name}_images.txt")
        
        # Save split statistics
        split_stats = {}
        for split_name, image_ids in splits.items():
            stats = {'total_images': len(image_ids)}
            
            # Count labels
            label_counts = {'damaged': 0, 'occlusion': 0, 'cropped': 0}
            for image_id in image_ids:
                if image_id in image_annotations:
                    for label in label_counts.keys():
                        if image_annotations[image_id].get(label, False):
                            label_counts[label] += 1
            
            stats['label_counts'] = label_counts
            stats['label_percentages'] = {
                label: count / max(len(image_ids), 1) 
                for label, count in label_counts.items()
            }
            
            split_stats[split_name] = stats
        
        # Save statistics using cross-platform JSON saving
        stats_path = self.splits_dir / "split_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(split_stats, f, indent=2)
        
        logger.info("Splits saved successfully")
    
    def visualize_splits(self, splits: Dict[str, List[str]], 
                        image_annotations: Dict[str, Dict]):
        """Create visualizations of the splits.
        
        Args:
            splits: Dictionary with splits
            image_annotations: Mapping of image IDs to labels
        """
        logger.info("Creating split visualizations...")
        
        # Prepare data for visualization
        data = []
        for split_name, image_ids in splits.items():
            for image_id in image_ids:
                if image_id in image_annotations:
                    labels = image_annotations[image_id]
                    data.append({
                        'split': split_name,
                        'image_id': image_id,
                        'damaged': labels.get('damaged', False),
                        'occlusion': labels.get('occlusion', False),
                        'cropped': labels.get('cropped', False)
                    })
        
        df = pd.DataFrame(data)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Split sizes
        split_sizes = df['split'].value_counts()
        axes[0, 0].bar(split_sizes.index, split_sizes.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('Split Sizes')
        axes[0, 0].set_ylabel('Number of Images')
        
        # Label distributions per split
        label_cols = ['damaged', 'occlusion', 'cropped']
        for i, label in enumerate(label_cols):
            ax = axes[0, 1] if i == 0 else axes[1, i-1]
            
            label_counts = df.groupby('split')[label].sum()
            label_counts.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_title(f'{label.capitalize()} Distribution')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save using cross-platform path
        viz_path = self.splits_dir / "split_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Split visualizations created")
    
    def run(self):
        """Run the complete splitting pipeline."""
        logger.info("Starting hybrid smart data splitting...")
        
        # Load data
        road_to_images, image_annotations = self.load_coryell_data()
        
        # Calculate road statistics
        road_stats = self.calculate_road_statistics(road_to_images, image_annotations)
        
        # Create balanced splits
        splits = self.create_balanced_road_splits(road_to_images, image_annotations, road_stats)
        
        # Save splits
        self.save_splits(splits, image_annotations, road_stats)
        
        # Create visualizations
        self.visualize_splits(splits, image_annotations)
        
        logger.info("Hybrid smart data splitting completed successfully!")
        
        return splits, image_annotations, road_stats

def main():
    """Main function to run the smart data splitter."""
    try:
        splitter = HybridSmartDataSplitter()
        splits, annotations, stats = splitter.run()
        
        # Print summary
        print("\n" + "="*50)
        print("SPLIT SUMMARY")
        print("="*50)
        for split_name, image_ids in splits.items():
            print(f"{split_name.upper()}: {len(image_ids)} images")
        
        print(f"\nTotal images: {sum(len(images) for images in splits.values())}")
        print("Split files saved to data/splits/")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in data splitting: {e}")
        raise

if __name__ == "__main__":
    main() 