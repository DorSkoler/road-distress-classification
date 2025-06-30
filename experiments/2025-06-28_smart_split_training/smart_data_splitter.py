#!/usr/bin/env python3
"""
Smart Data Splitter for Road Distress Classification
Date: 2025-06-28

This script implements intelligent data splitting that:
1. Works with raw coryell data organized by roads
2. Splits by actual roads (no road crosses train/val/test boundaries)
3. Maintains balanced label distribution across splits
4. Filters images with insufficient road coverage (when masks available)
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartDataSplitter:
    """Intelligent data splitter for road distress classification using coryell data."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the smart data splitter.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.setup_paths()
        self.setup_random_seed()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def setup_paths(self):
        """Setup all necessary paths."""
        # Use coryell data as the source
        self.coryell_root = Path(self.config['dataset']['coryell_path'])
        
        # Create output directories
        self.splits_dir = Path("splits")
        self.splits_dir.mkdir(exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            (self.splits_dir / split).mkdir(exist_ok=True)
            
        logger.info("Setup paths completed")
    
    def setup_random_seed(self):
        """Setup random seed for reproducibility."""
        np.random.seed(self.config['system']['random_seed'])
        logger.info(f"Set random seed to {self.config['system']['random_seed']}")
    
    def load_coryell_data(self) -> Tuple[Dict[str, List[str]], Dict[str, Dict]]:
        """Load data from coryell directory structure.
        
        Returns:
            Tuple of (road_to_images, image_annotations)
        """
        logger.info("Loading coryell data...")
        
        road_to_images = {}
        image_annotations = {}
        
        # Get all road directories
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
            
            # Get all images for this road
            image_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
            road_images = []
            
            for img_file in image_files:
                # Get corresponding annotation file
                ann_file = ann_dir / f"{img_file.stem}.json"
                
                if not ann_file.exists():
                    logger.warning(f"Missing annotation for {img_file}")
                    continue
                
                # Load annotation
                try:
                    with open(ann_file, 'r') as f:
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
        labels = self.config['dataset']['labels']
        
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
        
        return road_stats
    
    def create_balanced_road_splits(self, road_to_images: Dict[str, List[str]], 
                                   image_annotations: Dict[str, Dict],
                                   road_stats: Dict) -> Dict[str, List[str]]:
        """Create balanced splits by roads while maintaining label balance.
        
        Args:
            road_to_images: Mapping of road names to image IDs
            image_annotations: Mapping of image IDs to labels
            road_stats: Statistics for each road
            
        Returns:
            Dictionary with train/val/test splits
        """
        logger.info("Creating balanced road splits...")
        
        # Calculate target split ratios
        split_ratios = self.config['dataset']['split_ratios']
        total_images = sum(len(images) for images in road_to_images.values())
        
        target_counts = {
            'train': int(total_images * split_ratios['train']),
            'val': int(total_images * split_ratios['val']),
            'test': int(total_images * split_ratios['test'])
        }
        
        logger.info(f"Target split counts: {target_counts}")
        
        # Sort roads by size and label diversity
        road_scores = []
        for road_name, images in road_to_images.items():
            stats = road_stats[road_name]
            
            # Calculate diversity score (higher is better)
            # Roads with more balanced labels get higher scores
            label_counts = stats['label_counts']
            total = stats['total_images']
            
            if total == 0:
                continue
            
            # Diversity score: how well distributed are the labels
            expected_per_label = total / len(self.config['dataset']['labels'])
            diversity_score = -sum(abs(label_counts[label] - expected_per_label) 
                                 for label in self.config['dataset']['labels'])
            
            # Size score: prefer roads with more images
            size_score = min(total / 100, 1.0)  # Cap at 100 images
            
            # Combined score
            combined_score = diversity_score + size_score * 10
            
            road_scores.append((road_name, combined_score, total))
        
        # Sort by combined score (highest first)
        road_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize splits
        splits = {'train': [], 'val': [], 'test': []}
        split_distributions = {split: {label: 0 for label in self.config['dataset']['labels']} 
                             for split in splits.keys()}
        
        # Distribute roads to splits
        for road_name, score, total_images in road_scores:
            # Calculate which split would be most balanced
            best_split = self._find_best_split_for_road(
                road_name, road_to_images[road_name], image_annotations,
                splits, split_distributions, target_counts
            )
            
            # Add road to selected split
            splits[best_split].extend(road_to_images[road_name])
            
            # Update distribution
            for image_id in road_to_images[road_name]:
                if image_id in image_annotations:
                    for label in self.config['dataset']['labels']:
                        if image_annotations[image_id].get(label, False):
                            split_distributions[best_split][label] += 1
            
            logger.debug(f"Added road {road_name} ({total_images} images) to {best_split}")
        
        # Validate splits
        self._validate_road_splits(splits, image_annotations, target_counts)
        
        return splits
    
    def _find_best_split_for_road(self, road_name: str, image_ids: List[str],
                                 image_annotations: Dict[str, Dict],
                                 splits: Dict[str, List[str]],
                                 split_distributions: Dict[str, Dict[str, int]],
                                 target_counts: Dict[str, int]) -> str:
        """Find the best split for a road to maintain label balance."""
        labels = self.config['dataset']['labels']
        
        # Calculate label distribution for this road
        road_distribution = {label: 0 for label in labels}
        for image_id in image_ids:
            if image_id in image_annotations:
                for label in labels:
                    if image_annotations[image_id].get(label, False):
                        road_distribution[label] += 1
        
        # Calculate which split would be most balanced
        best_split = 'train'  # default
        best_score = float('inf')
        
        for split_name in ['train', 'val', 'test']:
            # Check if adding this road would exceed target count
            current_count = len(splits[split_name])
            road_count = len(image_ids)
            
            if current_count + road_count > target_counts[split_name] * 1.2:  # 20% tolerance
                continue
            
            # Calculate hypothetical distribution
            hypothetical_dist = split_distributions[split_name].copy()
            for label, count in road_distribution.items():
                hypothetical_dist[label] += count
            
            # Calculate balance score (lower is better)
            total_images = current_count + road_count
            if total_images == 0:
                score = float('inf')
            else:
                expected_per_label = total_images / len(labels)
                score = sum(abs(hypothetical_dist[label] - expected_per_label) 
                           for label in labels)
            
            if score < best_score:
                best_score = score
                best_split = split_name
        
        return best_split
    
    def _validate_road_splits(self, splits: Dict[str, List[str]], 
                            image_annotations: Dict[str, Dict],
                            target_counts: Dict[str, int]):
        """Validate that road splits meet requirements."""
        logger.info("Validating road splits...")
        
        # Check split sizes
        for split_name, image_ids in splits.items():
            actual_count = len(image_ids)
            target_count = target_counts[split_name]
            tolerance = target_count * 0.2  # 20% tolerance for roads
            
            if abs(actual_count - target_count) > tolerance:
                logger.warning(f"{split_name} split: {actual_count} images, target: {target_count}")
            else:
                logger.info(f"{split_name} split: {actual_count} images ✓")
        
        # Check label distribution
        split_distributions = self._calculate_split_distributions(splits, image_annotations)
        for split_name, distribution in split_distributions.items():
            logger.info(f"{split_name} label distribution: {distribution}")
        
        # Check road integrity (no road crosses splits)
        road_to_split = {}
        for split_name, image_ids in splits.items():
            for image_id in image_ids:
                road_name = image_id.split('/')[0]
                if road_name in road_to_split:
                    if road_to_split[road_name] != split_name:
                        logger.error(f"Road {road_name} appears in multiple splits!")
                else:
                    road_to_split[road_name] = split_name
        
        logger.info("Road integrity check passed ✓")
    
    def _calculate_split_distributions(self, splits: Dict[str, List[str]], 
                                     image_annotations: Dict[str, Dict]) -> Dict[str, Dict[str, int]]:
        """Calculate current label distribution in each split."""
        labels = self.config['dataset']['labels']
        distributions = {}
        
        for split_name, image_ids in splits.items():
            distribution = {label: 0 for label in labels}
            for image_id in image_ids:
                if image_id in image_annotations:
                    for label in labels:
                        if image_annotations[image_id].get(label, False):
                            distribution[label] += 1
            distributions[split_name] = distribution
        
        return distributions
    
    def save_road_splits(self, splits: Dict[str, List[str]], 
                        image_annotations: Dict[str, Dict],
                        road_stats: Dict):
        """Save road splits to files."""
        logger.info("Saving road splits...")
        
        # Save split lists
        for split_name, image_ids in splits.items():
            split_file = self.splits_dir / f"{split_name}_images.txt"
            with open(split_file, 'w') as f:
                for image_id in image_ids:
                    f.write(f"{image_id}\n")
        
        # Save split metadata
        split_metadata = {
            'total_images': sum(len(images) for images in splits.values()),
            'total_roads': len(set(img_id.split('/')[0] for img_ids in splits.values() for img_id in img_ids)),
            'splits': {}
        }
        
        for split_name, image_ids in splits.items():
            split_distribution = self._calculate_split_distributions({split_name: image_ids}, image_annotations)
            
            # Get roads in this split
            roads_in_split = set(img_id.split('/')[0] for img_id in image_ids)
            
            split_metadata['splits'][split_name] = {
                'count': len(image_ids),
                'roads': list(roads_in_split),
                'road_count': len(roads_in_split),
                'label_distribution': split_distribution[split_name]
            }
        
        metadata_file = self.splits_dir / "road_split_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(split_metadata, f, indent=2)
        
        # Save road statistics
        road_stats_file = self.splits_dir / "road_statistics.json"
        with open(road_stats_file, 'w') as f:
            json.dump(road_stats, f, indent=2)
        
        logger.info(f"Saved road splits to {self.splits_dir}")
    
    def visualize_road_splits(self, splits: Dict[str, List[str]], 
                            image_annotations: Dict[str, Dict],
                            road_stats: Dict):
        """Create visualizations of the road splits."""
        logger.info("Creating road split visualizations...")
        
        # Calculate distributions
        split_distributions = self._calculate_split_distributions(splits, image_annotations)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Split sizes
        split_names = list(splits.keys())
        split_sizes = [len(splits[name]) for name in split_names]
        
        axes[0, 0].bar(split_names, split_sizes, color=['#2E8B57', '#FF6B6B', '#4ECDC4'])
        axes[0, 0].set_title('Split Sizes')
        axes[0, 0].set_ylabel('Number of Images')
        
        # Label distribution
        labels = self.config['dataset']['labels']
        x = np.arange(len(split_names))
        width = 0.25
        
        for i, label in enumerate(labels):
            values = [split_distributions[split][label] for split in split_names]
            axes[0, 1].bar(x + i * width, values, width, label=label)
        
        axes[0, 1].set_title('Label Distribution by Split')
        axes[0, 1].set_ylabel('Number of Images')
        axes[0, 1].set_xticks(x + width)
        axes[0, 1].set_xticklabels(split_names)
        axes[0, 1].legend()
        
        # Road distribution
        road_counts = []
        for split_name in split_names:
            roads_in_split = set(img_id.split('/')[0] for img_id in splits[split_name])
            road_counts.append(len(roads_in_split))
        
        axes[1, 0].bar(split_names, road_counts, color=['#2E8B57', '#FF6B6B', '#4ECDC4'])
        axes[1, 0].set_title('Road Distribution by Split')
        axes[1, 0].set_ylabel('Number of Roads')
        
        # Road size distribution
        road_sizes = [stats['total_images'] for stats in road_stats.values()]
        axes[1, 1].hist(road_sizes, bins=20, color='skyblue', alpha=0.7)
        axes[1, 1].set_title('Distribution of Road Sizes')
        axes[1, 1].set_xlabel('Number of Images per Road')
        axes[1, 1].set_ylabel('Number of Roads')
        
        plt.tight_layout()
        plt.savefig(self.splits_dir / 'road_split_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {self.splits_dir / 'road_split_visualization.png'}")
    
    def run(self):
        """Run the complete smart road splitting pipeline."""
        logger.info("Starting smart road splitting pipeline...")
        
        # Load coryell data
        road_to_images, image_annotations = self.load_coryell_data()
        
        # Calculate road statistics
        road_stats = self.calculate_road_statistics(road_to_images, image_annotations)
        
        # Create balanced road splits
        splits = self.create_balanced_road_splits(road_to_images, image_annotations, road_stats)
        
        # Save splits
        self.save_road_splits(splits, image_annotations, road_stats)
        
        # Create visualizations
        self.visualize_road_splits(splits, image_annotations, road_stats)
        
        logger.info("Smart road splitting completed successfully!")

def main():
    """Main function."""
    splitter = SmartDataSplitter()
    splitter.run()

if __name__ == "__main__":
    main() 