import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict

class OrganizedDatasetLoader:
    def __init__(self, dataset_path: str):
        """
        Initialize the dataset loader with the organized dataset path.
        
        Args:
            dataset_path: Path to the organized dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.metadata_path = self.dataset_path / 'metadata.json'
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Cache for loaded data
        self._cache = {}
    
    def get_split_paths(self, split: str) -> Tuple[Path, Path]:
        """Get paths for images and annotations of a specific split."""
        return (
            self.dataset_path / 'images' / split,
            self.dataset_path / 'annotations' / split
        )
    
    def load_split(self, split: str, cache: bool = True) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Load all images and annotations for a specific split.
        
        Args:
            split: One of 'train', 'val', or 'test'
            cache: Whether to cache the loaded data
            
        Returns:
            Tuple of (images, annotations)
        """
        if cache and split in self._cache:
            return self._cache[split]
            
        img_dir, ann_dir = self.get_split_paths(split)
        images = []
        annotations = []
        
        # Load all images and annotations
        for img_path in img_dir.glob('*'):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.tif']:
                continue
                
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue
                
            # Load annotation
            ann_path = ann_dir / f"{img_path.stem}.json"
            if ann_path.exists():
                with open(ann_path, 'r') as f:
                    ann = json.load(f)
            else:
                ann = {"damage": "Not Damaged", "occlusion": "Not Occluded", "crop": "Not Cropped"}
            
            images.append(img)
            annotations.append(ann)
        
        if cache:
            self._cache[split] = (images, annotations)
            
        return images, annotations
    
    def get_split_stats(self, split: str) -> Dict:
        """Get statistics for a specific split."""
        return self.metadata['splits'][split]
    
    def get_image_path(self, split: str, filename: str) -> Path:
        """Get the full path for an image in a specific split."""
        return self.dataset_path / 'images' / split / filename
    
    def get_annotation_path(self, split: str, filename: str) -> Path:
        """Get the full path for an annotation in a specific split."""
        return self.dataset_path / 'annotations' / split / f"{Path(filename).stem}.json" 