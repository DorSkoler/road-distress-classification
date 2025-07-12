#!/usr/bin/env python3
"""
CLAHE-Enhanced Dataset for Road Distress Classification

This dataset class loads images and applies optimal CLAHE parameters
from a JSON file during training, with configurable mask opacity.
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, Optional, List


class CLAHEDataset(Dataset):
    """Dataset that applies optimized CLAHE parameters from JSON"""
    
    def __init__(
        self, 
        images_dir: str,
        masks_dir: str, 
        labels_file: str,
        clahe_params_json: str,
        mask_opacity: float = 1.0,
        img_size: int = 256,
        transform=None
    ):
        """
        Args:
            images_dir: Directory containing original images
            masks_dir: Directory containing masks
            labels_file: Path to labels CSV file
            clahe_params_json: Path to CLAHE parameters JSON
            mask_opacity: Opacity for mask overlay (0.0-1.0)
            img_size: Target image size
            transform: Additional transforms (applied after CLAHE)
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_opacity = mask_opacity
        self.img_size = img_size
        self.transform = transform
        
        # Load labels
        self.labels_df = pd.read_csv(labels_file)
        
        # Load CLAHE parameters
        self.clahe_params = self.load_clahe_params(clahe_params_json)
        
        # Filter samples to only include those with CLAHE parameters
        self.samples = self.create_sample_list()
        
        print(f"CLAHEDataset initialized:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Mask opacity: {mask_opacity}")
        print(f"  CLAHE params loaded: {len(self.clahe_params)}")
    
    def load_clahe_params(self, json_path: str) -> Dict[str, Dict]:
        """Load CLAHE parameters from JSON"""
        params = {}
        
        if not os.path.exists(json_path):
            print(f"Warning: CLAHE parameters file not found: {json_path}")
            print("Using default CLAHE parameters for all images")
            return params
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Convert JSON data to expected format
            for image_path, param_data in data.items():
                tile_grid_size = param_data['tile_grid_size']
                params[image_path] = {
                    'clip_limit': param_data['clip_limit'],
                    'tile_grid_x': tile_grid_size[0],
                    'tile_grid_y': tile_grid_size[1]
                }
            
        except Exception as e:
            print(f"Error loading CLAHE parameters from {json_path}: {str(e)}")
            print("Using default CLAHE parameters for all images")
            return {}
        
        return params
    
    def create_sample_list(self) -> List[Dict]:
        """Create list of valid samples with labels and CLAHE parameters"""
        samples = []
        
        for _, row in self.labels_df.iterrows():
            image_name = row['image_name']
            
            # Check if image file exists
            image_path = self.images_dir / image_name
            if not image_path.exists():
                continue
            
            # Check if mask exists (optional)
            mask_name = image_name.replace('.png', '_mask.png').replace('.jpg', '_mask.png')
            mask_path = self.masks_dir / mask_name
            
            # Get CLAHE parameters (use defaults if not found)
            relative_path = str(image_path.relative_to(self.images_dir.parent))
            clahe_params = self.clahe_params.get(relative_path, {
                'clip_limit': 3.0,
                'tile_grid_x': 8,
                'tile_grid_y': 8
            })
            
            sample = {
                'image_path': image_path,
                'mask_path': mask_path if mask_path.exists() else None,
                'label': {
                    'damage': int(row.get('damage', 0)),
                    'occlusion': int(row.get('occlusion', 0)),
                    'crop': int(row.get('crop', 0))
                },
                'clahe_params': clahe_params
            }
            samples.append(sample)
        
        return samples
    
    def apply_clahe(self, image: np.ndarray, clahe_params: Dict) -> np.ndarray:
        """Apply CLAHE with specified parameters"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=clahe_params['clip_limit'],
            tileGridSize=(clahe_params['tile_grid_x'], clahe_params['tile_grid_y'])
        )
        enhanced_l = clahe.apply(l_channel)
        
        # Reconstruct image
        lab[:, :, 0] = enhanced_l
        enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr
    
    def apply_mask_overlay(self, image: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
        """Apply mask overlay with specified opacity"""
        if opacity == 0.0:
            return image
        
        # Ensure mask is 3-channel
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Apply overlay with opacity
        overlay = cv2.addWeighted(image, 1.0 - opacity, mask, opacity, 0)
        return overlay
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample['image_path']))
        if image is None:
            raise ValueError(f"Could not load image: {sample['image_path']}")
        
        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Apply CLAHE enhancement
        enhanced_image = self.apply_clahe(image, sample['clahe_params'])
        
        # Load and apply mask if available and opacity > 0
        if sample['mask_path'] and self.mask_opacity > 0.0:
            mask = cv2.imread(str(sample['mask_path']))
            if mask is not None:
                mask = cv2.resize(mask, (self.img_size, self.img_size))
                enhanced_image = self.apply_mask_overlay(enhanced_image, mask, self.mask_opacity)
        
        # Convert to RGB for PyTorch
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL for transforms
        pil_image = Image.fromarray(enhanced_image)
        
        # Apply additional transforms
        if self.transform:
            pil_image = self.transform(pil_image)
        else:
            # Default transform: to tensor and normalize
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pil_image = transform(pil_image)
        
        # Create multi-label target
        target = torch.tensor([
            sample['label']['damage'],
            sample['label']['occlusion'],
            sample['label']['crop']
        ], dtype=torch.float32)
        
        return pil_image, target
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a sample for debugging"""
        sample = self.samples[idx]
        return {
            'image_path': sample['image_path'],
            'mask_path': sample['mask_path'],
            'label': sample['label'],
            'clahe_params': sample['clahe_params']
        }


class CLAHEDataModule:
    """Data module for CLAHE-enhanced datasets"""
    
    def __init__(
        self,
        train_images_dir: str,
        val_images_dir: str,
        test_images_dir: str,
        train_masks_dir: str,
        val_masks_dir: str, 
        test_masks_dir: str,
        train_labels_csv: str,
        val_labels_csv: str,
        test_labels_csv: str,
        clahe_params_json: str,
        mask_opacity: float = 1.0,
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 256
    ):
        self.train_images_dir = train_images_dir
        self.val_images_dir = val_images_dir
        self.test_images_dir = test_images_dir
        self.train_masks_dir = train_masks_dir
        self.val_masks_dir = val_masks_dir
        self.test_masks_dir = test_masks_dir
        self.train_labels_csv = train_labels_csv
        self.val_labels_csv = val_labels_csv
        self.test_labels_csv = test_labels_csv
        self.clahe_params_json = clahe_params_json
        self.mask_opacity = mask_opacity
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        
        # No augmentation transforms (as specified)
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_train_dataset(self) -> CLAHEDataset:
        return CLAHEDataset(
            images_dir=self.train_images_dir,
            masks_dir=self.train_masks_dir,
            labels_file=self.train_labels_csv,
            clahe_params_json=self.clahe_params_json,
            mask_opacity=self.mask_opacity,
            img_size=self.img_size,
            transform=self.base_transform
        )
    
    def get_val_dataset(self) -> CLAHEDataset:
        return CLAHEDataset(
            images_dir=self.val_images_dir,
            masks_dir=self.val_masks_dir,
            labels_file=self.val_labels_csv,
            clahe_params_json=self.clahe_params_json,
            mask_opacity=self.mask_opacity,
            img_size=self.img_size,
            transform=self.base_transform
        )
    
    def get_test_dataset(self) -> CLAHEDataset:
        return CLAHEDataset(
            images_dir=self.test_images_dir,
            masks_dir=self.test_masks_dir,
            labels_file=self.test_labels_csv,
            clahe_params_json=self.clahe_params_json,
            mask_opacity=self.mask_opacity,
            img_size=self.img_size,
            transform=self.base_transform
        )
    
    def get_train_loader(self) -> torch.utils.data.DataLoader:
        dataset = self.get_train_dataset()
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_val_loader(self) -> torch.utils.data.DataLoader:
        dataset = self.get_val_dataset()
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_test_loader(self) -> torch.utils.data.DataLoader:
        dataset = self.get_test_dataset()
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        ) 