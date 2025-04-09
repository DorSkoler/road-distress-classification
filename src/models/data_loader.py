import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RoadDistressDataset(Dataset):
    def __init__(self, 
                 images: List[np.ndarray],
                 labels: List[Dict],
                 transform: A.Compose = None,
                 is_training: bool = True):
        """
        Dataset class for road distress classification.
        
        Args:
            images: List of images (numpy arrays)
            labels: List of label dictionaries
            transform: Albumentations transform pipeline
            is_training: Whether this is for training (affects augmentation)
        """
        self.images = images
        self.labels = labels
        self.is_training = is_training
        
        # Default transforms if none provided
        if transform is None:
            if is_training:
                self.transform = A.Compose([
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(p=0.5),
                    A.GaussNoise(p=0.3),
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([
                    A.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
        else:
            self.transform = transform
            
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert damage label to binary (0: Not Damaged, 1: Damaged)
        damage_label = 1 if label['damage'] == 'Damaged' else 0
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
            
        return image, torch.tensor(damage_label, dtype=torch.long)

def create_data_loaders(loader: 'OrganizedDatasetLoader',
                       batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        loader: OrganizedDatasetLoader instance
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load data for each split
    train_images, train_labels = loader.load_split('train')
    val_images, val_labels = loader.load_split('val')
    test_images, test_labels = loader.load_split('test')
    
    # Create datasets
    train_dataset = RoadDistressDataset(
        images=train_images,
        labels=train_labels,
        is_training=True
    )
    
    val_dataset = RoadDistressDataset(
        images=val_images,
        labels=val_labels,
        is_training=False
    )
    
    test_dataset = RoadDistressDataset(
        images=test_images,
        labels=test_labels,
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 