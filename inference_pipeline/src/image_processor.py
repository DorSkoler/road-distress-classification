#!/usr/bin/env python3
"""
Image Preprocessing Pipeline for Road Distress Inference
Date: 2025-08-01

This module handles preprocessing of arbitrary input images to match
the training format expected by Model B.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Tuple, Union, Optional
import torchvision.transforms as transforms
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Handles preprocessing of input images for Model B inference.
    
    Model B was trained on 256x256 images without CLAHE enhancement
    and without masks, using standard augmentation during training.
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (256, 256),
                 normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        """
        Initialize the image processor.
        
        Args:
            target_size: Target image size (height, width)
            normalize_mean: ImageNet normalization mean
            normalize_std: ImageNet normalization std
        """
        self.target_size = target_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        # Create preprocessing transforms
        self.preprocess_transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
        
        # Transform for visualization (without normalization)
        self.viz_transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        
        logger.info(f"ImageProcessor initialized:")
        logger.info(f"  - Target size: {target_size}")
        logger.info(f"  - Normalization: mean={normalize_mean}, std={normalize_std}")
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image in RGB format
        """
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.info(f"Loaded image: {image_path}")
            logger.info(f"  - Original size: {image.size}")
            logger.info(f"  - Mode: {image.mode}")
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def preprocess_for_model(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            
        Returns:
            Preprocessed tensor ready for model input [1, 3, H, W]
        """
        # Handle different input types
        if isinstance(image, str):
            pil_image = self.load_image(image)
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype == np.uint8:
                pil_image = Image.fromarray(image)
            else:
                # Assume float array in [0, 1] range
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Ensure RGB format
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Apply preprocessing transforms
        tensor = self.preprocess_transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        logger.debug(f"Preprocessed image to tensor shape: {tensor.shape}")
        
        return tensor
    
    def preprocess_for_visualization(self, image: Union[str, Image.Image, np.ndarray]) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocess image for both model inference and visualization.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            
        Returns:
            Tuple of (model_tensor, resized_pil_image)
        """
        # Handle different input types
        if isinstance(image, str):
            pil_image = self.load_image(image)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Ensure RGB format
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Resize for visualization
        resized_image = pil_image.resize(self.target_size, Image.Resampling.BILINEAR)
        
        # Create model input tensor
        model_tensor = self.preprocess_transform(pil_image).unsqueeze(0)
        
        return model_tensor, resized_image
    
    def denormalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a tensor back to [0, 1] range for visualization.
        
        Args:
            tensor: Normalized tensor [C, H, W] or [B, C, H, W]
            
        Returns:
            Denormalized tensor in [0, 1] range
        """
        mean = torch.tensor(self.normalize_mean).view(-1, 1, 1)
        std = torch.tensor(self.normalize_std).view(-1, 1, 1)
        
        if tensor.dim() == 4:  # Batch dimension
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        
        # Denormalize
        denormalized = tensor * std + mean
        
        # Clamp to [0, 1] range
        denormalized = torch.clamp(denormalized, 0, 1)
        
        return denormalized
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert a tensor to PIL Image.
        
        Args:
            tensor: Tensor [C, H, W] or [B, C, H, W]
            
        Returns:
            PIL Image
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Denormalize if needed
        if tensor.min() < 0:  # Likely normalized
            tensor = self.denormalize_tensor(tensor)
        
        # Convert to numpy
        numpy_array = tensor.permute(1, 2, 0).cpu().numpy()
        numpy_array = (numpy_array * 255).astype(np.uint8)
        
        return Image.fromarray(numpy_array)
    
    def get_original_size_info(self, image: Union[str, Image.Image, np.ndarray]) -> Tuple[int, int]:
        """
        Get original image size information.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (width, height)
        """
        if isinstance(image, str):
            pil_image = self.load_image(image)
            return pil_image.size
        elif isinstance(image, np.ndarray):
            return (image.shape[1], image.shape[0])  # (width, height)
        elif isinstance(image, Image.Image):
            return image.size
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def resize_to_original(self, processed_result: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize processed result back to original image size.
        
        Args:
            processed_result: Processed result array (e.g., heatmap)
            original_size: Original image size (width, height)
            
        Returns:
            Resized result array
        """
        if len(processed_result.shape) == 3:  # Color image
            resized = cv2.resize(processed_result, original_size, interpolation=cv2.INTER_LINEAR)
        else:  # Grayscale
            resized = cv2.resize(processed_result, original_size, interpolation=cv2.INTER_LINEAR)
        
        return resized