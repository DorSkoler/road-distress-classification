"""
Road Mask Generator for Inference Pipeline
Generates road segmentation masks for input images
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
import logging
from pathlib import Path
from PIL import Image
from typing import Union, Optional
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


class RoadMaskGenerator:
    """Road mask generator using U-Net segmentation model."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the road mask generator.
        
        Args:
            model_path: Path to the segmentation model checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        
        # Initialize model
        self._load_model()
        
        logger.info(f"RoadMaskGenerator initialized on {self.device}")
    
    def is_available(self) -> bool:
        """Check if the mask generator is available for use."""
        return self.model is not None
        
    def _load_model(self) -> nn.Module:
        """Load the road segmentation model."""
        try:
            # Create U-Net model with ResNet34 backbone
            model = smp.Unet(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                classes=1,
                activation=None
            )
            
            # Load weights if path is provided and exists
            if self.model_path and Path(self.model_path).exists():
                try:
                    # Load exactly as in test_inference.py
                    model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
                    logger.info(f"✓ Loaded mask generation model from {self.model_path}")
                except Exception as e:
                    logger.warning(f"Could not load model weights: {e}")
                    logger.warning("Using model with ImageNet weights only")
            else:
                logger.info("No model checkpoint provided - using ImageNet pretrained weights")
                
            model.to(self.device)
            model.eval()
            self.model = model
            
        except Exception as e:
            logger.error(f"Failed to load mask model: {e}")
            self.model = None
            
    def generate_mask(self, image: Union[np.ndarray, Image.Image], confidence_threshold: float = 0.3) -> Optional[np.ndarray]:
        """
        Generate road mask for given image using the same preprocessing as experiments.
        
        Args:
            image: Input image as numpy array (H, W, C) or PIL Image
            confidence_threshold: Threshold for binary mask generation
            
        Returns:
            Binary road mask as numpy array (H, W) or None if generation fails
        """
        if self.model is None:
            logger.warning("No mask model available - returning None")
            return None
            
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
                
            # Ensure RGB format
            if image.shape[-1] == 4:  # RGBA
                image = image[:, :, :3]
            elif len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[-1] == 1:  # Single channel
                image = np.repeat(image, 3, axis=2)
                
            # Store original size for later resizing
            original_size = (image.shape[0], image.shape[1])
            
            # Preprocess image following experiments pattern
            image_tensor = self._preprocess_image(image)
            
            # Generate mask
            mask_binary = self._generate_mask_tensor(image_tensor, confidence_threshold)
            
            # Resize mask back to original size (keeping 0-255 range from test_inference.py)
            if mask_binary.shape != original_size:
                mask_resized = cv2.resize(mask_binary, (original_size[1], original_size[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                mask_final = mask_resized
            else:
                mask_final = mask_binary
            
            return mask_final
            
        except Exception as e:
            logger.error(f"Error generating mask: {e}")
            return None
            
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image exactly as in test_inference.py."""
        # Convert to PIL Image first (as in test_inference.py)
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert('RGB')
        else:
            pil_image = image.convert('RGB')
        
        # Resize to 256x256 (must match training size as in test_inference.py)
        image_resized = pil_image.resize((256, 256))
        
        # Convert to tensor exactly as in test_inference.py
        img_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        return img_tensor.to(self.device)
    
    def _generate_mask_tensor(self, image_tensor: torch.Tensor, confidence_threshold: float) -> np.ndarray:
        """Generate mask tensor exactly as in test_inference.py."""
        with torch.no_grad():
            # Forward pass (exactly as test_inference.py)
            pred = self.model(image_tensor)
            
            # Apply sigmoid and convert to numpy exactly as test_inference.py
            pred_mask = torch.sigmoid(pred).cpu().numpy()[0, 0]
            
            logger.debug(f"Raw prediction range: {pred_mask.min()} to {pred_mask.max()}")
            
            # Apply threshold and convert to 0-255 range exactly as test_inference.py
            pred_mask_binary = (pred_mask > confidence_threshold).astype(np.uint8) * 255
            
            logger.debug(f"After threshold and *255: {pred_mask_binary.min()} to {pred_mask_binary.max()}")
            logger.debug(f"Final mask dtype: {pred_mask_binary.dtype}")
            
            return pred_mask_binary
            
    def create_mask_overlay(self, image: np.ndarray, mask: np.ndarray, opacity: float = 0.5) -> np.ndarray:
        """
        Create visualization overlay of mask on original image (same as preprocessing annotations).
        
        Args:
            image: Original image (H, W, C)
            mask: Binary mask (H, W) with values 0-255 (from test_inference.py format)
            opacity: Overlay opacity (0.0-1.0)
            
        Returns:
            Image with mask overlay (same style as preprocessing annotations)
        """
        try:
            logger.debug(f"Creating mask overlay: image shape {image.shape}, mask shape {mask.shape}")
            logger.debug(f"Mask range: {mask.min()} to {mask.max()}")
            
            if opacity == 0.0:
                return image.copy()
            
            # Resize mask to match image if needed
            if mask.shape != image.shape[:2]:
                logger.info(f"Resizing mask from {mask.shape} to {image.shape[:2]}")
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask
            
            # Handle mask in 0-255 range (from test_inference.py)
            if len(mask_resized.shape) == 2:
                # Create colored mask (green for roads, same as annotations)
                colored_mask = np.zeros_like(image)
                colored_mask[:, :, 1] = mask_resized  # Green channel, mask already 0-255
            else:
                colored_mask = mask_resized.copy()
            
            logger.debug(f"Colored mask range: {colored_mask.min()} to {colored_mask.max()}")
            
            # Apply overlay with opacity using cv2.addWeighted (same as experiments)
            overlay = cv2.addWeighted(image, 1.0 - opacity, colored_mask, opacity, 0)
            
            logger.info(f"✅ Successfully created mask overlay with opacity {opacity}")
            
            return overlay
            
        except Exception as e:
            logger.error(f"❌ Error creating mask overlay: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return image
            
    def get_mask_statistics(self, mask: np.ndarray) -> dict:
        """
        Calculate statistics for the generated mask.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Dictionary with mask statistics
        """
        if mask is None:
            return {"road_coverage": 0.0, "total_pixels": 0, "road_pixels": 0}
            
        total_pixels = mask.shape[0] * mask.shape[1]
        road_pixels = np.sum(mask > 0)
        road_coverage = road_pixels / total_pixels
        
        return {
            "road_coverage": float(road_coverage),
            "total_pixels": int(total_pixels),
            "road_pixels": int(road_pixels)
        }
        
    def is_available(self) -> bool:
        """Check if mask generation is available."""
        return self.model is not None
