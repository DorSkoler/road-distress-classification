#!/usr/bin/env python3
"""
Hybrid Road Distress Classification Model
Date: 2025-07-05

This module implements the hybrid model architecture combining:
- Successful UNet + EfficientNet-B3 architecture from 2025-05-10 experiment
- Cross-platform compatibility for Windows/Mac/Linux
- Support for 4 model variants with different input strategies

Architecture adapted from the successful 88.99% accuracy model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class HybridRoadDistressModel(nn.Module):
    """
    Hybrid road distress classification model with mask support.
    
    Based on the successful UNet + EfficientNet-B3 architecture that achieved
    88.99% overall accuracy in the 2025-05-10 experiment.
    
    Supports 4 different training variants:
    - Model A: Pictures + full masks
    - Model B: Pictures + augmentation (no masks)
    - Model C: Pictures + augmentation + full masks
    - Model D: Pictures + augmentation + weighted masks (50% non-road weight)
    """
    
    def __init__(self, 
                 num_classes: int = 3,
                 use_masks: bool = True,
                 mask_weight: float = 1.0,
                 dropout_rate: float = 0.5,
                 encoder_name: str = 'efficientnet-b3',
                 encoder_weights: str = 'imagenet'):
        """
        Initialize the hybrid model.
        
        Args:
            num_classes: Number of output classes (default: 3 for damage, occlusion, crop)
            use_masks: Whether to use mask-based training
            mask_weight: Weight for non-road pixels (1.0 = full masking, 0.5 = weighted masking)
            dropout_rate: Dropout rate for classification head
            encoder_name: Encoder backbone name
            encoder_weights: Pre-trained weights for encoder
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_masks = use_masks
        self.mask_weight = mask_weight
        self.dropout_rate = dropout_rate
        
        # Store model configuration
        self.config = {
            'num_classes': num_classes,
            'use_masks': use_masks,
            'mask_weight': mask_weight,
            'dropout_rate': dropout_rate,
            'encoder_name': encoder_name,
            'encoder_weights': encoder_weights
        }
        
        # UNet backbone with EfficientNet-B3 encoder (from successful 10/05 experiment)
        self.backbone = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None  # No activation for raw logits
        )
        
        # Classification head (from successful 10/05 experiment)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized HybridRoadDistressModel:")
        logger.info(f"  - Classes: {num_classes}")
        logger.info(f"  - Use masks: {use_masks}")
        logger.info(f"  - Mask weight: {mask_weight}")
        logger.info(f"  - Encoder: {encoder_name}")
        logger.info(f"  - Parameters: {self.count_parameters():,}")
    
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input images [B, C, H, W]
            mask: Optional road masks [B, 1, H, W] (values 0-1, 1=road)
            
        Returns:
            Classification logits [B, num_classes]
        """
        # Get features from UNet backbone
        features = self.backbone(x)  # [B, num_classes, H, W]
        
        # Apply masking strategy based on model variant
        if self.use_masks and mask is not None:
            masked_features = self._apply_masking(features, mask)
        else:
            masked_features = features
        
        # Classification
        logits = self.classifier(masked_features)
        
        return logits
    
    def _apply_masking(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply masking strategy to features.
        
        Args:
            features: Feature maps from backbone [B, num_classes, H, W]
            mask: Road masks [B, 1, H, W]
            
        Returns:
            Masked features [B, num_classes, H, W]
        """
        # Ensure mask has same spatial dimensions as features
        if mask.shape[-2:] != features.shape[-2:]:
            mask = F.interpolate(mask, size=features.shape[-2:], mode='bilinear', align_corners=False)
        
        # Expand mask to match feature channels
        mask_expanded = mask.expand_as(features)
        
        if self.mask_weight == 1.0:
            # Full masking (Models A & C): zero out non-road pixels
            masked_features = features * mask_expanded
        else:
            # Weighted masking (Model D): reduce non-road pixel contribution
            # Road pixels: weight = 1.0, Non-road pixels: weight = mask_weight
            weighted_mask = mask_expanded + (1 - mask_expanded) * self.mask_weight
            masked_features = features * weighted_mask
        
        return masked_features
    
    def get_feature_maps(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature maps for visualization.
        
        Args:
            x: Input images [B, C, H, W]
            mask: Optional road masks [B, 1, H, W]
            
        Returns:
            Dictionary of feature maps
        """
        # Get encoder features
        encoder_features = self.backbone.encoder(x)
        
        # Get decoder features
        decoder_features = self.backbone.decoder(*encoder_features)
        
        # Get segmentation head output
        segmentation_head = self.backbone.segmentation_head(decoder_features)
        
        # Apply masking if specified
        if self.use_masks and mask is not None:
            masked_features = self._apply_masking(segmentation_head, mask)
        else:
            masked_features = segmentation_head
        
        # Get classification features
        pooled_features = F.adaptive_avg_pool2d(masked_features, 1)
        flattened_features = torch.flatten(pooled_features, 1)
        
        return {
            'encoder_features': encoder_features,
            'decoder_features': decoder_features,
            'segmentation_features': segmentation_head,
            'masked_features': masked_features,
            'pooled_features': pooled_features,
            'flattened_features': flattened_features
        }
    
    def predict(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores.
        
        Args:
            x: Input images [B, C, H, W]
            mask: Optional road masks [B, 1, H, W]
            threshold: Decision threshold for binary classification
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.eval()
        
        with torch.no_grad():
            logits = self.forward(x, mask)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()
        
        return predictions, probabilities
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging and debugging."""
        return {
            'model_name': 'HybridRoadDistressModel',
            'architecture': 'UNet + EfficientNet-B3',
            'total_parameters': self.count_parameters(),
            'config': self.config,
            'device': next(self.parameters()).device.type,
            'training_mode': self.training
        }


class ModelVariantFactory:
    """Factory class for creating different model variants."""
    
    @staticmethod
    def create_model_a(num_classes: int = 3, **kwargs) -> HybridRoadDistressModel:
        """
        Create Model A: Pictures + full masks.
        
        Args:
            num_classes: Number of output classes
            **kwargs: Additional model parameters
            
        Returns:
            Model A instance
        """
        return HybridRoadDistressModel(
            num_classes=num_classes,
            use_masks=True,
            mask_weight=1.0,
            **kwargs
        )
    
    @staticmethod
    def create_model_b(num_classes: int = 3, **kwargs) -> HybridRoadDistressModel:
        """
        Create Model B: Pictures + augmentation (no masks).
        
        Args:
            num_classes: Number of output classes
            **kwargs: Additional model parameters
            
        Returns:
            Model B instance
        """
        return HybridRoadDistressModel(
            num_classes=num_classes,
            use_masks=False,
            mask_weight=0.0,
            **kwargs
        )
    
    @staticmethod
    def create_model_c(num_classes: int = 3, **kwargs) -> HybridRoadDistressModel:
        """
        Create Model C: Pictures + augmentation + full masks.
        
        Args:
            num_classes: Number of output classes
            **kwargs: Additional model parameters
            
        Returns:
            Model C instance
        """
        return HybridRoadDistressModel(
            num_classes=num_classes,
            use_masks=True,
            mask_weight=1.0,
            **kwargs
        )
    
    @staticmethod
    def create_model_d(num_classes: int = 3, **kwargs) -> HybridRoadDistressModel:
        """
        Create Model D: Pictures + augmentation + weighted masks (50% non-road weight).
        
        Args:
            num_classes: Number of output classes
            **kwargs: Additional model parameters
            
        Returns:
            Model D instance
        """
        return HybridRoadDistressModel(
            num_classes=num_classes,
            use_masks=True,
            mask_weight=0.5,  # 50% weight for non-road pixels
            **kwargs
        )
    
    @staticmethod
    def create_variant(variant_name: str, num_classes: int = 3, **kwargs) -> HybridRoadDistressModel:
        """
        Create model variant by name.
        
        Args:
            variant_name: Name of the variant ('model_a', 'model_b', 'model_c', 'model_d')
            num_classes: Number of output classes
            **kwargs: Additional model parameters
            
        Returns:
            Model instance for the specified variant
        """
        variant_creators = {
            'model_a': ModelVariantFactory.create_model_a,
            'model_b': ModelVariantFactory.create_model_b,
            'model_c': ModelVariantFactory.create_model_c,
            'model_d': ModelVariantFactory.create_model_d
        }
        
        if variant_name not in variant_creators:
            raise ValueError(f"Unknown variant: {variant_name}. Available: {list(variant_creators.keys())}")
        
        return variant_creators[variant_name](num_classes, **kwargs)


def create_model(variant: str = 'model_a', **kwargs) -> HybridRoadDistressModel:
    """
    Convenience function to create a model variant.
    
    Args:
        variant: Model variant name
        **kwargs: Additional model parameters
        
    Returns:
        Model instance
    """
    return ModelVariantFactory.create_variant(variant, **kwargs)


# Model variant configurations for easy reference
MODEL_VARIANTS = {
    'model_a': {
        'name': 'Model A',
        'description': 'Pictures + full masks',
        'use_masks': True,
        'mask_weight': 1.0,
        'use_augmentation': False
    },
    'model_b': {
        'name': 'Model B', 
        'description': 'Pictures + augmentation (no masks)',
        'use_masks': False,
        'mask_weight': 0.0,
        'use_augmentation': True
    },
    'model_c': {
        'name': 'Model C',
        'description': 'Pictures + augmentation + full masks',
        'use_masks': True,
        'mask_weight': 1.0,
        'use_augmentation': True
    },
    'model_d': {
        'name': 'Model D',
        'description': 'Pictures + augmentation + weighted masks (50% non-road)',
        'use_masks': True,
        'mask_weight': 0.5,
        'use_augmentation': True
    }
}


if __name__ == "__main__":
    # Test model creation
    print("Testing model variants...")
    
    for variant_name in MODEL_VARIANTS.keys():
        print(f"\n{variant_name.upper()}:")
        model = create_model(variant_name)
        info = model.get_model_info()
        print(f"  - Parameters: {info['total_parameters']:,}")
        print(f"  - Use masks: {info['config']['use_masks']}")
        print(f"  - Mask weight: {info['config']['mask_weight']}")
        
        # Test forward pass
        x = torch.randn(2, 3, 256, 256)
        mask = torch.ones(2, 1, 256, 256)
        
        if info['config']['use_masks']:
            output = model(x, mask)
        else:
            output = model(x)
        
        print(f"  - Output shape: {output.shape}")
        print(f"  - ✓ Forward pass successful")
    
    print("\n✓ All model variants created and tested successfully!") 