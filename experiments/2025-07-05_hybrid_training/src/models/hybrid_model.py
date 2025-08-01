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
import timm
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
                 encoder_name: str = 'efficientnet_b3',
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
        
        # EfficientNet-B3 backbone for classification (improved from UNet)
        self.backbone = timm.create_model(
            encoder_name,
            pretrained=(encoder_weights == 'imagenet'),
            num_classes=0,  # Remove classification head
            global_pool='',  # Remove global pooling
            drop_rate=0.0   # We'll add our own dropout
        )
        
        # Get feature dimension from backbone
        backbone_features = self.backbone.num_features
        
        # Feature processing layers with regularization
        self.feature_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),  # First dropout layer
            nn.Linear(backbone_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),  # Second dropout layer
        )
        
        # Classification head with better regularization
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # Reduced dropout for final layer
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
        # Get features from EfficientNet backbone
        features = self.backbone(x)  # [B, backbone_features, H, W]
        
        # Apply masking strategy based on model variant
        if self.use_masks and mask is not None:
            masked_features = self._apply_masking(features, mask)
        else:
            masked_features = features
        
        # Process features with regularization
        processed_features = self.feature_processor(masked_features)
        
        # Classification
        logits = self.classifier(processed_features)
        
        return logits
    
    def _apply_masking(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply masking strategy to features.
        
        Args:
            features: Feature maps from backbone [B, backbone_features, H, W]
            mask: Road masks [B, 1, H, W]
            
        Returns:
            Masked features [B, backbone_features, H, W]
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
        # Get backbone features
        backbone_features = self.backbone(x)
        
        # Apply masking if specified
        if self.use_masks and mask is not None:
            masked_features = self._apply_masking(backbone_features, mask)
        else:
            masked_features = backbone_features
        
        # Get processed features
        processed_features = self.feature_processor(masked_features)
        
        # Get classification output
        logits = self.classifier(processed_features)
        
        return {
            'backbone_features': backbone_features,
            'masked_features': masked_features,
            'processed_features': processed_features,
            'logits': logits
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
            'architecture': 'EfficientNet-B3 + Enhanced Classification Head',
            'total_parameters': self.count_parameters(),
            'config': self.config,
            'device': next(self.parameters()).device.type,
            'training_mode': self.training
        }


class ModelVariantFactory:
    """Factory class for creating different model variants with specific configurations."""
    
    @staticmethod
    def create_model_a(num_classes: int = 3, **kwargs) -> HybridRoadDistressModel:
        """
        Create Model A: Pictures + full masks (no augmentation)
        
        This is the baseline model that uses original images with full road masking.
        Masks are applied at full weight (1.0) to focus entirely on road pixels.
        """
        return HybridRoadDistressModel(
            num_classes=num_classes,
            use_masks=True,
            mask_weight=1.0,
            dropout_rate=kwargs.get('dropout_rate', 0.5),
            encoder_name=kwargs.get('encoder_name', 'efficientnet_b3'),
            encoder_weights=kwargs.get('encoder_weights', 'imagenet')
        )
    
    @staticmethod
    def create_model_b(num_classes: int = 3, **kwargs) -> HybridRoadDistressModel:
        """
        Create Model B: Pictures + augmentation (no masks)
        
        This model focuses on learning from augmented images without any masking.
        Helps the model become more robust to various image conditions.
        """
        return HybridRoadDistressModel(
            num_classes=num_classes,
            use_masks=False,
            mask_weight=0.0,
            dropout_rate=kwargs.get('dropout_rate', 0.5),
            encoder_name=kwargs.get('encoder_name', 'efficientnet_b3'),
            encoder_weights=kwargs.get('encoder_weights', 'imagenet')
        )
    
    @staticmethod
    def create_model_c(num_classes: int = 3, **kwargs) -> HybridRoadDistressModel:
        """
        Create Model C: Pictures + augmentation + full masks
        
        Combines data augmentation with full road masking for robust training
        that focuses on road pixels while handling various image conditions.
        """
        return HybridRoadDistressModel(
            num_classes=num_classes,
            use_masks=True,
            mask_weight=1.0,
            dropout_rate=kwargs.get('dropout_rate', 0.5),
            encoder_name=kwargs.get('encoder_name', 'efficientnet_b3'),
            encoder_weights=kwargs.get('encoder_weights', 'imagenet')
        )
    
    @staticmethod
    def create_model_d(num_classes: int = 3, **kwargs) -> HybridRoadDistressModel:
        """
        Create Model D: Pictures + augmentation + weighted masks (50% non-road weight)
        
        Uses partial masking with augmentation to learn from both road and non-road
        pixels while maintaining focus on road surface distress.
        """
        return HybridRoadDistressModel(
            num_classes=num_classes,
            use_masks=True,
            mask_weight=0.5,
            dropout_rate=kwargs.get('dropout_rate', 0.5),
            encoder_name=kwargs.get('encoder_name', 'efficientnet_b3'),
            encoder_weights=kwargs.get('encoder_weights', 'imagenet')
        )
    
    @staticmethod
    def create_model_e(num_classes: int = 3, **kwargs) -> HybridRoadDistressModel:
        """
        Create Model E: CLAHE enhanced images + full masks (no augmentation)
        
        Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing
        with full road masking (1.0 opacity) to enhance contrast and focus on road pixels.
        Optimized for images with optimized CLAHE parameters from JSON.
        """
        return HybridRoadDistressModel(
            num_classes=num_classes,
            use_masks=True,
            mask_weight=1.0,
            dropout_rate=kwargs.get('dropout_rate', 0.5),
            encoder_name=kwargs.get('encoder_name', 'efficientnet_b3'),
            encoder_weights=kwargs.get('encoder_weights', 'imagenet')
        )
    
    @staticmethod
    def create_model_f(num_classes: int = 3, **kwargs) -> HybridRoadDistressModel:
        """
        Create Model F: CLAHE enhanced images + partial masks (no augmentation)
        
        Uses CLAHE preprocessing with partial road masking (0.5 opacity) to learn
        from both road and non-road pixels while maintaining contrast enhancement.
        Optimized for images with optimized CLAHE parameters from JSON.
        """
        return HybridRoadDistressModel(
            num_classes=num_classes,
            use_masks=True,
            mask_weight=0.5,
            dropout_rate=kwargs.get('dropout_rate', 0.5),
            encoder_name=kwargs.get('encoder_name', 'efficientnet_b3'),
            encoder_weights=kwargs.get('encoder_weights', 'imagenet')
        )
    
    @staticmethod
    def create_model_g(num_classes: int = 3, **kwargs) -> HybridRoadDistressModel:
        """
        Create Model G: CLAHE enhanced images + full masks + augmentation
        
        Combines CLAHE preprocessing, full road masking (1.0 opacity), and data
        augmentation for robust training with enhanced contrast and road focus.
        Optimized for images with optimized CLAHE parameters from JSON.
        """
        return HybridRoadDistressModel(
            num_classes=num_classes,
            use_masks=True,
            mask_weight=1.0,
            dropout_rate=kwargs.get('dropout_rate', 0.5),
            encoder_name=kwargs.get('encoder_name', 'efficientnet_b3'),
            encoder_weights=kwargs.get('encoder_weights', 'imagenet')
        )
    
    @staticmethod
    def create_model_h(num_classes: int = 3, **kwargs) -> HybridRoadDistressModel:
        """
        Create Model H: CLAHE enhanced images + partial masks + augmentation
        
        Uses CLAHE preprocessing, partial road masking (0.5 opacity), and data
        augmentation for balanced learning from road and non-road pixels with
        enhanced contrast and robustness.
        Optimized for images with optimized CLAHE parameters from JSON.
        """
        return HybridRoadDistressModel(
            num_classes=num_classes,
            use_masks=True,
            mask_weight=0.5,
            dropout_rate=kwargs.get('dropout_rate', 0.5),
            encoder_name=kwargs.get('encoder_name', 'efficientnet_b3'),
            encoder_weights=kwargs.get('encoder_weights', 'imagenet')
        )
    
    @staticmethod
    def create_model_baseline(num_classes: int = 3, **kwargs) -> HybridRoadDistressModel:
        """
        Create Model Baseline: Pure baseline model
        
        Uses only original images with no enhancements whatsoever:
        - No road masking (0.0 opacity)
        - No data augmentation
        - No CLAHE preprocessing
        
        This serves as the comparison baseline to demonstrate the value
        of all other enhancements.
        """
        return HybridRoadDistressModel(
            num_classes=num_classes,
            use_masks=False,
            mask_weight=0.0,
            dropout_rate=kwargs.get('dropout_rate', 0.5),
            encoder_name=kwargs.get('encoder_name', 'efficientnet_b3'),
            encoder_weights=kwargs.get('encoder_weights', 'imagenet')
        )

    @staticmethod
    def create_variant(variant_name: str, num_classes: int = 3, **kwargs) -> HybridRoadDistressModel:
        """
        Create a model variant by name.
        
        Args:
            variant_name: Name of the variant ('model_a', 'model_b', 'model_c', 'model_d', 
                         'model_e', 'model_f', 'model_g', 'model_h')
            num_classes: Number of output classes
            **kwargs: Additional parameters for model creation
            
        Returns:
            Configured HybridRoadDistressModel instance
        """
        variant_methods = {
            'model_a': ModelVariantFactory.create_model_a,
            'model_b': ModelVariantFactory.create_model_b,
            'model_c': ModelVariantFactory.create_model_c,
            'model_d': ModelVariantFactory.create_model_d,
            'model_e': ModelVariantFactory.create_model_e,
            'model_f': ModelVariantFactory.create_model_f,
            'model_g': ModelVariantFactory.create_model_g,
            'model_h': ModelVariantFactory.create_model_h,
            'model_baseline': ModelVariantFactory.create_model_baseline
        }
        
        if variant_name not in variant_methods:
            raise ValueError(f"Unknown variant: {variant_name}. Available variants: {list(variant_methods.keys())}")
        
        logger.info(f"Creating {variant_name} with {num_classes} classes")
        return variant_methods[variant_name](num_classes=num_classes, **kwargs)


def create_model(variant: str = 'model_a', **kwargs) -> HybridRoadDistressModel:
    """
    Convenience function to create a model variant.
    
    Args:
        variant: Model variant name ('model_a', 'model_b', 'model_c', 'model_d', 
                'model_e', 'model_f', 'model_g', 'model_h')
        **kwargs: Additional model parameters
        
    Returns:
        Configured HybridRoadDistressModel instance
        
    Available Variants:
        - model_a: Pictures + full masks (no augmentation)
        - model_b: Pictures + augmentation (no masks)
        - model_c: Pictures + augmentation + full masks
        - model_d: Pictures + augmentation + partial masks (0.5)
        - model_e: CLAHE enhanced images + full masks (no augmentation)
        - model_f: CLAHE enhanced images + partial masks (no augmentation)
        - model_g: CLAHE enhanced images + full masks + augmentation
        - model_h: CLAHE enhanced images + partial masks + augmentation
        - model_baseline: Pure baseline (no masks, no augmentation, no CLAHE)
    """
    return ModelVariantFactory.create_variant(variant, **kwargs)


# Model variant configurations for easy reference
MODEL_VARIANTS = ['model_a', 'model_b', 'model_c', 'model_d', 'model_e', 'model_f', 'model_g', 'model_h']


if __name__ == "__main__":
    # Test model creation
    print("Testing model variants...")
    
    for variant_name in MODEL_VARIANTS:
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