#!/usr/bin/env python3
"""
Model B Loader for Road Distress Inference Pipeline
Date: 2025-08-01

This module loads the best performing Model B checkpoint and provides
an interface for inference on arbitrary images.
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class HybridRoadDistressModel(nn.Module):
    """
    Model B architecture: EfficientNet-B3 backbone without masks.
    Optimized for road distress classification with confidence extraction.
    """
    
    def __init__(self, 
                 num_classes: int = 3,
                 dropout_rate: float = 0.5,
                 encoder_name: str = 'efficientnet_b3',
                 encoder_weights: str = 'imagenet'):
        super().__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # EfficientNet-B3 backbone
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
            nn.Dropout(0.3),
            nn.Linear(backbone_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
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
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # Extract features using backbone
        features = self.backbone(x)
        
        # Process features
        processed_features = self.feature_processor(features)
        
        # Classification
        logits = self.classifier(processed_features)
        
        return logits
    
    def forward_with_features(self, x: torch.Tensor) -> tuple:
        """Forward pass returning both logits and intermediate features."""
        # Extract features using backbone
        backbone_features = self.backbone(x)
        
        # Process features
        processed_features = self.feature_processor(backbone_features)
        
        # Classification
        logits = self.classifier(processed_features)
        
        return logits, backbone_features, processed_features


class ModelLoader:
    """Handles loading and managing the Model B checkpoint."""
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Initialize the model loader.
        
        Args:
            checkpoint_path: Path to the best_model.pth checkpoint
            device: Device to load model on ('cuda', 'cpu', or None for auto)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = ['damage', 'occlusion', 'crop']
        
        logger.info(f"ModelLoader initialized:")
        logger.info(f"  - Checkpoint: {self.checkpoint_path}")
        logger.info(f"  - Device: {self.device}")
    
    def load_model(self) -> HybridRoadDistressModel:
        """Load the model from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load checkpoint with weights_only=False for compatibility
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        try:
            # Try with weights_only=True first (secure)
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        except Exception as e:
            logger.warning(f"Failed to load with weights_only=True: {e}")
            logger.info("Falling back to weights_only=False (trusted checkpoint)")
            # Fall back to weights_only=False for older checkpoints
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Create model with same configuration as training
        self.model = HybridRoadDistressModel(
            num_classes=3,
            dropout_rate=0.5,
            encoder_name='efficientnet_b3',
            encoder_weights='imagenet'
        )
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
        logger.info(f"  - Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  - Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        return self.model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
        except Exception:
            # Fall back for older checkpoints
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        info = {
            'class_names': self.class_names,
            'num_classes': 3,
            'device': self.device,
            'checkpoint_path': str(self.checkpoint_path)
        }
        
        # Add training info if available
        if 'epoch' in checkpoint:
            info['epoch'] = checkpoint['epoch']
        if 'best_accuracy' in checkpoint:
            info['best_accuracy'] = checkpoint['best_accuracy']
        if 'best_f1' in checkpoint:
            info['best_f1'] = checkpoint['best_f1']
        
        return info
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions with the loaded model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.sigmoid(logits)  # Multi-label classification
        
        return probabilities
    
    def predict_with_confidence(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions and return detailed confidence information."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            logits, backbone_features, processed_features = self.model.forward_with_features(x)
            probabilities = torch.sigmoid(logits)
            
            # Calculate confidence scores
            confidence_scores = torch.max(torch.stack([probabilities, 1 - probabilities], dim=-1), dim=-1)[0]
            
            return {
                'logits': logits,
                'probabilities': probabilities,
                'confidence': confidence_scores,
                'backbone_features': backbone_features,
                'processed_features': processed_features,
                'predictions': (probabilities > 0.5).float()
            }


def load_best_model_b(experiments_path: str = "../experiments/2025-07-05_hybrid_training") -> ModelLoader:
    """
    Convenience function to load the best Model B checkpoint.
    
    Args:
        experiments_path: Path to the experiments directory
    
    Returns:
        ModelLoader instance with Model B loaded
    """
    checkpoint_path = Path(experiments_path) / "results" / "model_b" / "checkpoints" / "best_model.pth"
    
    loader = ModelLoader(str(checkpoint_path))
    loader.load_model()
    
    return loader