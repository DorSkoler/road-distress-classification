import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Optional, Tuple, Dict, Any
import yaml
import os


class AttentionFusion(nn.Module):
    """Attention-based fusion module for combining image and mask features."""
    
    def __init__(self, image_dim: int, mask_dim: int, fusion_dim: int = 512):
        super().__init__()
        self.image_dim = image_dim
        self.mask_dim = mask_dim
        self.fusion_dim = fusion_dim
        
        # Attention mechanism
        self.image_attention = nn.Sequential(
            nn.Linear(image_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )
        
        self.mask_attention = nn.Sequential(
            nn.Linear(mask_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(image_dim + mask_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(self, image_features: torch.Tensor, mask_features: torch.Tensor) -> torch.Tensor:
        # Calculate attention weights
        image_attn = self.image_attention(image_features)
        mask_attn = self.mask_attention(mask_features)
        
        # Normalize attention weights
        total_attn = image_attn + mask_attn + 1e-8
        image_attn = image_attn / total_attn
        mask_attn = mask_attn / total_attn
        
        # Apply attention
        attended_image = image_features * image_attn
        attended_mask = mask_features * mask_attn
        
        # Concatenate and fuse
        combined = torch.cat([attended_image, attended_mask], dim=1)
        fused = self.fusion_layer(combined)
        
        return fused


class SimpleMaskEncoder(nn.Module):
    """Simple CNN encoder for road masks."""
    
    def __init__(self, input_channels: int = 1, output_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            # 512x512 -> 256x256
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 256x256 -> 128x128
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 128x128 -> 64x64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 64x64 -> 32x32
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DualInputRoadDistressClassifier(nn.Module):
    """Dual-input classifier for road distress detection with optional mask input."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Configuration
        self.config = config
        self.num_classes = config.get('dataset', {}).get('num_classes', 3)  # Fallback to 3 classes
        self.use_masks = config.get('model', {}).get('dual_input', {}).get('enabled', False)
        self.mask_fusion = config.get('model', {}).get('dual_input', {}).get('mask_fusion', 'attention')
        self.mask_encoder_type = config.get('model', {}).get('dual_input', {}).get('mask_encoder', 'simple')
        
        # Backbone for image features
        backbone_name = config.get('model', {}).get('architecture', {}).get('backbone', 'resnet50')
        self.pretrained = config.get('model', {}).get('architecture', {}).get('pretrained', True)
        
        if backbone_name == 'efficientnet_b3':
            self.backbone = timm.create_model('efficientnet_b3', 
                                            pretrained=self.pretrained,
                                            num_classes=0)  # Remove classifier
            self.image_dim = 1536  # EfficientNet-B3 feature dimension
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=self.pretrained)
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.image_dim = 2048  # ResNet-50 feature dimension
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Freeze backbone if specified
        if config.get('model', {}).get('architecture', {}).get('freeze_backbone', False):
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Mask encoder (if using masks)
        if self.use_masks:
            if self.mask_encoder_type == 'simple':
                self.mask_encoder = SimpleMaskEncoder(input_channels=1, output_dim=256)
                self.mask_dim = 256
            else:
                raise ValueError(f"Unsupported mask encoder: {self.mask_encoder_type}")
        
        # Fusion mechanism
        if self.use_masks:
            if self.mask_fusion == 'attention':
                self.fusion = AttentionFusion(self.image_dim, self.mask_dim)
                fusion_output_dim = 512
            elif self.mask_fusion == 'concatenation':
                self.fusion = None
                fusion_output_dim = self.image_dim + self.mask_dim
            else:
                raise ValueError(f"Unsupported fusion method: {self.mask_fusion}")
        else:
            fusion_output_dim = self.image_dim
        
        # Classification head - Multi-label binary classification
        classification_config = config.get('model', {}).get('classification', {})
        hidden_size = classification_config.get('hidden_size', 512)
        dropout = classification_config.get('dropout', 0.3)
        activation = classification_config.get('activation', 'relu')
        
        # Three independent binary classifiers for: damaged, occluded, cropped
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, hidden_size),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3)  # 3 independent binary outputs
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            images: Input images [batch_size, 3, height, width]
            masks: Optional road masks [batch_size, 1, height, width]
        
        Returns:
            Classification logits [batch_size, num_classes]
        """
        # Extract image features
        image_features = self.backbone(images)
        if isinstance(image_features, tuple):
            image_features = image_features[0]
        
        # Flatten image features
        image_features = image_features.view(image_features.size(0), -1)
        
        if self.use_masks and masks is not None:
            # Extract mask features
            mask_features = self.mask_encoder(masks)
            
            # Fuse features
            if self.mask_fusion == 'attention':
                fused_features = self.fusion(image_features, mask_features)
            elif self.mask_fusion == 'concatenation':
                fused_features = torch.cat([image_features, mask_features], dim=1)
            else:
                raise ValueError(f"Unsupported fusion method: {self.mask_fusion}")
        else:
            # Use only image features
            fused_features = image_features
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_feature_maps(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Extract intermediate feature maps for visualization."""
        features = {}
        
        # Image features
        image_features = self.backbone(images)
        if isinstance(image_features, tuple):
            image_features = image_features[0]
        features['image_features'] = image_features.view(image_features.size(0), -1)
        
        if self.use_masks and masks is not None:
            # Mask features
            mask_features = self.mask_encoder(masks)
            features['mask_features'] = mask_features
            
            # Fused features
            if self.mask_fusion == 'attention':
                fused_features = self.fusion(features['image_features'], mask_features)
            elif self.mask_fusion == 'concatenation':
                fused_features = torch.cat([features['image_features'], mask_features], dim=1)
            features['fused_features'] = fused_features
        
        return features


class RoadDistressDataset(torch.utils.data.Dataset):
    """Dataset for road distress classification with optional mask support."""
    
    def __init__(self, 
                 image_paths: list,
                 labels: list,
                 mask_paths: Optional[list] = None,
                 transform=None,
                 mask_transform=None,
                 use_masks: bool = True):
        self.image_paths = image_paths
        self.labels = labels
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform
        self.use_masks = use_masks
        
        # Validate inputs
        if self.use_masks and self.mask_paths is None:
            raise ValueError("Mask paths required when use_masks=True")
        if self.use_masks and len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Number of image and mask paths must match")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = torch.load(image_path) if image_path.endswith('.pt') else self._load_image(image_path)
        
        # Load mask if required
        mask = None
        if self.use_masks and self.mask_paths is not None:
            mask_path = self.mask_paths[idx]
            mask = torch.load(mask_path) if mask_path.endswith('.pt') else self._load_mask(mask_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.mask_transform and mask is not None:
            mask = self.mask_transform(mask)
        
        # Get label and convert to tensor
        label = self.labels[idx]
        if isinstance(label, list):
            label = torch.tensor(label, dtype=torch.float32)
        elif not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float32)
        
        if self.use_masks and mask is not None:
            return image, mask, label
        else:
            return image, label
    
    def _load_image(self, path):
        """Load image from path."""
        import cv2
        import numpy as np
        
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not load image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to target size
        target_size = (512, 512)  # Default size, can be made configurable
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image
    
    def _load_mask(self, path):
        """Load mask from path."""
        import cv2
        import numpy as np
        
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {path}")
        
        # Resize mask to target size
        target_size = (512, 512)  # Default size, can be made configurable
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        return mask


def create_model_variant(config: Dict[str, Any], variant_name: str) -> DualInputRoadDistressClassifier:
    """Create a model variant based on configuration."""
    
    # Create a copy of config and modify for this variant
    variant_config = yaml.safe_load(yaml.dump(config))  # Deep copy
    
    # Ensure comparative_training section exists
    if 'comparative_training' not in variant_config:
        variant_config['comparative_training'] = {
            'variants': {
                'model_a': {
                    'name': 'original_only',
                    'use_masks': False,
                    'use_augmentation': False,
                    'description': 'Original images only'
                },
                'model_b': {
                    'name': 'with_masks',
                    'use_masks': True,
                    'use_augmentation': False,
                    'description': 'Original images + road masks'
                },
                'model_c': {
                    'name': 'with_augmentation',
                    'use_masks': False,
                    'use_augmentation': True,
                    'description': 'Original + augmented images'
                },
                'model_d': {
                    'name': 'full_pipeline',
                    'use_masks': True,
                    'use_augmentation': True,
                    'description': 'Original + augmented + masks'
                }
            }
        }
    
    # Get variant settings
    variant_settings = variant_config['comparative_training']['variants'][variant_name]
    
    # Ensure model section exists
    if 'model' not in variant_config:
        variant_config['model'] = {}
    
    # Ensure dual_input section exists
    if 'dual_input' not in variant_config['model']:
        variant_config['model']['dual_input'] = {}
    
    # Update model configuration based on variant
    variant_config['model']['dual_input']['enabled'] = variant_settings['use_masks']
    
    # Create model
    model = DualInputRoadDistressClassifier(variant_config)
    
    return model


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Get a summary of model parameters and architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture': model.__class__.__name__
    }
    
    return summary


if __name__ == "__main__":
    # Test the model
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model variants
    variants = ['model_a', 'model_b', 'model_c', 'model_d']
    
    for variant in variants:
        print(f"\n=== {variant.upper()} ===")
        model = create_model_variant(config, variant)
        summary = get_model_summary(model)
        
        print(f"Total parameters: {summary['total_parameters']:,}")
        print(f"Trainable parameters: {summary['trainable_parameters']:,}")
        print(f"Model size: {summary['model_size_mb']:.2f} MB")
        
        # Test forward pass
        batch_size = 2
        image_size = config.get('dataset', {}).get('image_size', [512, 512])
        
        images = torch.randn(batch_size, 3, image_size[0], image_size[1])
        
        # Get variant settings
        variant_settings = config.get('comparative_training', {}).get('variants', {}).get(variant, {})
        use_masks = variant_settings.get('use_masks', False)
        
        if use_masks:
            masks = torch.randn(batch_size, 1, image_size[0], image_size[1])
            outputs = model(images, masks)
        else:
            outputs = model(images)
        
        print(f"Output shape: {outputs.shape}")
        print(f"Expected shape: [{batch_size}, {config.get('dataset', {}).get('num_classes', 3)}]") 