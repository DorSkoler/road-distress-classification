import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torchvision.models import EfficientNet_B3_Weights, ResNet50_Weights

class RoadDistressModel(nn.Module):
    """
    Model for road distress classification using a pre-trained backbone
    """
    def __init__(self, num_classes=3, pretrained=True, backbone_type='efficientnet_b3', dropout_rate=0.5):
        """
        Initialize the model
        
        Args:
            num_classes (int): Number of output classes (default: 3 for damage, occlusion, crop)
            pretrained (bool): Whether to use pretrained weights (default: True)
            backbone_type (str): Type of backbone to use ('efficientnet_b3' or 'resnet50')
            dropout_rate (float): Dropout rate for regularization (default: 0.5)
        """
        super(RoadDistressModel, self).__init__()
        
        # Initialize backbone based on type
        if backbone_type == 'efficientnet_b3':
            weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            
            # Enhanced classifier head with batch normalization and residual connections
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, num_classes)
            )
            
            # Replace the original classifier
            self.backbone.classifier = nn.Identity()
        elif backbone_type == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            
            # Enhanced classifier head with batch normalization and residual connections
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, num_classes)
            )
            
            # Replace the original classifier
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # Initialize weights for new layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize the weights of the new layers
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_optimizer(self, learning_rate=1e-4, weight_decay=1e-4):
        """
        Get the optimizer for training
        
        Args:
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay factor
            
        Returns:
            optimizer: The optimizer
        """
        return AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    def get_scheduler(self, optimizer, num_epochs, warmup_epochs=5):
        """
        Get the learning rate scheduler with warmup
        
        Args:
            optimizer: The optimizer
            num_epochs (int): Number of epochs
            warmup_epochs (int): Number of warmup epochs
            
        Returns:
            scheduler: The learning rate scheduler
        """
        # Create warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        # Create cosine scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-6
        )
        
        # Combine schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
        return scheduler
    
    def get_loss_function(self):
        """
        Get the loss function for training
        
        Returns:
            criterion: The loss function
        """
        return nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss for multi-label classification 