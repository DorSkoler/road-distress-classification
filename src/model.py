import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

class RoadDistressModel(nn.Module):
    """
    Model for road distress classification using a pre-trained backbone
    """
    def __init__(self, num_classes=3, pretrained=True, backbone_type='efficientnet_b3'):
        """
        Initialize the model
        
        Args:
            num_classes (int): Number of output classes (default: 3 for damage, occlusion, crop)
            pretrained (bool): Whether to use pretrained weights (default: True)
            backbone_type (str): Type of backbone to use ('efficientnet_b3' or 'resnet50')
        """
        super(RoadDistressModel, self).__init__()
        
        # Initialize backbone based on type
        if backbone_type == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            # Replace classifier
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        elif backbone_type == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            # Replace classifier
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
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
    
    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
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
    
    def get_scheduler(self, optimizer, num_epochs):
        """
        Get the learning rate scheduler
        
        Args:
            optimizer: The optimizer
            num_epochs (int): Number of epochs
            
        Returns:
            scheduler: The learning rate scheduler
        """
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_epochs // 3,  # Restart every 1/3 of total epochs
            T_mult=2,  # Double the restart interval after each restart
            eta_min=1e-6  # Minimum learning rate
        )
    
    def get_loss_function(self):
        """
        Get the loss function for training
        
        Returns:
            criterion: The loss function
        """
        return nn.CrossEntropyLoss() 