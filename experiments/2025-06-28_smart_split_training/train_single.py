import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import json
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from dual_input_model import (
    DualInputRoadDistressClassifier, 
    RoadDistressDataset, 
    create_model_variant,
    get_model_summary
)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_single_model(variant_name: str = 'model_a', epochs: int = 5):
    """Train a single model variant for testing."""
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    torch.manual_seed(config.get('system', {}).get('random_seed', 42))
    np.random.seed(config.get('system', {}).get('random_seed', 42))
    
    print(f"Training {variant_name} for {epochs} epochs...")
    
    # Create model
    model = create_model_variant(config, variant_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model created on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    # Load a small subset of data for testing
    try:
        # Load split information
        splits_dir = 'splits'
        with open(os.path.join(splits_dir, 'train_images.txt'), 'r') as f:
            train_images = [line.strip() for line in f.readlines()][:100]  # Just 100 images for testing
        with open(os.path.join(splits_dir, 'val_images.txt'), 'r') as f:
            val_images = [line.strip() for line in f.readlines()][:50]  # Just 50 images for testing
        
        # Load labels
        labels_map = {'damaged': 0, 'occlusion': 1, 'cropped': 2}
        
        def load_labels(image_list):
            labels = []
            for img_path in image_list:
                # Extract label from path (assuming structure: .../label/image.jpg)
                parts = img_path.split('/')
                label_name = parts[-2] if len(parts) > 1 else 'damaged'  # fallback
                labels.append(labels_map.get(label_name, 0))
            return labels
        
        train_labels = load_labels(train_images)
        val_labels = load_labels(val_images)
        
        # Construct full paths to actual image files
        coryell_path = config.get('dataset', {}).get('coryell_path', '../../data/coryell')
        
        def construct_image_paths(image_list):
            full_paths = []
            for img_path in image_list:
                # img_path format: "Co Rd 232/018_31.615684_-97.742088"
                # Need to construct: "../../data/coryell/Co Rd 232/img/018_31.615684_-97.742088.png"
                road_name = img_path.split('/')[0]  # "Co Rd 232"
                img_name = img_path.split('/')[1]   # "018_31.615684_-97.742088"
                full_path = os.path.join(coryell_path, road_name, 'img', f"{img_name}.png")
                full_paths.append(full_path)
            return full_paths
        
        train_images = construct_image_paths(train_images)
        val_images = construct_image_paths(val_images)
        
        # Setup mask paths if using masks
        train_masks = None
        val_masks = None
        
        variant_config = config.get('comparative_training', {}).get('variants', {}).get(variant_name, {})
        use_masks = variant_config.get('use_masks', False)
        
        if use_masks:
            masks_dir = 'masks'
            
            def construct_mask_paths(image_list, split_name):
                mask_paths = []
                for img_path in image_list:
                    # img_path format: "../../data/coryell/Co Rd 232/img/018_31.615684_-97.742088.png"
                    # Need to construct: "masks/train/Co Rd 232/018_31.615684_-97.742088.png"
                    parts = img_path.split('/')
                    road_name = parts[-3]  # "Co Rd 232"
                    img_name = parts[-1].replace('.png', '')  # "018_31.615684_-97.742088"
                    mask_path = os.path.join(masks_dir, split_name, road_name, f"{img_name}.png")
                    mask_paths.append(mask_path)
                return mask_paths
            
            train_masks = construct_mask_paths(train_images, 'train')
            val_masks = construct_mask_paths(val_images, 'val')
        
        # Setup augmentation if using augmented data
        use_augmentation = variant_config.get('use_augmentation', False)
        if use_augmentation:
            augmented_dir = 'augmented'
            train_images = [os.path.join(augmented_dir, 'train', os.path.basename(img)) for img in train_images]
            val_images = [os.path.join(augmented_dir, 'val', os.path.basename(img)) for img in val_images]
            
            if use_masks:
                train_masks = [os.path.join(augmented_dir, 'train', os.path.basename(mask)) for mask in train_masks]
                val_masks = [os.path.join(augmented_dir, 'val', os.path.basename(mask)) for mask in val_masks]
        
        # Create datasets
        train_dataset = RoadDistressDataset(
            image_paths=train_images,
            labels=train_labels,
            mask_paths=train_masks,
            use_masks=use_masks
        )
        
        val_dataset = RoadDistressDataset(
            image_paths=val_images,
            labels=val_labels,
            mask_paths=val_masks,
            use_masks=use_masks
        )
        
        # Create data loaders
        batch_size = 4  # Small batch size for testing
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move data to device
                if use_masks:
                    images, masks, targets = batch
                    images, masks, targets = images.to(device), masks.to(device), targets.to(device)
                else:
                    images, targets = batch
                    images, targets = images.to(device), targets.to(device)
                    masks = None
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images, masks)
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                train_correct += (predictions == targets).sum().item()
                train_total += targets.size(0)
                
                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if use_masks:
                        images, masks, targets = batch
                        images, masks, targets = images.to(device), masks.to(device), targets.to(device)
                    else:
                        images, targets = batch
                        images, targets = images.to(device), targets.to(device)
                        masks = None
                    
                    outputs = model(images, masks)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    predictions = torch.argmax(outputs, dim=1)
                    val_correct += (predictions == targets).sum().item()
                    val_total += targets.size(0)
            
            # Print progress
            train_acc = 100.0 * train_correct / train_total
            val_acc = 100.0 * val_correct / val_total
            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"  Train - Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")
        
        print(f"\nTraining completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test with model_a (simplest variant)
    success = train_single_model('model_a', epochs=3)
    if success:
        print("\nTest successful! Ready to run full training.")
    else:
        print("\nTest failed. Please check the error messages above.") 