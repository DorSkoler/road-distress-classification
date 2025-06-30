import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import json
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
import argparse
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


class DiceLoss(nn.Module):
    """Dice Loss for multi-class classification."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        intersection = (inputs * targets_one_hot).sum(dim=0)
        union = inputs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class RoadDistressTrainer:
    """Trainer for road distress classification models."""
    
    def __init__(self, config: Dict[str, Any], variant_name: str):
        self.config = config
        self.variant_name = variant_name
        
        # Ensure comparative_training section exists
        if 'comparative_training' not in config:
            config['comparative_training'] = {
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
        
        self.variant_config = config['comparative_training']['variants'][variant_name]
        
        # Setup device
        device_config = config.get('training', {}).get('device', 'auto')
        self.device = torch.device(device_config if device_config != 'auto' 
                                 else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Setup directories
        self.setup_directories()
        
        # Setup model
        self.model = create_model_variant(config, variant_name).to(self.device)
        
        # Setup data loaders
        self.train_loader, self.val_loader, self.test_loader = self.setup_data_loaders()
        
        # Setup training components
        self.setup_training_components()
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        self.epoch_metrics = []  # Store per-epoch metrics
        
        print(f"Initialized trainer for {variant_name}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_directories(self):
        """Setup output directories."""
        self.output_dir = os.path.join('results', self.variant_name)
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def setup_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Setup data loaders for training, validation, and test sets."""
        
        # Load split information
        splits_dir = 'splits'
        with open(os.path.join(splits_dir, 'train_images.txt'), 'r') as f:
            train_images = [line.strip() for line in f.readlines()]
        with open(os.path.join(splits_dir, 'val_images.txt'), 'r') as f:
            val_images = [line.strip() for line in f.readlines()]
        with open(os.path.join(splits_dir, 'test_images.txt'), 'r') as f:
            test_images = [line.strip() for line in f.readlines()]
        
        # Load labels from annotation JSONs  
        # Class mapping: 0=not_damaged, 1=damaged, 2=other_issues (occluded/cropped)
        labels_map = {'not_damaged': 0, 'damaged': 1, 'other_issues': 2}
        coryell_path = self.config.get('dataset', {}).get('coryell_path', '../../data/coryell')
        
        def load_labels_from_annotations(image_list):
            labels = []
            for img_path in image_list:
                # img_path format: "Co Rd 232/018_31.615684_-97.742088"
                road_name = img_path.split('/')[0]
                img_name = img_path.split('/')[1]
                ann_path = os.path.join(coryell_path, road_name, 'ann', f"{img_name}.json")
                
                try:
                    with open(ann_path, 'r') as f:
                        ann = json.load(f)
                    
                    # Parse the tags structure
                    tags = ann.get('tags', [])
                    
                    # Initialize flags for each condition
                    is_damaged = False
                    is_occluded = False
                    is_cropped = False
                    
                    # Check each tag
                    for tag in tags:
                        tag_name = tag.get('name', '')
                        tag_value = tag.get('value', '')
                        
                        if tag_name == 'Damage' and tag_value == 'Damaged':
                            is_damaged = True
                        elif tag_name == 'Occlusion' and tag_value == 'Occluded':
                            is_occluded = True
                        elif tag_name == 'Crop' and tag_value == 'Cropped':
                            is_cropped = True
                    
                    # Multi-label format: [damaged, occluded, cropped]
                    multi_label = [
                        1 if is_damaged else 0,      # damaged
                        1 if is_occluded else 0,     # occluded  
                        1 if is_cropped else 0       # cropped
                    ]
                    
                    labels.append(multi_label)
                    
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not load annotation for {img_path}: {e}")
                    labels.append([0, 0, 0])  # Default to no issues
                    
            return labels
        
        train_labels = load_labels_from_annotations(train_images)
        val_labels = load_labels_from_annotations(val_images)
        test_labels = load_labels_from_annotations(test_images)
        
        # Construct full paths to actual image files
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
        test_images = construct_image_paths(test_images)
        
        # Setup mask paths if using masks
        train_masks = None
        val_masks = None
        test_masks = None
        
        if self.variant_config['use_masks']:
            masks_dir = 'masks'
            
            def construct_mask_paths(image_list, split_name):
                mask_paths = []
                for img_path in image_list:
                    # img_path format: "../../data/coryell/Co Rd 232/img/018_31.615684_-97.742088.png"
                    # Need to construct: "masks/train/Co Rd 232/018_31.615684_-97.742088.png"
                    parts = img_path.split(os.sep)  # Use os.sep for cross-platform compatibility
                    road_name = parts[-3]  # "Co Rd 232"
                    img_name = parts[-1].replace('.png', '')  # "018_31.615684_-97.742088"
                    mask_path = os.path.join(masks_dir, split_name, road_name, f"{img_name}.png")
                    mask_paths.append(mask_path)
                return mask_paths
            
            train_masks = construct_mask_paths(train_images, 'train')
            val_masks = construct_mask_paths(val_images, 'val')
            test_masks = construct_mask_paths(test_images, 'test')
        
        # Setup augmentation if using augmented data
        if self.variant_config['use_augmentation']:
            augmented_dir = 'augmented'
            
            def construct_augmented_paths(image_list, split_name):
                aug_paths = []
                for img_path in image_list:
                    # img_path format: "../../data/coryell/Co Rd 232/img/018_31.615684_-97.742088.png"
                    # Need to construct: "augmented/train/images/Co Rd 232/018_31.615684_-97.742088_aug00_geometric.png"
                    parts = img_path.split(os.sep)
                    road_name = parts[-3]  # "Co Rd 232"
                    img_name = parts[-1].replace('.png', '')  # "018_31.615684_-97.742088"
                    # Use the first augmentation version (aug00_geometric)
                    aug_filename = f"{img_name}_aug00_geometric.png"
                    aug_path = os.path.join(augmented_dir, split_name, 'images', road_name, aug_filename)
                    aug_paths.append(aug_path)
                return aug_paths
            
            train_images = construct_augmented_paths(train_images, 'train')
            val_images = construct_augmented_paths(val_images, 'val')
            test_images = construct_augmented_paths(test_images, 'test')
            
            if self.variant_config['use_masks']:
                def construct_augmented_mask_paths(mask_list, split_name):
                    aug_mask_paths = []
                    for mask_path in mask_list:
                        # mask_path format: "masks/train/Co Rd 232/018_31.615684_-97.742088.png"
                        # Need to construct: "augmented/train/masks/Co Rd 232/018_31.615684_-97.742088_aug00_geometric.png"
                        parts = mask_path.split(os.sep)
                        road_name = parts[-2]  # "Co Rd 232"
                        mask_name = parts[-1].replace('.png', '')  # "018_31.615684_-97.742088"
                        # Use the first augmentation version (aug00_geometric)
                        aug_mask_filename = f"{mask_name}_aug00_geometric.png"
                        aug_mask_path = os.path.join(augmented_dir, split_name, 'masks', road_name, aug_mask_filename)
                        aug_mask_paths.append(aug_mask_path)
                    return aug_mask_paths
                
                train_masks = construct_augmented_mask_paths(train_masks, 'train')
                val_masks = construct_augmented_mask_paths(val_masks, 'val')
                test_masks = construct_augmented_mask_paths(test_masks, 'test')
        
        # Create datasets
        train_dataset = RoadDistressDataset(
            image_paths=train_images,
            labels=train_labels,
            mask_paths=train_masks,
            use_masks=self.variant_config['use_masks']
        )
        
        val_dataset = RoadDistressDataset(
            image_paths=val_images,
            labels=val_labels,
            mask_paths=val_masks,
            use_masks=self.variant_config['use_masks']
        )
        
        test_dataset = RoadDistressDataset(
            image_paths=test_images,
            labels=test_labels,
            mask_paths=test_masks,
            use_masks=self.variant_config['use_masks']
        )
        
        # Create data loaders
        batch_size = self.config.get('training', {}).get('batch_size', 16)
        num_workers = self.config.get('system', {}).get('num_workers', 4)
        pin_memory = self.config.get('system', {}).get('pin_memory', True)
        prefetch_factor = self.config.get('training', {}).get('prefetch_factor', 2)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor
        )
        
        print(f"Data loaders created:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        
        return train_loader, val_loader, test_loader
    
    def setup_training_components(self):
        """Setup optimizer, scheduler, and loss function."""
        
        # Get training config with fallbacks
        training_config = self.config.get('training', {})
        
        # Optimizer
        optimizer_config = training_config.get('optimizer', 'adamw')
        if optimizer_config == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=training_config.get('learning_rate', 0.001),
                weight_decay=training_config.get('weight_decay', 0.0001),
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_config == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=training_config.get('learning_rate', 0.001),
                weight_decay=training_config.get('weight_decay', 0.0001)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config}")
        
        # Scheduler
        scheduler_config = training_config.get('scheduler', 'cosine')
        if scheduler_config == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.get('num_epochs', 50)
            )
        elif scheduler_config == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Loss function - Multi-label classification
        loss_config = self.config.get('model', {}).get('loss', {})
        loss_type = loss_config.get('type', 'bce')
        
        if loss_type == 'focal':
            # TODO: Implement multi-label focal loss if needed
            self.criterion = nn.BCEWithLogitsLoss()
            print("Warning: Focal loss not implemented for multi-label, using BCEWithLogitsLoss")
        elif loss_type == 'dice':
            # TODO: Implement multi-label dice loss if needed  
            self.criterion = nn.BCEWithLogitsLoss()
            print("Warning: Dice loss not implemented for multi-label, using BCEWithLogitsLoss")
        elif loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            print(f"Warning: Unsupported loss type {loss_type}, using BCEWithLogitsLoss")
    
    def setup_logging(self):
        """Setup TensorBoard logging."""
        self.writer = SummaryWriter(self.log_dir)
        
        # Log model architecture
        model_summary = get_model_summary(self.model)
        self.writer.add_text('Model/Architecture', str(self.model))
        self.writer.add_text('Model/Summary', json.dumps(model_summary, indent=2))
        
        # Log configuration
        self.writer.add_text('Config', json.dumps(self.config, indent=2))
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            if self.variant_config['use_masks']:
                images, masks, targets = batch
                images, masks, targets = images.to(self.device), masks.to(self.device), targets.to(self.device)
            else:
                images, targets = batch
                images, targets = images.to(self.device), targets.to(self.device)
                masks = None
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, masks)
            loss = self.criterion(outputs, targets.float())  # BCEWithLogitsLoss expects float targets
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            # Multi-label predictions: apply sigmoid and threshold at 0.5
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            # Debug model outputs on first batch of first epoch
            if batch_idx == 0 and self.current_epoch == 0:
                print(f"    Debug - Raw outputs shape: {outputs.shape}")
                print(f"    Debug - Raw outputs (first 3): {outputs[:3].detach().cpu()}")
                print(f"    Debug - Targets (first 3): {targets[:3].cpu()}")
                print(f"    Debug - Probabilities (first 3): {probabilities[:3].detach().cpu()}")
                print(f"    Debug - Predictions (first 3): {predictions[:3].cpu()}")
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate multi-label metrics
        all_targets = np.array(all_targets, dtype=np.float32)
        all_predictions = np.array(all_predictions, dtype=np.float32)
        
        # Debug info
        if self.current_epoch == 0:  # Only print on first epoch to avoid spam
            print(f"    Debug - Targets shape: {all_targets.shape}, dtype: {all_targets.dtype}")
            print(f"    Debug - Predictions shape: {all_predictions.shape}, dtype: {all_predictions.dtype}")
            print(f"    Debug - First 5 targets: {all_targets[:5]}")
            print(f"    Debug - First 5 predictions: {all_predictions[:5]}")
            print(f"    Debug - Target label distribution: {np.mean(all_targets, axis=0)}")
            print(f"    Debug - Prediction label distribution: {np.mean(all_predictions, axis=0)}")
        
        # Multi-label metrics
        exact_match_accuracy = np.mean(np.all(all_targets == all_predictions, axis=1))
        
        # Per-label metrics - flatten arrays for sklearn compatibility
        all_targets_flat = all_targets.flatten()
        all_predictions_flat = all_predictions.flatten()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets_flat, all_predictions_flat, average='macro', zero_division=0
        )
        accuracy = exact_match_accuracy
        
        metrics = {
            'loss': total_loss / len(self.train_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                if self.variant_config['use_masks']:
                    images, masks, targets = batch
                    images, masks, targets = images.to(self.device), masks.to(self.device), targets.to(self.device)
                else:
                    images, targets = batch
                    images, targets = images.to(self.device), targets.to(self.device)
                    masks = None
                
                # Forward pass
                outputs = self.model(images, masks)
                loss = self.criterion(outputs, targets.float())
                
                # Statistics
                total_loss += loss.item()
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate multi-label metrics
        all_targets = np.array(all_targets, dtype=np.float32)
        all_predictions = np.array(all_predictions, dtype=np.float32)
        
        # Debug info for validation
        if self.current_epoch == 0:  # Only print on first epoch to avoid spam
            print(f"    Val Debug - Targets shape: {all_targets.shape}, dtype: {all_targets.dtype}")
            print(f"    Val Debug - Predictions shape: {all_predictions.shape}, dtype: {all_predictions.dtype}")
            print(f"    Val Debug - First 5 targets: {all_targets[:5]}")
            print(f"    Val Debug - First 5 predictions: {all_predictions[:5]}")
        
        # Multi-label metrics
        exact_match_accuracy = np.mean(np.all(all_targets == all_predictions, axis=1))
        
        # Per-label metrics - flatten arrays for sklearn compatibility
        all_targets_flat = all_targets.flatten()
        all_predictions_flat = all_predictions.flatten()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets_flat, all_predictions_flat, average='macro', zero_division=0
        )
        accuracy = exact_match_accuracy
        
        metrics = {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def test_model(self) -> Dict[str, float]:
        """Test the model on test set."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Move data to device
                if self.variant_config['use_masks']:
                    images, masks, targets = batch
                    images, masks, targets = images.to(self.device), masks.to(self.device), targets.to(self.device)
                else:
                    images, targets = batch
                    images, targets = images.to(self.device), targets.to(self.device)
                    masks = None
                
                # Forward pass
                outputs = self.model(images, masks)
                loss = self.criterion(outputs, targets.float())
                
                # Statistics
                total_loss += loss.item()
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate multi-label metrics
        all_targets = np.array(all_targets, dtype=np.float32)
        all_predictions = np.array(all_predictions, dtype=np.float32)
        
        # Debug info for test
        print(f"    Test Debug - Targets shape: {all_targets.shape}, dtype: {all_targets.dtype}")
        print(f"    Test Debug - Predictions shape: {all_predictions.shape}, dtype: {all_predictions.dtype}")
        print(f"    Test Debug - First 5 targets: {all_targets[:5]}")
        print(f"    Test Debug - First 5 predictions: {all_predictions[:5]}")
        print(f"    Test Debug - Target label distribution: {np.mean(all_targets, axis=0)}")
        print(f"    Test Debug - Prediction label distribution: {np.mean(all_predictions, axis=0)}")
        
        # Multi-label metrics
        exact_match_accuracy = np.mean(np.all(all_targets == all_predictions, axis=1))
        
        # Per-label metrics - flatten arrays for sklearn compatibility
        all_targets_flat = all_targets.flatten()
        all_predictions_flat = all_predictions.flatten()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets_flat, all_predictions_flat, average='macro', zero_division=0
        )
        accuracy = exact_match_accuracy
        
        # Per-label metrics
        class_names = ['damaged', 'occluded', 'cropped']
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
                all_targets[:, i], all_predictions[:, i], average='binary', zero_division=0
            )
            per_class_metrics[class_name] = {
                'precision': class_precision,
                'recall': class_recall,
                'f1': class_f1
            }
        
        metrics = {
            'loss': total_loss / len(self.test_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class': per_class_metrics
        }
        
        # Save predictions
        if self.config.get('evaluation', {}).get('save_predictions', True):
            predictions_file = os.path.join(self.output_dir, 'test_predictions.json')
            
            # Convert numpy arrays to lists for JSON serialization
            def make_json_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.floating, float)):
                    return float(obj)
                elif isinstance(obj, (np.integer, int)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(v) for v in obj]
                else:
                    return obj
            
            predictions_data = {
                'predictions': all_predictions.tolist(),
                'targets': all_targets.tolist(),
                'probabilities': [p.tolist() for p in all_probabilities],
                'metrics': make_json_serializable(metrics)
            }
            with open(predictions_file, 'w') as f:
                json.dump(predictions_data, f, indent=2)
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config,
            'variant_name': self.variant_name
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path, _use_new_zipfile_serialization=False)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path, _use_new_zipfile_serialization=False)
            print(f"  Saved best model with metric: {self.best_metric:.4f}")
    
    def save_epoch_metrics(self):
        """Save all epoch metrics to JSON (serializable)."""
        metrics_path = os.path.join(self.output_dir, 'epoch_metrics.json')
        
        def make_serializable(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            else:
                return obj
        
        serializable_metrics = make_serializable(self.epoch_metrics)
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def create_test_visualizations(self, test_metrics):
        """Create visualizations for test results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create visualizations directory
            viz_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # 1. Per-label metrics bar chart
            if 'per_class' in test_metrics:
                per_class = test_metrics['per_class']
                labels = list(per_class.keys())
                f1_scores = [per_class[label]['f1'] for label in labels]
                precisions = [per_class[label]['precision'] for label in labels]
                recalls = [per_class[label]['recall'] for label in labels]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(labels))
                width = 0.25
                
                ax.bar(x - width, f1_scores, width, label='F1-Score', alpha=0.8)
                ax.bar(x, precisions, width, label='Precision', alpha=0.8)
                ax.bar(x + width, recalls, width, label='Recall', alpha=0.8)
                
                ax.set_xlabel('Labels')
                ax.set_ylabel('Score')
                ax.set_title(f'Per-Label Metrics - {self.variant_name}')
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'per_label_metrics.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            # 2. Overall metrics summary
            fig, ax = plt.subplots(figsize=(8, 6))
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            values = [test_metrics[metric] for metric in metrics]
            
            bars = ax.bar(metrics, values, alpha=0.7, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
            ax.set_ylabel('Score')
            ax.set_title(f'Overall Test Metrics - {self.variant_name}')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'overall_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Training curves (if epoch metrics available)
            if len(self.epoch_metrics) > 0:
                epochs = [m['epoch'] for m in self.epoch_metrics]
                train_f1 = [m['train']['f1'] for m in self.epoch_metrics]
                val_f1 = [m['val']['f1'] for m in self.epoch_metrics]
                train_loss = [m['train']['loss'] for m in self.epoch_metrics]
                val_loss = [m['val']['loss'] for m in self.epoch_metrics]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # F1 curves
                ax1.plot(epochs, train_f1, label='Train F1', linewidth=2)
                ax1.plot(epochs, val_f1, label='Val F1', linewidth=2)
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('F1-Score')
                ax1.set_title(f'F1-Score Training Curves - {self.variant_name}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Loss curves
                ax2.plot(epochs, train_loss, label='Train Loss', linewidth=2)
                ax2.plot(epochs, val_loss, label='Val Loss', linewidth=2)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.set_title(f'Loss Training Curves - {self.variant_name}')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"  Visualizations saved to: {viz_dir}")
            
        except ImportError:
            print("  Warning: matplotlib/seaborn not available, skipping visualizations")
        except Exception as e:
            print(f"  Warning: Failed to create visualizations: {e}")

    def load_best_model(self):
        """Load the best model checkpoint."""
        best_path = os.path.join(self.checkpoint_dir, 'best.pth')
        if os.path.exists(best_path):
            try:
                # Try with weights_only=False for PyTorch 2.6+ compatibility
                checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  Loaded best model from epoch {checkpoint['epoch']} with metric: {checkpoint['best_metric']:.4f}")
            except Exception as e:
                print(f"  Error loading checkpoint: {e}")
                print(f"  Using current model state")
        else:
            print(f"  Warning: No best model found at {best_path}, using current model")

    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.variant_name}")
        print(f"Description: {self.variant_config['description']}")
        
        start_time = time.time()
        
        for epoch in range(self.config.get('training', {}).get('num_epochs', 50)):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch + 1}/{self.config.get('training', {}).get('num_epochs', 50)}")
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Log metrics
            for metric_name, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{metric_name}', value, epoch)
            for metric_name, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{metric_name}', value, epoch)
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
            
            # Store epoch metrics
            epoch_data = {
                'epoch': epoch + 1,
                'train': train_metrics.copy(),
                'val': val_metrics.copy(),
                'learning_rate': current_lr
            }
            self.epoch_metrics.append(epoch_data)
            
            # Print progress
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Check for best model
            best_metric_name = self.config.get('logging', {}).get('best_metric', 'f1')
            current_metric = val_metrics[best_metric_name]
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(is_best=False)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Save epoch metrics
            self.save_epoch_metrics()
            
            # Early stopping
            early_stopping_config = self.config.get('training', {}).get('early_stopping', {})
            if early_stopping_config.get('enabled', False):
                patience = early_stopping_config.get('patience', 10)
                if self.patience_counter >= patience:
                    print(f"  Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Training completed
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Save final epoch metrics
        self.save_epoch_metrics()
        
        # Load best model and test
        print("\nLoading best model for testing...")
        self.load_best_model()
        
        # Test best model
        print("\nTesting best model...")
        test_metrics = self.test_model()
        
        # Log final results
        for metric_name, value in test_metrics.items():
            if metric_name != 'per_class':
                self.writer.add_scalar(f'Test/{metric_name}', value, 0)
        
        # Save final results
        results = {
            'variant_name': self.variant_name,
            'description': self.variant_config['description'],
            'training_time': training_time,
            'best_metric': self.best_metric,
            'test_metrics': test_metrics,
            'config': self.config
        }
        
        results_file = os.path.join(self.output_dir, 'training_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nFinal Test Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1-Score: {test_metrics['f1']:.4f}")
        
        self.create_test_visualizations(test_metrics)
        
        self.writer.close()
        return results

    def stop_and_test(self):
        """Manually stop training and test the best model."""
        print(f"\nManually stopping training for {self.variant_name}")
        
        # Load best model
        print("Loading best model for testing...")
        self.load_best_model()
        
        # Test best model
        print("Testing best model...")
        test_metrics = self.test_model()
        
        # Create visualizations
        self.create_test_visualizations(test_metrics)
        
        # Save results
        results = {
            'variant_name': self.variant_name,
            'description': self.variant_config['description'],
            'best_metric': self.best_metric,
            'test_metrics': test_metrics,
            'config': self.config
        }
        
        results_file = os.path.join(self.output_dir, 'training_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTest Results for {self.variant_name}:")
        print(f"  Overall Metrics:")
        print(f"    Accuracy (exact match): {test_metrics['accuracy']:.4f}")
        print(f"    Precision (macro avg): {test_metrics['precision']:.4f}")
        print(f"    Recall (macro avg): {test_metrics['recall']:.4f}")
        print(f"    F1-Score (macro avg): {test_metrics['f1']:.4f}")
        
        if 'per_class' in test_metrics:
            print(f"\n  Per-Class Metrics:")
            for class_name, metrics in test_metrics['per_class'].items():
                print(f"    {class_name.capitalize()}:")
                print(f"      Precision: {metrics['precision']:.4f}")
                print(f"      Recall: {metrics['recall']:.4f}")
                print(f"      F1-Score: {metrics['f1']:.4f}")
        
        self.writer.close()
        return results


def main():
    """Main function to train all model variants."""
    
    parser = argparse.ArgumentParser(description='Train road distress classification models')
    parser.add_argument('--stop-current', action='store_true', 
                       help='Stop current model training and test best model')
    parser.add_argument('--variant', type=str, default=None,
                       help='Specific variant to train (model_a, model_b, model_c, model_d)')
    args = parser.parse_args()
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    torch.manual_seed(config.get('system', {}).get('random_seed', 42))
    np.random.seed(config.get('system', {}).get('random_seed', 42))
    
    # Determine which variants to train
    if args.variant:
        variants = [args.variant]
    else:
        variants = ['model_a', 'model_b', 'model_c', 'model_d']
    
    all_results = {}
    
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"TRAINING {variant.upper()}")
        print(f"{'='*60}")
        
        try:
            trainer = RoadDistressTrainer(config, variant)
            
            if args.stop_current:
                # Just test the current best model
                results = trainer.stop_and_test()
            else:
                # Full training
                results = trainer.train()
            
            all_results[variant] = results
            
            # Clear GPU memory after each model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error training {variant}: {str(e)}")
            all_results[variant] = {'error': str(e)}
            
            # Clear GPU memory even on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save combined results
    combined_results_file = 'results/combined_training_results.json'
    os.makedirs('results', exist_ok=True)
    with open(combined_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    
    for variant, results in all_results.items():
        if 'error' not in results:
            test_metrics = results['test_metrics']
            print(f"\n{variant.upper()}: {results['description']}")
            print(f"  Overall Metrics:")
            print(f"    Accuracy (exact match): {test_metrics['accuracy']:.4f}")
            print(f"    F1-Score (macro avg): {test_metrics['f1']:.4f}")
            if 'training_time' in results:
                print(f"    Training Time: {results['training_time']:.2f}s")
            
            if 'per_class' in test_metrics:
                print(f"  Per-Class F1-Scores:")
                for class_name, metrics in test_metrics['per_class'].items():
                    print(f"    {class_name.capitalize()}: {metrics['f1']:.4f}")
        else:
            print(f"\n{variant.upper()}: FAILED - {results['error']}")
    
    print(f"\nResults saved to: {combined_results_file}")


if __name__ == "__main__":
    main() 