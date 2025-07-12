#!/usr/bin/env python3
"""
Training Script for Model F: CLAHE Enhanced Images with Half Mask Overlay

Model F Configuration:
- CLAHE enhancement using optimized parameters from CSV
- Mask overlay at 0.5 opacity (50% overlay)
- No data augmentation
- Multi-label classification (damage, occlusion, crop)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from clahe_dataset import CLAHEDataModule
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class RoadDistressModelF(nn.Module):
    """Model F: CLAHE + Half Mask Integration for Road Distress Classification"""
    
    def __init__(self, num_classes=3, backbone='efficientnet-b3'):
        super(RoadDistressModelF, self).__init__()
        
        # Use EfficientNet backbone
        if backbone == 'efficientnet-b3':
            self.backbone = models.efficientnet_b3(pretrained=True)
            backbone_features = 1536
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            backbone_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        
        # Enhanced classifier head for multi-label classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class ModelFTrainer:
    """Trainer for Model F"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        output_dir: str = 'experiments/model_f'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function for multi-label classification
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=50,  # Will be set properly during training
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
        
        # Tensorboard writer
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # Training state
        self.best_val_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, epoch: int) -> tuple:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Convert to binary predictions
                predictions = torch.sigmoid(outputs) > 0.5
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # Per-class metrics
        class_names = ['damage', 'occlusion', 'crop']
        metrics = {}
        
        for i, class_name in enumerate(class_names):
            metrics[f'{class_name}_accuracy'] = accuracy_score(all_targets[:, i], all_predictions[:, i])
            metrics[f'{class_name}_precision'] = precision_score(all_targets[:, i], all_predictions[:, i], zero_division=0)
            metrics[f'{class_name}_recall'] = recall_score(all_targets[:, i], all_predictions[:, i], zero_division=0)
            metrics[f'{class_name}_f1'] = f1_score(all_targets[:, i], all_predictions[:, i], zero_division=0)
        
        # Overall metrics
        overall_accuracy = accuracy_score(all_targets.flatten(), all_predictions.flatten())
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, overall_accuracy, metrics
    
    def save_checkpoint(self, epoch: int, val_accuracy: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_accuracy': val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved! Val Accuracy: {val_accuracy:.4f}")
    
    def train(self, num_epochs: int = 50, early_stopping_patience: int = 10):
        """Complete training loop"""
        print(f"Starting Model F training...")
        print(f"Output directory: {self.output_dir}")
        
        # Update scheduler epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            epochs=num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3
        )
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_accuracy, class_metrics = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Val', val_accuracy, epoch)
            
            for metric_name, metric_value in class_metrics.items():
                self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
            
            # Print epoch results
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            
            # Print per-class metrics
            for class_name in ['damage', 'occlusion', 'crop']:
                acc = class_metrics[f'{class_name}_accuracy']
                f1 = class_metrics[f'{class_name}_f1']
                print(f"  {class_name.capitalize()}: Acc={acc:.4f}, F1={f1:.4f}")
            
            # Check for best model
            is_best = val_accuracy > best_val_accuracy
            if is_best:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_accuracy, is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch+1} epochs (patience: {early_stopping_patience})")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        
        # Save training summary
        summary = {
            'model_type': 'Model F (CLAHE + Half Masks)',
            'best_val_accuracy': float(best_val_accuracy),
            'total_epochs': epoch + 1,
            'final_train_loss': float(self.train_losses[-1]),
            'final_val_loss': float(self.val_losses[-1]),
            'training_completed': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Model F: CLAHE + Half Mask Overlay')
    parser.add_argument('--train-images', required=True, help='Training images directory')
    parser.add_argument('--val-images', required=True, help='Validation images directory')
    parser.add_argument('--test-images', required=True, help='Test images directory')
    parser.add_argument('--train-masks', required=True, help='Training masks directory')
    parser.add_argument('--val-masks', required=True, help='Validation masks directory')
    parser.add_argument('--test-masks', required=True, help='Test masks directory')
    parser.add_argument('--train-labels', required=True, help='Training labels CSV')
    parser.add_argument('--val-labels', required=True, help='Validation labels CSV')
    parser.add_argument('--test-labels', required=True, help='Test labels CSV')
    parser.add_argument('--clahe-params', required=True, help='CLAHE parameters JSON')
    parser.add_argument('--output-dir', default='experiments/model_f', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--backbone', default='efficientnet-b3', choices=['efficientnet-b3', 'resnet50'])
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data module
    data_module = CLAHEDataModule(
        train_images_dir=args.train_images,
        val_images_dir=args.val_images,
        test_images_dir=args.test_images,
        train_masks_dir=args.train_masks,
        val_masks_dir=args.val_masks,
        test_masks_dir=args.test_masks,
        train_labels_csv=args.train_labels,
        val_labels_csv=args.val_labels,
        test_labels_csv=args.test_labels,
        clahe_params_json=args.clahe_params,
        mask_opacity=0.5,  # Model F uses half opacity
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    
    # Create data loaders
    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Model setup
    model = RoadDistressModelF(num_classes=3, backbone=args.backbone)
    print(f"Model: {args.backbone} backbone with CLAHE + Half Mask integration")
    
    # Trainer setup
    trainer = ModelFTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir
    )
    
    # Start training
    trainer.train(num_epochs=args.epochs, early_stopping_patience=10)


if __name__ == '__main__':
    main() 