import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

from model import RoadDistressModel
from data_loader import RoadDistressDataset

class ExperimentLogger:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join("experiments", f"{experiment_name}_{self.timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize metrics dictionary
        self.metrics = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        # Create experiment config file
        self.config = {
            'experiment_name': experiment_name,
            'timestamp': self.timestamp,
            'model_config': {},
            'training_config': {},
            'augmentation_config': {}
        }
    
    def log_config(self, model_config, training_config, augmentation_config):
        """Log experiment configuration"""
        self.config['model_config'] = model_config
        self.config['training_config'] = training_config
        self.config['augmentation_config'] = augmentation_config
        
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def log_metrics(self, epoch, metrics_dict):
        """Log metrics for current epoch"""
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Save metrics to file
        with open(os.path.join(self.log_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def save_model(self, model, optimizer, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, os.path.join(self.log_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        # Save best model if it's the best so far
        if metrics['val_acc'] == max(self.metrics['val_acc']):
            torch.save(model.state_dict(), os.path.join(self.log_dir, 'best_model.pth'))

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Use automatic mixed precision
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate metrics
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate epoch metrics
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
    
    # Calculate metrics
    val_loss = total_loss / len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    
    return val_loss, val_acc

def main():
    # Experiment configuration
    experiment_name = "efficientnet_b3_enhanced"
    
    # Model configuration
    model_config = {
        'num_classes': 3,
        'pretrained': True,
        'backbone_type': 'efficientnet_b3'
    }
    
    # Training configuration
    train_config = {
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,  # Increased learning rate
        'weight_decay': 1e-4,
        'gradient_clip': 1.0,
        'early_stopping_patience': 5
    }
    
    # Augmentation configuration
    aug_config = {
        'resize': (224, 224),
        'random_horizontal_flip': True,
        'random_rotation': 15,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1
        },
        'random_affine': {
            'degrees': 10,
            'translate': (0.1, 0.1),
            'scale': (0.9, 1.1)
        },
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    
    # Initialize logger
    logger = ExperimentLogger(experiment_name)
    logger.log_config(model_config, train_config, aug_config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = RoadDistressDataset(
        data_dir='organized_dataset',  # Updated path
        split='train',
        transform_type='train'
    )
    val_dataset = RoadDistressDataset(
        data_dir='organized_dataset',  # Updated path
        split='val',
        transform_type='val'
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = RoadDistressModel(**model_config).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = model.get_optimizer(
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    scheduler = model.get_scheduler(optimizer, train_config['num_epochs'])
    
    # Initialize loss function
    criterion = model.get_loss_function()
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(train_config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{train_config['num_epochs']}")
        
        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler
        )
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        logger.log_metrics(epoch, {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save model checkpoint
        logger.save_model(model, optimizer, epoch, {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= train_config['early_stopping_patience']:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    
    logger.close()

if __name__ == '__main__':
    main() 