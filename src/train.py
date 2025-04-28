import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
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
    
    def close(self):
        """Close the logger and save final metrics"""
        # Save final metrics
        with open(os.path.join(self.log_dir, 'final_metrics.json'), 'w') as f:
            json.dump({
                'best_val_acc': max(self.metrics['val_acc']),
                'final_train_acc': self.metrics['train_acc'][-1],
                'final_val_acc': self.metrics['val_acc'][-1],
                'final_train_loss': self.metrics['train_loss'][-1],
                'final_val_loss': self.metrics['val_loss'][-1]
            }, f, indent=4)
        
        # Create visualization of training progress
        self._create_training_plots()
    
    def _create_training_plots(self):
        """Create and save training progress plots"""
        try:
            import matplotlib.pyplot as plt
            
            # Create accuracy plot
            plt.figure(figsize=(10, 5))
            plt.plot(self.metrics['train_acc'], label='Training Accuracy')
            plt.plot(self.metrics['val_acc'], label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(os.path.join(self.log_dir, 'accuracy_plot.png'))
            plt.close()
            
            # Create loss plot
            plt.figure(figsize=(10, 5))
            plt.plot(self.metrics['train_loss'], label='Training Loss')
            plt.plot(self.metrics['val_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(self.log_dir, 'loss_plot.png'))
            plt.close()
            
            # Create learning rate plot
            plt.figure(figsize=(10, 5))
            plt.plot(self.metrics['lr'])
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.savefig(os.path.join(self.log_dir, 'lr_plot.png'))
            plt.close()
            
        except ImportError:
            print("Warning: matplotlib not installed. Skipping plot generation.")

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler, gradient_clip):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Use automatic mixed precision
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        
        # Apply gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate metrics
        _, predicted = torch.max(outputs.data, 1)
        _, labels_max = torch.max(labels, 1)  # Convert one-hot to class indices
        total += labels.size(0)
        correct += (predicted == labels_max).sum().item()
        
        total_loss += loss.item()
        current_acc = 100 * correct / total
        pbar.set_postfix({'loss': loss.item(), 'acc': f'{current_acc:.2f}%'})
    
    # Calculate epoch metrics
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            _, predicted = torch.max(outputs.data, 1)
            _, labels_max = torch.max(labels, 1)  # Convert one-hot to class indices
            total += labels.size(0)
            correct += (predicted == labels_max).sum().item()
            
            total_loss += loss.item()
    
    # Calculate metrics
    val_loss = total_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc

def main():
    # Experiment configuration
    experiment_name = "efficientnet_b3_enhanced"
    
    # Model configuration
    model_config = {
        'num_classes': 3,
        'pretrained': True,
        'backbone_type': 'efficientnet_b3',
        'dropout_rate': 0.5  # Increased dropout for better regularization
    }
    
    # Data augmentation configuration
    augmentation_config = {
        'random_horizontal_flip': {'p': 0.5},
        'random_vertical_flip': {'p': 0.3},
        'random_rotation': {'degrees': 15, 'p': 0.5},
        'color_jitter': {
            'brightness': 0.3,
            'contrast': 0.3,
            'saturation': 0.3,
            'hue': 0.1,
            'p': 0.8
        },
        'random_affine': {
            'degrees': 10,
            'translate': (0.1, 0.1),
            'scale': (0.9, 1.1),
            'shear': 5,
            'p': 0.5
        },
        'random_perspective': {
            'distortion_scale': 0.2,
            'p': 0.3
        },
        'random_erasing': {
            'p': 0.5,
            'scale': (0.02, 0.2),
            'ratio': (0.3, 3.3),
            'value': 'random'
        },
        'gaussian_blur': {
            'kernel_size': 3,
            'sigma': (0.1, 2.0),
            'p': 0.3
        },
        'random_adjust_sharpness': {
            'sharpness_factor': 2,
            'p': 0.3
        },
        'random_autocontrast': {'p': 0.3},
        'random_equalize': {'p': 0.3}
    }
    
    # Training configuration
    training_config = {
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.0005,
        'weight_decay': 0.01,
        'early_stopping_patience': 10,
        'gradient_clip': 1.0,
        'lr_scheduler': {
            'type': 'cosine',
            'T_max': 50,
            'eta_min': 1e-6,
            'warmup_epochs': 5
        },
        'optimizer': {
            'type': 'adamw',
            'betas': (0.9, 0.999),
            'eps': 1e-8
        }
    }
    
    # Initialize logger
    logger = ExperimentLogger(experiment_name)
    logger.log_config(model_config, training_config, augmentation_config)
    
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
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = RoadDistressModel(**model_config).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = model.get_optimizer(
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    scheduler = model.get_scheduler(
        optimizer,
        training_config['num_epochs'],
        warmup_epochs=training_config['lr_scheduler']['warmup_epochs']
    )
    
    # Initialize loss function
    criterion = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss for multi-label classification
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(training_config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{training_config['num_epochs']}")
        
        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler,
            training_config['gradient_clip']
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
            if patience_counter >= training_config['early_stopping_patience']:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    
    logger.close()

if __name__ == '__main__':
    main() 