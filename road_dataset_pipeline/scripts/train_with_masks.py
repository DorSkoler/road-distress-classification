import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class RoadDatasetWithMasks(Dataset):
    def __init__(self, image_dir, mask_dir, json_dir, split='train', transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.json_dir = json_dir
        self.transform = transform
        self.split = split
        
        # Get all image-mask pairs for the specified split
        self.samples = []
        img_split_dir = os.path.join(image_dir, split)
        mask_split_dir = os.path.join(mask_dir, split)
        
        if not os.path.exists(img_split_dir):
            print(f"Warning: {split} directory not found at {img_split_dir}")
            return
            
        for img_name in os.listdir(img_split_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                base_name = os.path.splitext(img_name)[0]
                mask_name = f"{base_name}_mask.png"
                
                if os.path.exists(os.path.join(mask_split_dir, mask_name)):
                    # Find corresponding JSON file
                    json_name = f"{base_name}.json"
                    json_path = os.path.join(json_dir, json_name)
                    
                    if os.path.exists(json_path):
                        # Load JSON data
                        with open(json_path, 'r') as f:
                            img_data = json.load(f)
                        
                        # Initialize labels
                        labels = torch.zeros(3)  # [Damage, Occlusion, Crop]
                        
                        # Extract labels from tags
                        for tag in img_data.get('tags', []):
                            if tag['name'] == 'Damage':
                                labels[0] = 1 if tag['value'] == 'Damaged' else 0
                            elif tag['name'] == 'Occlusion':
                                labels[1] = 1 if tag['value'] == 'Occluded' else 0
                            elif tag['name'] == 'Crop':
                                labels[2] = 1 if tag['value'] == 'Cropped' else 0
                        
                        self.samples.append({
                            'image': os.path.join(img_split_dir, img_name),
                            'mask': os.path.join(mask_split_dir, mask_name),
                            'labels': labels
                        })
                    else:
                        print(f"Warning: No JSON file found for {img_name}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image and mask
        image = cv2.imread(sample['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        
        # Resize to model input size
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        return image, mask, sample['labels']

class RoadDistressModelWithMasks(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # EfficientNet-B3 backbone
        self.backbone = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=num_classes,
            activation=None
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, mask):
        # Get features from backbone
        features = self.backbone(x)  # [B, num_classes, H, W]
        
        # Apply mask to focus on road pixels
        # Expand mask to match feature dimensions
        mask = mask.expand_as(features)  # [B, num_classes, H, W]
        
        # Zero out non-road pixels
        masked_features = features * mask
        
        # Get classification predictions
        return self.classifier(masked_features)

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
        """Create and save training visualization plots"""
        plots_dir = os.path.join(self.log_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics['train_loss'], label='Train Loss')
        plt.plot(self.metrics['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, 'loss_plot.png'))
        plt.close()
        
        # Accuracy plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics['train_acc'], label='Train Accuracy')
        plt.plot(self.metrics['val_acc'], label='Val Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, 'accuracy_plot.png'))
        plt.close()
        
        # Learning rate plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics['lr'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.savefig(os.path.join(plots_dir, 'lr_plot.png'))
        plt.close()

def train_model():
    # Configuration optimized for RTX 4070 Ti Super
    config = {
        'model': {
            'backbone': 'efficientnet-b3',
            'num_classes': 3,  # Damage, Occlusion, Crop
            'activation': None
        },
        'training': {
            'batch_size': 64,
            'num_epochs': 50,
            'learning_rate': 1e-3,
            'weight_decay': 0.02,
            'optimizer': 'AdamW',
            'scheduler': 'OneCycleLR',
            'warmup_pct': 0.3,
            'gradient_clip': 1.0,
            'mixed_precision': True,
            'early_stopping_patience': 10
        },
        'data': {
            'image_size': 256,
            'num_workers': 8,
            'pin_memory': True,
            'persistent_workers': True
        },
        'hardware': {
            'device': 'cuda',
            'cudnn_benchmark': True
        }
    }
    
    # Initialize logger
    logger = ExperimentLogger("efficientnet_b3_with_masks")
    logger.log_config(config['model'], config['training'], config['data'])
    
    BATCH_SIZE = config['training']['batch_size']
    NUM_EPOCHS = config['training']['num_epochs']
    LEARNING_RATE = config['training']['learning_rate']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create train and validation datasets
    train_dataset = RoadDatasetWithMasks(
        image_dir='../filtered',
        mask_dir='../filtered_masks',
        json_dir='../tagged_json',
        split='train'
    )
    val_dataset = RoadDatasetWithMasks(
        image_dir='../filtered',
        mask_dir='../filtered_masks',
        json_dir='../tagged_json',
        split='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=config['data']['persistent_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # Create model
    model = RoadDistressModelWithMasks(num_classes=3).to(DEVICE)
    
    # Enable cuDNN benchmarking
    torch.backends.cudnn.benchmark = config['hardware']['cudnn_benchmark']
    
    # Loss function for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=config['training']['warmup_pct'],
        div_factor=25,
        final_div_factor=1e4
    )
    
    # Initialize gradient scaler
    scaler = GradScaler()
    
    # Training loop
    print(f"Training on {DEVICE}")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, masks, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training")):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            with autocast():
                preds = model(images, masks)
                loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(preds) > 0.5).float()
            train_total += labels.numel()
            train_correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, masks, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"):
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                
                with autocast():
                    preds = model(images, masks)
                    loss = criterion(preds, labels)
                
                val_loss += loss.item()
                predicted = (torch.sigmoid(preds) > 0.5).float()
                val_total += labels.numel()
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Log metrics
        logger.log_metrics(epoch, {
            'train_loss': avg_train_loss,
            'train_acc': train_accuracy,
            'val_loss': avg_val_loss,
            'val_acc': val_accuracy,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Save model checkpoint
        logger.save_model(model, optimizer, epoch, {
            'train_loss': avg_train_loss,
            'train_acc': train_accuracy,
            'val_loss': avg_val_loss,
            'val_acc': val_accuracy
        })
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
    
    # Close logger and save final metrics
    logger.close()
    print("\nTraining complete!")
    print(f"Best validation accuracy: {max(logger.metrics['val_acc']):.2f}%")
    print(f"All training information saved to: {logger.log_dir}")

if __name__ == '__main__':
    train_model() 