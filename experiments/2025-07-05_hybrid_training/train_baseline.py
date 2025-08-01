#!/usr/bin/env python3
"""
Training script for the baseline model.

This script trains the simplest possible model:
- No road masking
- No data augmentation  
- No CLAHE preprocessing
- Just EfficientNet-B3 on original images

This serves as the baseline to demonstrate the value of all enhancements.
"""

import sys
import logging
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from models.hybrid_model import create_model
from data.dataset import create_dataset
from utils.platform_utils import PlatformManager

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('baseline_training.log')
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_epoch(model, train_loader, criterion, optimizer, device, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Handle different batch formats (with or without masks)
        if len(batch) == 2:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
        else:
            # This shouldn't happen for baseline, but handle gracefully
            images, _, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    avg_loss = total_loss / num_batches
    return avg_loss

def validate(model, val_loader, criterion, device, logger):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Handle different batch formats
            if len(batch) == 2:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
            else:
                images, _, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Convert to binary predictions
            predictions = torch.sigmoid(outputs) > 0.5
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    
    # Overall accuracy (exact match)
    exact_match = np.all(all_predictions == all_labels, axis=1)
    exact_match_accuracy = np.mean(exact_match)
    
    # Hamming accuracy (per-label accuracy)
    hamming_accuracy = np.mean(all_predictions == all_labels)
    
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, exact_match_accuracy, hamming_accuracy

def main():
    """Main training function."""
    logger = setup_logging()
    logger.info("Starting baseline model training...")
    
    # Load configuration
    config = load_config('config/base_config.yaml')
    
    # Setup platform
    platform_manager = PlatformManager(config)
    device = platform_manager.get_device()
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating baseline model...")
    model = create_model('model_baseline', num_classes=3)
    model = model.to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = create_dataset('train', config, 'model_baseline')
    val_dataset = create_dataset('val', config, 'model_baseline')
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=platform_manager.get_num_workers(),
        pin_memory=str(device) == 'cuda'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=platform_manager.get_num_workers(),
        pin_memory=str(device) == 'cuda'
    )
    
    # Setup training components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    num_epochs = int(config['training']['num_epochs'])
    best_val_accuracy = 0.0
    patience_counter = 0
    max_patience = int(config['training']['early_stopping_patience'])
    
    results_dir = Path('results/model_baseline')
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = results_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    logger.info("Starting training loop...")
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, logger)
        
        # Validate
        val_loss, exact_match_acc, hamming_acc = validate(model, val_loader, criterion, device, logger)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Exact Match Accuracy: {exact_match_acc:.4f}")
        logger.info(f"  Hamming Accuracy: {hamming_acc:.4f}")
        
        # Save best model
        is_best = exact_match_acc > best_val_accuracy
        if is_best:
            best_val_accuracy = exact_match_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': exact_match_acc,
                'val_loss': val_loss,
                'config': config
            }, checkpoints_dir / 'best_model.pth')
            logger.info(f"New best model saved! Accuracy: {exact_match_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    
    # Save training summary
    summary = {
        'model_variant': 'model_baseline',
        'total_epochs': epoch + 1,
        'best_epoch': epoch + 1 - patience_counter,
        'best_val_accuracy': float(best_val_accuracy),
        'final_train_loss': float(train_loss),
        'final_val_loss': float(val_loss),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'config': config
    }
    
    import json
    with open(results_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Training summary saved to {results_dir / 'training_summary.json'}")

if __name__ == "__main__":
    main()