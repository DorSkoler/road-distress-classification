#!/usr/bin/env python3
"""
Cross-Platform Trainer for Hybrid Training Experiment
Date: 2025-07-05

This module implements a comprehensive training pipeline for individual model variants:
- Based on successful 88.99% accuracy configuration from 2025-05-10 experiment
- Cross-platform compatibility for Windows/Mac/Linux
- Support for all 4 model variants (A, B, C, D)
- Advanced training features: mixed precision, gradient clipping, early stopping

Adapted from successful architectures with modern training practices.
"""

# Suppress all warnings FIRST before any imports
import os
import warnings
import logging
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress warnings in subprocesses

# Suppress TensorFlow logging at the root level
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow.python.util.deprecation').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)



import sys
import json
import yaml
import time
import signal
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt



# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.platform_utils import PlatformManager, get_platform_manager
from models.hybrid_model import create_model, MODEL_VARIANTS
from data.dataset import create_dataset, get_dataset_stats

logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    shutdown_requested = True
    print("\n\n⚠️  Shutdown requested! Finishing current batch and saving checkpoint...")
    print("Press Ctrl+C again to force quit (may lose progress)")

# Set up signal handler
signal.signal(signal.SIGINT, signal_handler)

def worker_init_fn(worker_id):
    """Initialize worker processes to suppress warnings."""
    import warnings
    import os
    import logging
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    # Suppress TensorFlow logging in workers
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('tensorflow.python.util.deprecation').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)

class HybridTrainer:
    """Cross-platform trainer for hybrid road distress classification."""
    
    def __init__(self, config_path: str = "config/base_config.yaml", variant: str = 'model_a'):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
            variant: Model variant to train ('model_a', 'model_b', 'model_c', 'model_d')
        """
        self.variant = variant
        self.config = self._load_config(config_path)
        self.platform_utils = PlatformManager(self.config)
        
        # Setup training environment
        self.setup_environment()
        self.setup_device()
        self.setup_paths()
        
        # Initialize training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None  # For mixed precision
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        logger.info(f"Initialized HybridTrainer for {variant}")
        logger.info(f"Platform: {self.platform_utils.platform_info['os']}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply platform-specific configurations
        # Configuration is loaded as-is, platform manager will handle platform-specific logic
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def setup_environment(self):
        """Setup training environment."""
        # Set random seeds for reproducibility
        seed = self.config.get('system', {}).get('random_seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        logger.info(f"Set random seed to {seed}")
    
    def setup_device(self):
        """Setup device for training with cross-platform support."""
        device_str = self.platform_utils.get_device()
        self.device = torch.device(device_str)
        
        # Log device information
        if self.device.type == 'cuda':
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif self.device.type == 'mps':
            logger.info("Using Apple Silicon MPS acceleration")
        else:
            logger.info("Using CPU (no GPU acceleration available)")
    
    def setup_paths(self):
        """Setup all necessary paths with cross-platform compatibility."""
        # Results directory for this variant
        self.results_dir = Path(f"results/model_{self.variant[-1]}").resolve()
        self.platform_utils.create_directory(self.results_dir)
        
        # Subdirectories
        self.checkpoints_dir = self.results_dir / "checkpoints"
        self.logs_dir = self.results_dir / "logs"
        self.plots_dir = self.results_dir / "plots"
        
        for dir_path in [self.checkpoints_dir, self.logs_dir, self.plots_dir]:
            self.platform_utils.create_directory(dir_path)
        
        logger.info(f"Results directory: {self.results_dir}")
    
    def create_model(self) -> nn.Module:
        """Create model for the specified variant."""
        logger.info(f"Creating model for variant: {self.variant}")
        
        model_config = self.config.get('model', {})
        
        model = create_model(
            variant=self.variant,
            num_classes=model_config.get('num_classes', 3),
            dropout_rate=model_config.get('dropout_rate', 0.5),
            encoder_name=model_config.get('encoder_name', 'efficientnet-b3'),
            encoder_weights=model_config.get('encoder_weights', 'imagenet')
        )
        
        model = model.to(self.device)
        
        # Log model information
        info = model.get_model_info()
        logger.info(f"Model created:")
        logger.info(f"  - Architecture: {info['architecture']}")
        logger.info(f"  - Parameters: {info['total_parameters']:,}")
        logger.info(f"  - Use masks: {info['config']['use_masks']}")
        logger.info(f"  - Mask weight: {info['config']['mask_weight']}")
        
        return model
    
    def create_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create datasets and data loaders for training."""
        logger.info("Creating datasets...")
        
        # Create datasets for each split
        train_dataset = create_dataset('train', self.config, self.variant)
        val_dataset = create_dataset('val', self.config, self.variant)
        test_dataset = create_dataset('test', self.config, self.variant)
        
        # Log dataset statistics
        for split, dataset in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
            stats = get_dataset_stats(dataset)
            logger.info(f"{split.capitalize()} dataset:")
            logger.info(f"  - Total samples: {stats['total_samples']}")
            logger.info(f"  - Original: {stats['original_samples']}")
            logger.info(f"  - Augmented: {stats['augmented_samples']}")
            logger.info(f"  - Class distribution: {stats['class_distribution']}")
        
        # Create data loaders with platform-specific configuration
        dataset_config = self.config['dataset']
        batch_size = dataset_config['batch_size']
        num_workers = dataset_config.get('num_workers')
        
        # Get optimal number of workers from platform manager if not specified
        if num_workers is None:
            num_workers = self.platform_utils.get_num_workers()
        
        # Validate that num_workers is properly configured
        if num_workers is None:
            raise ValueError("num_workers could not be determined from configuration or platform manager")
        
        # Adjust batch size and workers based on platform
        if self.platform_utils.platform_info['is_mac']:
            # Reduce for thermal management on Mac
            batch_size = min(batch_size, 32)
            num_workers = min(num_workers, 4)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda',
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn if num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda',
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn if num_workers > 0 else None
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda',
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn if num_workers > 0 else None
        )
        
        logger.info(f"Data loaders created:")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Num workers: {num_workers}")
        logger.info(f"  - Train batches: {len(train_loader)}")
        logger.info(f"  - Val batches: {len(val_loader)}")
        logger.info(f"  - Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        training_config = self.config['training']
        
        optimizer_name = training_config.get('optimizer', 'AdamW').lower()
        learning_rate = float(training_config['learning_rate'])
        weight_decay = float(training_config.get('weight_decay', 0.02))
        
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        logger.info(f"Created optimizer: {optimizer_name}")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - Weight decay: {weight_decay}")
        
        return optimizer
    
    def create_scheduler(self, optimizer: optim.Optimizer, train_loader: DataLoader):
        """Create learning rate scheduler."""
        training_config = self.config['training']
        scheduler_name = training_config.get('scheduler', 'OneCycleLR').lower()
        
        if scheduler_name == 'onecyclelr':
            max_lr = float(training_config['learning_rate'])
            epochs = int(training_config['num_epochs'])
            steps_per_epoch = len(train_loader)
            
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=float(training_config.get('warmup_pct', 0.3)),
                anneal_strategy='cos'
            )
        elif scheduler_name == 'cosineannealinglr':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(training_config['num_epochs'])
            )
        elif scheduler_name == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(training_config.get('step_size', 10)),
                gamma=float(training_config.get('gamma', 0.1))
            )
        else:
            scheduler = None
        
        if scheduler:
            logger.info(f"Created scheduler: {scheduler_name}")
        else:
            logger.info("No scheduler configured")
        
        return scheduler
    
    def create_criterion(self) -> nn.Module:
        """Create loss criterion with class weights and label smoothing support."""
        loss_config = self.config['training'].get('loss', {})
        
        # Get class weights if specified
        class_weights = loss_config.get('class_weights', None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            logger.info(f"Using class weights: {class_weights.tolist()}")
        
        # Get label smoothing if specified
        label_smoothing = loss_config.get('label_smoothing', 0.0)
        
        if label_smoothing > 0:
            # Use CrossEntropyLoss with label smoothing for multi-class
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
            logger.info(f"Created CrossEntropyLoss with label_smoothing={label_smoothing}")
        elif class_weights is not None:
            # Use weighted BCEWithLogitsLoss for multi-label
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            logger.info(f"Created weighted BCEWithLogitsLoss")
        else:
            # Standard BCEWithLogitsLoss
            criterion = nn.BCEWithLogitsLoss()
            logger.info(f"Created standard BCEWithLogitsLoss")
        
        return criterion
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module, epoch: int) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Check for shutdown request
            if shutdown_requested:
                logger.info("Shutdown requested during training epoch, stopping...")
                break
                
            # Handle different batch formats
            if len(batch) == 3:  # With masks
                images, masks, labels = batch
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
            else:  # Without masks
                images, labels = batch
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                masks = None
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    if masks is not None:
                        outputs = model(images, masks)
                    else:
                        outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config['training']['gradient_clip']
                    )
                
                # Optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Regular forward pass
                if masks is not None:
                    outputs = model(images, masks)
                else:
                    outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config['training']['gradient_clip']
                    )
                
                # Optimizer step
                optimizer.step()
            
            # Update scheduler if it's step-based
            if self.scheduler and hasattr(self.scheduler, 'step'):
                # Check if it's a step-based scheduler (OneCycleLR)
                if hasattr(self.scheduler, 'step_count'):
                    self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return total_loss / num_batches
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            
            for batch in pbar:
                # Check for shutdown request
                if shutdown_requested:
                    logger.info("Shutdown requested during validation, stopping...")
                    break
                    
                # Handle different batch formats
                if len(batch) == 3:  # With masks
                    images, masks, labels = batch
                    images = images.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                else:  # Without masks
                    images, labels = batch
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    masks = None
                
                # Forward pass
                if masks is not None:
                    outputs = model(images, masks)
                else:
                    outputs = model(images)
                
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Store predictions and labels for metrics
                predictions = torch.sigmoid(outputs)
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        metrics = self.calculate_metrics(all_predictions, all_labels)
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, metrics
    
    def calculate_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        # Convert to binary predictions
        binary_preds = (predictions > 0.5).float()
        
        # Calculate per-class metrics
        metrics = {}
        class_names = ['damage', 'occlusion', 'crop']
        
        for i, class_name in enumerate(class_names):
            pred_class = binary_preds[:, i]
            label_class = labels[:, i]
            
            # True positives, false positives, false negatives
            tp = (pred_class * label_class).sum().item()
            fp = (pred_class * (1 - label_class)).sum().item()
            fn = ((1 - pred_class) * label_class).sum().item()
            tn = ((1 - pred_class) * (1 - label_class)).sum().item()
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            metrics[f'{class_name}_precision'] = precision
            metrics[f'{class_name}_recall'] = recall
            metrics[f'{class_name}_f1'] = f1
            metrics[f'{class_name}_accuracy'] = accuracy
        
        # Overall metrics
        overall_accuracy = (binary_preds == labels).float().mean().item()
        
        # Calculate weighted F1 (same as before)
        weighted_f1 = np.mean([metrics[f'{name}_f1'] for name in class_names])
        
        # Calculate macro F1 (unweighted average - more honest with imbalanced classes)
        f1_scores = [metrics[f'{name}_f1'] for name in class_names]
        macro_f1 = np.mean(f1_scores)
        
        # Calculate macro precision and recall
        precisions = [metrics[f'{name}_precision'] for name in class_names]
        recalls = [metrics[f'{name}_recall'] for name in class_names]
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        
        metrics['overall_accuracy'] = overall_accuracy
        metrics['weighted_f1'] = weighted_f1
        metrics['f1_macro'] = macro_f1
        metrics['precision_macro'] = macro_precision
        metrics['recall_macro'] = macro_recall
        
        return metrics
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'variant': self.variant
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoints_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch}")
        
        logger.debug(f"Saved checkpoint: {checkpoint_path}")
    
    def create_plots(self):
        """Create training plots."""
        if not self.train_losses or not self.val_losses:
            return
        
        # Use matplotlib with cross-platform backend
        plt.switch_backend('Agg')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        if self.val_metrics:
            val_accuracies = [m.get('overall_accuracy', 0) for m in self.val_metrics]
            axes[0, 1].plot(epochs, val_accuracies, 'g-', label='Val Accuracy')
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # F1 Score plot
        if self.val_metrics:
            val_f1s = [m.get('weighted_f1', 0) for m in self.val_metrics]
            axes[1, 0].plot(epochs, val_f1s, 'm-', label='Weighted F1')
            axes[1, 0].set_title('Validation F1 Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Per-class metrics
        if self.val_metrics:
            class_names = ['damage', 'occlusion', 'crop']
            colors = ['red', 'blue', 'green']
            
            for i, (class_name, color) in enumerate(zip(class_names, colors)):
                f1_scores = [m.get(f'{class_name}_f1', 0) for m in self.val_metrics]
                axes[1, 1].plot(epochs, f1_scores, color=color, label=f'{class_name.capitalize()} F1')
            
            axes[1, 1].set_title('Per-Class F1 Scores')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f"training_curves_{self.variant}.png"
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training plots: {plot_path}")
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.variant}")
        
        # Create model and training components
        self.model = self.create_model()
        train_loader, val_loader, test_loader = self.create_datasets()
        self.optimizer = self.create_optimizer(self.model)
        self.scheduler = self.create_scheduler(self.optimizer, train_loader)
        self.criterion = self.create_criterion()
        
        # Setup mixed precision if enabled
        if self.config['training'].get('mixed_precision', False) and self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
            logger.info("Enabled mixed precision training")
        
        # Setup TensorBoard logging
        writer = SummaryWriter(self.logs_dir)
        
        # Early stopping configuration
        early_stopping_patience = self.config['training'].get('early_stopping_patience', 10)
        early_stopping_counter = 0
        
        # Training loop
        num_epochs = int(self.config['training']['num_epochs'])
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Check for shutdown request
            if shutdown_requested:
                logger.info("Shutdown requested, stopping training gracefully...")
                break
                
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch(self.model, train_loader, self.optimizer, self.criterion, epoch)
            
            # Validate epoch
            val_loss, val_metrics = self.validate_epoch(self.model, val_loader, self.criterion, epoch)
            
            # Update scheduler if it's epoch-based
            if self.scheduler and not hasattr(self.scheduler, '_step_count'):
                self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Log to TensorBoard
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Accuracy/Val', val_metrics['overall_accuracy'], epoch)
            writer.add_scalar('F1/Val', val_metrics['weighted_f1'], epoch)
            writer.add_scalar('F1_Macro/Val', val_metrics['f1_macro'], epoch)
            writer.add_scalar('Precision_Macro/Val', val_metrics['precision_macro'], epoch)
            writer.add_scalar('Recall_Macro/Val', val_metrics['recall_macro'], epoch)
            writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Check for best model using configurable metric
            best_metric_name = self.config['logging'].get('best_metric', 'weighted_f1')
            current_metric = val_metrics.get(best_metric_name, val_metrics['weighted_f1'])
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(self.model, self.optimizer, epoch, val_metrics, is_best)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Val Accuracy: {val_metrics['overall_accuracy']:.4f}")
            logger.info(f"  Val F1 (weighted): {val_metrics['weighted_f1']:.4f}")
            logger.info(f"  Val F1 (macro): {val_metrics['f1_macro']:.4f}")
            logger.info(f"  Best {best_metric_name}: {self.best_metric:.4f} (epoch {self.best_epoch+1})")
            
            # Early stopping check
            if early_stopping_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
                break
                
            # Check for shutdown request after epoch
            if shutdown_requested:
                logger.info("Shutdown requested, saving final checkpoint...")
                self.save_checkpoint(self.model, self.optimizer, epoch, val_metrics, is_best=False)
                break
        
        # Training completed
        total_time = time.time() - start_time
        best_metric_name = self.config['logging'].get('best_metric', 'weighted_f1')
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
        logger.info(f"Best {best_metric_name}: {self.best_metric:.4f} at epoch {self.best_epoch+1}")
        
        # Create final plots
        self.create_plots()
        
        # Save training summary
        self.save_training_summary(total_time)
        
        writer.close()
        
        return {
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'total_time': total_time,
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
            'final_val_loss': self.val_losses[-1] if self.val_losses else 0
        }
    
    def save_training_summary(self, total_time: float):
        """Save training summary to file."""
        summary = {
            'variant': self.variant,
            'config': self.config,
            'training_results': {
                'best_metric': self.best_metric,
                'best_epoch': self.best_epoch,
                'total_epochs': len(self.train_losses),
                'total_time_hours': total_time / 3600,
                'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
                'final_val_loss': self.val_losses[-1] if self.val_losses else 0
            },
            'best_metrics': self.val_metrics[self.best_epoch] if self.best_epoch < len(self.val_metrics) else {},
            'platform_info': {
                'platform': self.platform_utils.platform_info['os'],
                'device': str(self.device),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        summary_path = self.results_dir / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved training summary: {summary_path}")


def main():
    """Main function for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train hybrid road distress classification model')
    parser.add_argument('--variant', type=str, default='model_a', 
                       choices=['model_a', 'model_b', 'model_c', 'model_d'],
                       help='Model variant to train')
    parser.add_argument('--config', type=str, default='config/base_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        trainer = HybridTrainer(args.config, args.variant)
        results = trainer.train()
        
        if shutdown_requested:
            print(f"\n🛑 Training interrupted for {args.variant}")
            print("Checkpoint saved. You can resume training later.")
        else:
            print(f"\n✅ Training completed for {args.variant}:")
            print(f"  Best metric: {results['best_metric']:.4f}")
            print(f"  Best epoch: {results['best_epoch']+1}")
            print(f"  Total time: {results['total_time']/3600:.2f} hours")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Training forcefully interrupted for {args.variant}")
        print("Some progress may be lost.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise

if __name__ == "__main__":
    main() 