#!/usr/bin/env python3
"""
Training Script for Model E: CLAHE Enhanced Images with Full Mask Overlay

Model E Configuration:
- CLAHE enhancement using optimized parameters from JSON
- Mask overlay at 1.0 opacity (full overlay)
- Mask generation on-the-fly if not exists
- Cleanup preprocessing data between epochs to save storage
- No data augmentation
- Multi-label classification (damage, occlusion, crop)
"""

import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import argparse
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests
import urllib.parse

def download_model_from_gdrive(file_id: str, output_path: str) -> bool:
    """
    Download model file from Google Drive
    
    Args:
        file_id: Google Drive file ID
        output_path: Local path to save the file
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Google Drive download URL
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        print(f"Downloading model from Google Drive...")
        print(f"URL: {url}")
        print(f"Saving to: {output_path}")
        
        # Start download
        response = requests.get(url, stream=True)
        
        # Check if we need to handle the virus scan warning
        if 'virus scan warning' in response.text.lower():
            # Extract the confirm token
            for line in response.text.split('\n'):
                if 'confirm=' in line:
                    confirm_token = line.split('confirm=')[1].split('&')[0]
                    break
            else:
                confirm_token = None
            
            if confirm_token:
                # Retry with confirm token
                confirm_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                response = requests.get(confirm_url, stream=True)
        
        # Check if download was successful
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # No content-length header, download without progress bar
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            print(f"✓ Model downloaded successfully to {output_path}")
            return True
        else:
            print(f"✗ Download failed with status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False


class RoadMaskGenerator:
    """Road mask generator using segmentation model."""
    
    def __init__(self, model_path: str = "checkpoints/best_model.pth"):
        """Initialize the road mask generator."""
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Download model if it doesn't exist
        if not self.model_path.exists():
            print(f"Model not found at {self.model_path}")
            print("Downloading from Google Drive...")
            
            # Google Drive file ID from the URL
            file_id = "1w-55rSswB74tAsFwsv1xxdCXl0qNgV25"
            
            success = download_model_from_gdrive(file_id, str(self.model_path))
            if not success:
                print("Failed to download model. Using ImageNet weights only.")
        
        self.model = self._load_model()
        
    def _load_model(self) -> nn.Module:
        """Load the road segmentation model."""
        # Create U-Net model
        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            classes=1,
            activation=None
        )
        
        # Load weights if available
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(checkpoint)
                print(f"✓ Loaded mask generation model from {self.model_path}")
            except Exception as e:
                print(f"Warning: Could not load model weights: {e}")
                print("Using model with ImageNet weights only")
        else:
            print(f"Warning: Model file not found at {self.model_path}")
            print("Using model with ImageNet weights only")
        
        model.to(self.device)
        model.eval()
        return model
    
    def generate_mask(self, image: np.ndarray, confidence_threshold: float = 0.5) -> np.ndarray:
        """Generate road mask for given image."""
        # Preprocess image
        original_size = (image.shape[0], image.shape[1])
        image_resized = cv2.resize(image, (512, 512))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Generate mask
        with torch.no_grad():
            output = self.model(image_tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
            
        # Apply threshold and resize to original size
        mask_binary = (mask > confidence_threshold).astype(np.uint8)
        mask_resized = cv2.resize(mask_binary.astype(np.float32), (original_size[1], original_size[0]))
        mask_final = (mask_resized > 0.5).astype(np.uint8)
        
        return mask_final


class CLAHEDatasetWithMaskGeneration(Dataset):
    """Dataset that applies CLAHE and generates masks on-the-fly"""
    
    def __init__(
        self, 
        images_dir: str,
        labels_file: str,
        clahe_params_json: str,
        mask_opacity: float = 1.0,
        img_size: int = 256,
        temp_dir: str = "temp_preprocessing",
        transform=None
    ):
        """
        Args:
            images_dir: Directory containing original images
            labels_file: Path to labels CSV file
            clahe_params_json: Path to CLAHE parameters JSON
            mask_opacity: Opacity for mask overlay (0.0-1.0)
            img_size: Target image size
            temp_dir: Temporary directory for preprocessing data
            transform: Additional transforms (applied after CLAHE)
        """
        self.images_dir = Path(images_dir)
        self.mask_opacity = mask_opacity
        self.img_size = img_size
        self.temp_dir = Path(temp_dir)
        self.transform = transform
        
        # Create temp directory
        self.temp_dir.mkdir(exist_ok=True)
        
        # Load labels
        self.labels_df = pd.read_csv(labels_file)
        
        # Load CLAHE parameters
        self.clahe_params = self.load_clahe_params(clahe_params_json)
        
        # Initialize mask generator
        self.mask_generator = RoadMaskGenerator()
        
        # Filter samples to only include those with CLAHE parameters
        self.samples = self.create_sample_list()
        
        print(f"CLAHEDatasetWithMaskGeneration initialized:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Mask opacity: {mask_opacity}")
        print(f"  CLAHE params loaded: {len(self.clahe_params)}")
        print(f"  Temp directory: {self.temp_dir}")
    
    def load_clahe_params(self, json_path: str) -> Dict[str, Dict]:
        """Load CLAHE parameters from JSON"""
        params = {}
        
        if not os.path.exists(json_path):
            print(f"Warning: CLAHE parameters file not found: {json_path}")
            print("Using default CLAHE parameters for all images")
            return params
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Convert JSON data to expected format
            for image_path, param_data in data.items():
                tile_grid_size = param_data['tile_grid_size']
                params[image_path] = {
                    'clip_limit': param_data['clip_limit'],
                    'tile_grid_x': tile_grid_size[0],
                    'tile_grid_y': tile_grid_size[1]
                }
            
        except Exception as e:
            print(f"Error loading CLAHE parameters from {json_path}: {str(e)}")
            print("Using default CLAHE parameters for all images")
            return {}
        
        return params
    
    def create_sample_list(self) -> List[Dict]:
        """Create list of valid samples with labels and CLAHE parameters"""
        samples = []
        
        for _, row in self.labels_df.iterrows():
            image_name = row['image_name']
            
            # Check if image file exists
            image_path = self.images_dir / image_name
            if not image_path.exists():
                continue
            
            # Get CLAHE parameters (use defaults if not found)
            relative_path = str(image_path.relative_to(self.images_dir.parent))
            clahe_params = self.clahe_params.get(relative_path, {
                'clip_limit': 3.0,
                'tile_grid_x': 8,
                'tile_grid_y': 8
            })
            
            sample = {
                'image_path': image_path,
                'image_name': image_name,
                'label': {
                    'damage': int(row.get('damage', 0)),
                    'occlusion': int(row.get('occlusion', 0)),
                    'crop': int(row.get('crop', 0))
                },
                'clahe_params': clahe_params
            }
            samples.append(sample)
        
        return samples
    
    def apply_clahe(self, image: np.ndarray, clahe_params: Dict) -> np.ndarray:
        """Apply CLAHE with specified parameters"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=clahe_params['clip_limit'],
            tileGridSize=(clahe_params['tile_grid_x'], clahe_params['tile_grid_y'])
        )
        enhanced_l = clahe.apply(l_channel)
        
        # Reconstruct image
        lab[:, :, 0] = enhanced_l
        enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr
    
    def apply_mask_overlay(self, image: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
        """Apply mask overlay with specified opacity"""
        if opacity == 0.0:
            return image
        
        # Ensure mask is 3-channel
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Normalize mask to 0-255 range
        mask_normalized = (mask * 255).astype(np.uint8)
        
        # Apply overlay with opacity
        overlay = cv2.addWeighted(image, 1.0 - opacity, mask_normalized, opacity, 0)
        return overlay
    
    def get_processed_image_path(self, image_name: str) -> Path:
        """Get path for processed image in temp directory"""
        return self.temp_dir / f"{image_name}_{self.mask_opacity}.png"
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        # Check if processed image exists
        processed_path = self.get_processed_image_path(sample['image_name'])
        
        if processed_path.exists():
            # Load preprocessed image
            enhanced_image = cv2.imread(str(processed_path))
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
        else:
            # Load original image
            image = cv2.imread(str(sample['image_path']))
            if image is None:
                raise ValueError(f"Could not load image: {sample['image_path']}")
            
            # Resize image
            image = cv2.resize(image, (self.img_size, self.img_size))
            
            # Apply CLAHE enhancement
            enhanced_image = self.apply_clahe(image, sample['clahe_params'])
            
            # Generate and apply mask if opacity > 0
            if self.mask_opacity > 0.0:
                mask = self.mask_generator.generate_mask(enhanced_image)
                enhanced_image = self.apply_mask_overlay(enhanced_image, mask, self.mask_opacity)
            
            # Convert to RGB
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
            
            # Save processed image for reuse within epoch
            cv2.imwrite(str(processed_path), cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
        
        # Convert to PIL for transforms
        pil_image = Image.fromarray(enhanced_image)
        
        # Apply additional transforms
        if self.transform:
            pil_image = self.transform(pil_image)
        else:
            # Default transform: to tensor and normalize
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pil_image = transform(pil_image)
        
        # Create multi-label target
        target = torch.tensor([
            sample['label']['damage'],
            sample['label']['occlusion'],
            sample['label']['crop']
        ], dtype=torch.float32)
        
        return pil_image, target
    
    def cleanup_temp_data(self):
        """Clean up temporary preprocessing data"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(exist_ok=True)
            print(f"✓ Cleaned up temp directory: {self.temp_dir}")
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get sample information for debugging"""
        sample = self.samples[idx]
        return {
            'image_path': str(sample['image_path']),
            'image_name': sample['image_name'],
            'clahe_params': sample['clahe_params'],
            'label': sample['label']
        }


class RoadDistressModelE(nn.Module):
    """Model E: CLAHE + Full Mask Integration for Road Distress Classification"""
    
    def __init__(self, num_classes=3, backbone='efficientnet-b3'):
        super(RoadDistressModelE, self).__init__()
        
        # Use EfficientNet backbone
        if backbone == 'efficientnet-b3':
            self.backbone = models.efficientnet_b3(pretrained=True)
            # Get the feature dimension from the original classifier
            backbone_features = self.backbone.classifier[1].in_features
            # Remove the final classification layer
            self.backbone.classifier = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            backbone_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
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


class ModelETrainer:
    """Trainer for Model E with integrated preprocessing and cleanup"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: CLAHEDatasetWithMaskGeneration,
        val_dataset: CLAHEDatasetWithMaskGeneration,
        device: torch.device,
        batch_size: int = 32,
        num_workers: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        output_dir: str = 'experiments/model_e'
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
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
            steps_per_epoch=len(self.train_loader),
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
    
    def cleanup_between_epochs(self):
        """Clean up preprocessing data between epochs"""
        print("Cleaning up preprocessing data...")
        self.train_dataset.cleanup_temp_data()
        self.val_dataset.cleanup_temp_data()
        print("✓ Preprocessing data cleaned up")
    
    def train(self, num_epochs: int = 50, early_stopping_patience: int = 10):
        """Complete training loop with preprocessing cleanup"""
        print(f"Starting Model E training with integrated preprocessing...")
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
            print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
            
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
            
            # Clean up preprocessing data between epochs
            self.cleanup_between_epochs()
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch+1} epochs (patience: {early_stopping_patience})")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        
        # Save training summary
        summary = {
            'model_type': 'Model E (CLAHE + Full Masks + Integrated Preprocessing)',
            'best_val_accuracy': float(best_val_accuracy),
            'total_epochs': epoch + 1,
            'final_train_loss': float(self.train_losses[-1]),
            'final_val_loss': float(self.val_losses[-1]),
            'training_completed': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.writer.close()
        
        # Final cleanup
        self.cleanup_between_epochs()


def main():
    parser = argparse.ArgumentParser(description='Train Model E: CLAHE + Full Mask Overlay with Integrated Preprocessing')
    parser.add_argument('--train-images', required=True, help='Training images directory')
    parser.add_argument('--val-images', required=True, help='Validation images directory')
    parser.add_argument('--train-labels', required=True, help='Training labels CSV')
    parser.add_argument('--val-labels', required=True, help='Validation labels CSV')
    parser.add_argument('--clahe-params', required=True, help='CLAHE parameters JSON')
    parser.add_argument('--mask-model', default='experiments/2025-06-28_smart_split_training/../../checkpoints/best_model.pth', help='Road segmentation model path')
    parser.add_argument('--output-dir', default='experiments/model_e', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--backbone', default='efficientnet-b3', choices=['efficientnet-b3', 'resnet50'])
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--img-size', type=int, default=256, help='Input image size')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets with integrated preprocessing
    train_dataset = CLAHEDatasetWithMaskGeneration(
        images_dir=args.train_images,
        labels_file=args.train_labels,
        clahe_params_json=args.clahe_params,
        mask_opacity=1.0,  # Model E uses full opacity
        img_size=args.img_size,
        temp_dir="temp_preprocessing_train"
    )
    
    val_dataset = CLAHEDatasetWithMaskGeneration(
        images_dir=args.val_images,
        labels_file=args.val_labels,
        clahe_params_json=args.clahe_params,
        mask_opacity=1.0,  # Model E uses full opacity
        img_size=args.img_size,
        temp_dir="temp_preprocessing_val"
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Model setup
    model = RoadDistressModelE(num_classes=3, backbone=args.backbone)
    print(f"Model: {args.backbone} backbone with CLAHE + Full Mask integration")
    
    # Trainer setup
    trainer = ModelETrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir
    )
    
    # Start training
    trainer.train(num_epochs=args.epochs, early_stopping_patience=10)


if __name__ == '__main__':
    main() 