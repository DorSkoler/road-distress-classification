import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F

from model import RoadDistressModel
from data_loader import RoadDistressDataset

class ModelTrainer:
    def __init__(self, model_config, train_config, aug_config, experiment_name):
        self.model_config = model_config
        self.train_config = train_config
        self.aug_config = aug_config
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create experiment directory
        self.experiment_dir = os.path.join('experiments', experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': []
        }
        
        # Save configurations
        self.save_config()
        
    def save_config(self):
        config = {
            'model_config': self.model_config,
            'train_config': self.train_config,
            'aug_config': self.aug_config
        }
        with open(os.path.join(self.experiment_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
            
    def train(self, train_loader, val_loader):
        # Initialize model with correct parameters
        model = RoadDistressModel(
            num_classes=self.model_config['num_classes'],
            pretrained=self.model_config['pretrained'],
            backbone_type=self.model_config['backbone_type']
        ).to(self.device)
        
        # Get optimizer and scheduler
        optimizer = model.get_optimizer(
            learning_rate=self.train_config['learning_rate'],
            weight_decay=self.train_config['weight_decay']
        )
        scheduler = model.get_scheduler(optimizer, self.train_config['num_epochs'])
        
        # Initialize training variables
        best_val_acc = 0
        patience_counter = 0
        scaler = GradScaler()
        
        for epoch in range(self.train_config['num_epochs']):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = torch.zeros(3).to(self.device)  # One for each task
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.train_config["num_epochs"]}')
            for batch_idx, (images, labels) in enumerate(progress_bar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(images)
                    loss = F.binary_cross_entropy_with_logits(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.train_config['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum(dim=0)
                
                train_acc = 100. * train_correct.mean().item() / train_total
                progress_bar.set_postfix({
                    'loss': f'{train_loss/(batch_idx+1):.3f}',
                    'acc': f'{train_acc:.2f}%'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = torch.zeros(3).to(self.device)
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = F.binary_cross_entropy_with_logits(outputs, labels)
                    
                    val_loss += loss.item()
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum(dim=0)
            
            # Calculate metrics
            train_acc = 100. * train_correct.mean().item() / train_total
            val_acc = 100. * val_correct.mean().item() / val_total
            
            # Update learning rate
            scheduler.step()
            
            # Log metrics
            self.metrics['train_loss'].append(train_loss / len(train_loader))
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_loss'].append(val_loss / len(val_loader))
            self.metrics['val_acc'].append(val_acc)
            self.metrics['learning_rate'].append(scheduler.get_last_lr()[0])
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(self.experiment_dir, 'best_model.pth'))
                
                # Save per-task accuracies
                task_accuracies = {
                    'damage_acc': 100. * val_correct[0].item() / val_total,
                    'occlusion_acc': 100. * val_correct[1].item() / val_total,
                    'crop_acc': 100. * val_correct[2].item() / val_total
                }
                with open(os.path.join(self.experiment_dir, 'best_task_accuracies.json'), 'w') as f:
                    json.dump(task_accuracies, f, indent=4)
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.train_config['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            print(f'\nEpoch {epoch+1}:')
            print(f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            print(f'Per-task Val Acc: Damage={100.*val_correct[0].item()/val_total:.2f}%, ' +
                  f'Occlusion={100.*val_correct[1].item()/val_total:.2f}%, ' +
                  f'Crop={100.*val_correct[2].item()/val_total:.2f}%')
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            
        # Save final metrics
        self.save_metrics()
        return best_val_acc

    def save_metrics(self):
        with open(os.path.join(self.experiment_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)

def main():
    # Model configurations
    model_configs = [
        {
            'num_classes': 3,
            'pretrained': True,
            'backbone_type': 'efficientnet_b3'
        },
        {
            'num_classes': 3,
            'pretrained': True,
            'backbone_type': 'resnet50'
        }
    ]
    
    # Training configurations
    training_config = {
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'gradient_clip': 1.0,
        'early_stopping_patience': 5
    }
    
    # Augmentation configurations
    augmentation_config = {
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
    
    # Create datasets
    train_dataset = RoadDistressDataset(
        data_dir='organized_dataset',
        split='train',
        transform_type='train'
    )
    val_dataset = RoadDistressDataset(
        data_dir='organized_dataset',
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
    
    # Train each model
    results = []
    for model_config in model_configs:
        experiment_name = f"{model_config['backbone_type']}"
        print(f"\nTraining {experiment_name}...")
        
        trainer = ModelTrainer(
            model_config=model_config,
            train_config=training_config,
            aug_config=augmentation_config,
            experiment_name=experiment_name
        )
        
        best_val_acc = trainer.train(train_loader, val_loader)
        results.append({
            'model': model_config['backbone_type'],
            'best_val_acc': best_val_acc
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('experiments/training_results.csv', index=False)
    print("\nTraining results saved to experiments/training_results.csv")

if __name__ == '__main__':
    main() 