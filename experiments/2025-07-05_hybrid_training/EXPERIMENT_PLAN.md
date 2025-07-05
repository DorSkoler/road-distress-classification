# 2025-07-05 Hybrid Training Experiment Plan

## Overview

This experiment combines the best aspects of both previous experiments:
- **Data Split Strategy**: From 2025-06-28 (smart split training)
- **Model Architecture**: From 2025-05-10 (successful UNet + EfficientNet-B3)
- **New Variants**: 4 different training approaches to test various input combinations

## Experiment Goals

1. Use the proven smart data splitting approach from 28/06
2. Apply the successful UNet-based architecture from 10/05
3. Compare 4 different training strategies systematically
4. Evaluate the impact of masks and augmentation independently and combined

## Model Variants

| Model | Description | Input Components | Masking Strategy |
|-------|-------------|------------------|------------------|
| **Model A** | Pictures + Masks | Original images + road masks | Full masking (zero non-road) |
| **Model B** | Pictures + Augmentation | Original + augmented images | No masking |
| **Model C** | Pictures + Augmentation + Masks | Original + augmented images + masks | Full masking |
| **Model D** | Pictures + Augmentation + 50% Masks | Original + augmented images + masks | Weighted masking (50% weight to non-road) |

## Step-by-Step Implementation Plan

### Phase 1: Project Setup and Data Preparation

#### Step 1: Create Experiment Directory Structure
```
road-distress-classification/experiments/2025-07-05_hybrid_training/
├── README.md
├── EXPERIMENT_PLAN.md
├── config/
│   ├── base_config.yaml
│   ├── model_a_config.yaml
│   ├── model_b_config.yaml
│   ├── model_c_config.yaml
│   └── model_d_config.yaml
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hybrid_model.py
│   │   └── model_variants.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── smart_splitter.py
│   │   ├── mask_generator.py
│   │   ├── augmentation.py
│   │   └── dataset.py
│   └── training/
│       ├── __init__.py
│       ├── trainer.py
│       └── comparative_trainer.py
├── scripts/
│   ├── prepare_data.py
│   ├── train_model.py
│   └── evaluate_models.py
├── results/
│   ├── model_a/
│   ├── model_b/
│   ├── model_c/
│   └── model_d/
└── data/
    ├── splits/
    ├── masks/
    └── augmented/
```

#### Step 2: Copy and Adapt Smart Data Splitting
- Copy `smart_data_splitter.py` from 28/06 experiment
- Adapt it to work with the new directory structure
- Ensure road-wise splitting is preserved

#### Step 3: Copy and Adapt Mask Generation
- Copy `road_mask_generator.py` from 28/06 experiment
- Adapt mask generation to work with the new pipeline
- Ensure masks are compatible with the UNet architecture

#### Step 4: Create Augmentation Pipeline
- Copy augmentation logic from 28/06 experiment
- Make it compatible with the UNet-based training
- Ensure augmented images maintain quality

### Phase 2: Model Architecture Implementation

#### Step 5: Adapt UNet Architecture from 10/05
```python
# src/models/hybrid_model.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class HybridRoadDistressModel(nn.Module):
    def __init__(self, num_classes=3, use_masks=True, mask_weight=1.0):
        super().__init__()
        self.use_masks = use_masks
        self.mask_weight = mask_weight  # 1.0 for full masking, 0.5 for weighted
        
        # UNet with EfficientNet-B3 backbone (from 10/05 success)
        self.backbone = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=num_classes,
            activation=None
        )
        
        # Classification head (from 10/05 success)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, mask=None):
        # Get features from UNet backbone
        features = self.backbone(x)  # [B, num_classes, H, W]
        
        if self.use_masks and mask is not None:
            # Apply masking strategy
            if self.mask_weight == 1.0:
                # Full masking (Model A & C)
                mask_expanded = mask.expand_as(features)
                masked_features = features * mask_expanded
            else:
                # Weighted masking (Model D)
                mask_expanded = mask.expand_as(features)
                # Non-road pixels get reduced weight, road pixels get full weight
                weighted_mask = mask_expanded + (1 - mask_expanded) * self.mask_weight
                masked_features = features * weighted_mask
        else:
            # No masking (Model B)
            masked_features = features
        
        # Classification
        return self.classifier(masked_features)
```

#### Step 6: Create Model Variants
```python
# src/models/model_variants.py
from .hybrid_model import HybridRoadDistressModel

def create_model_variant(variant_name: str) -> HybridRoadDistressModel:
    """Create specific model variant"""
    if variant_name == 'model_a':
        # Pictures + Masks
        return HybridRoadDistressModel(use_masks=True, mask_weight=1.0)
    elif variant_name == 'model_b':
        # Pictures + Augmentation (no masks)
        return HybridRoadDistressModel(use_masks=False, mask_weight=0.0)
    elif variant_name == 'model_c':
        # Pictures + Augmentation + Masks
        return HybridRoadDistressModel(use_masks=True, mask_weight=1.0)
    elif variant_name == 'model_d':
        # Pictures + Augmentation + 50% Masks
        return HybridRoadDistressModel(use_masks=True, mask_weight=0.5)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")
```

### Phase 3: Data Pipeline Implementation

#### Step 7: Create Dataset Class
```python
# src/data/dataset.py
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import json
import os

class HybridRoadDataset(Dataset):
    def __init__(self, image_paths, mask_paths, label_paths, 
                 use_masks=True, use_augmentation=False, 
                 augmented_paths=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.label_paths = label_paths
        self.use_masks = use_masks
        self.use_augmentation = use_augmentation
        self.augmented_paths = augmented_paths if use_augmentation else None
        self.transform = transform
        
        # Combine original and augmented if needed
        if self.use_augmentation and self.augmented_paths:
            self.all_image_paths = self.image_paths + self.augmented_paths
            self.all_mask_paths = self.mask_paths + self.augmented_paths  # Corresponding masks
        else:
            self.all_image_paths = self.image_paths
            self.all_mask_paths = self.mask_paths
    
    def __len__(self):
        return len(self.all_image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.all_image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Load mask if needed
        mask = None
        if self.use_masks:
            mask_path = self.all_mask_paths[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        # Load labels (use original labels even for augmented images)
        original_idx = idx if idx < len(self.image_paths) else idx - len(self.image_paths)
        label_path = self.label_paths[original_idx]
        with open(label_path, 'r') as f:
            img_data = json.load(f)
        
        # Extract labels (same as 10/05 approach)
        labels = torch.zeros(3)
        for tag in img_data.get('tags', []):
            if tag['name'] == 'Damage':
                labels[0] = 1 if tag['value'] == 'Damaged' else 0
            elif tag['name'] == 'Occlusion':
                labels[1] = 1 if tag['value'] == 'Occluded' else 0
            elif tag['name'] == 'Crop':
                labels[2] = 1 if tag['value'] == 'Cropped' else 0
        
        if self.use_masks:
            return image, mask, labels
        else:
            return image, labels
```

### Phase 4: Training Pipeline Implementation

#### Step 8: Create Training Configuration
```yaml
# config/base_config.yaml
experiment:
  name: "hybrid_training_2025_07_05"
  version: "1.0"
  
dataset:
  coryell_path: "../../data/coryell"
  image_size: [256, 256]
  batch_size: 64  # Same as successful 10/05 experiment
  num_workers: 8
  
# Use smart splitting from 28/06
splitting:
  method: "smart_split"  # From 28/06 success
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  preserve_road_integrity: true
  
# Use successful training config from 10/05
training:
  num_epochs: 50
  learning_rate: 1e-3
  weight_decay: 0.02
  optimizer: "AdamW"
  scheduler: "OneCycleLR"
  warmup_pct: 0.3
  gradient_clip: 1.0
  mixed_precision: true
  early_stopping_patience: 10
  
# Model config (adapted from 10/05)
model:
  backbone: "efficientnet-b3"
  num_classes: 3
  activation: null
  
# Augmentation config (from 28/06 but conservative)
augmentation:
  samples_per_image: 3
  geometric:
    rotation:
      enabled: true
      range: [-5, 5]
      probability: 0.3
    flip:
      enabled: true
      probability: 0.5
  color:
    brightness:
      enabled: true
      range: [-0.1, 0.1]
      probability: 0.5
    contrast:
      enabled: true
      range: [-0.1, 0.1]
      probability: 0.5
```

#### Step 9: Create Comparative Training Pipeline
```python
# src/training/comparative_trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from datetime import datetime

class ComparativeTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def train_all_variants(self):
        """Train all 4 model variants"""
        variants = ['model_a', 'model_b', 'model_c', 'model_d']
        results = {}
        
        for variant in variants:
            print(f"\n{'='*50}")
            print(f"Training {variant.upper()}")
            print(f"{'='*50}")
            
            # Create model
            model = create_model_variant(variant)
            
            # Create dataset
            train_dataset = self._create_dataset(variant, 'train')
            val_dataset = self._create_dataset(variant, 'val')
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, 
                                    batch_size=self.config['dataset']['batch_size'],
                                    shuffle=True, 
                                    num_workers=self.config['dataset']['num_workers'])
            val_loader = DataLoader(val_dataset, 
                                  batch_size=self.config['dataset']['batch_size'],
                                  shuffle=False, 
                                  num_workers=self.config['dataset']['num_workers'])
            
            # Train model
            variant_results = self._train_variant(model, train_loader, val_loader, variant)
            results[variant] = variant_results
            
        return results
    
    def _create_dataset(self, variant, split):
        """Create dataset for specific variant and split"""
        # Load split information
        split_dir = f"data/splits/{split}"
        
        # Determine dataset configuration based on variant
        if variant == 'model_a':
            # Pictures + Masks
            use_masks = True
            use_augmentation = False
        elif variant == 'model_b':
            # Pictures + Augmentation
            use_masks = False
            use_augmentation = True
        elif variant == 'model_c':
            # Pictures + Augmentation + Masks
            use_masks = True
            use_augmentation = True
        elif variant == 'model_d':
            # Pictures + Augmentation + 50% Masks
            use_masks = True
            use_augmentation = True
        
        # Create dataset
        return HybridRoadDataset(
            image_paths=self._load_image_paths(split_dir),
            mask_paths=self._load_mask_paths(split_dir) if use_masks else None,
            label_paths=self._load_label_paths(split_dir),
            use_masks=use_masks,
            use_augmentation=use_augmentation,
            augmented_paths=self._load_augmented_paths(split_dir) if use_augmentation else None
        )
```

### Phase 5: Evaluation and Comparison

#### Step 10: Create Evaluation Framework
```python
# src/training/evaluator.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, models_dict, test_loader):
        self.models = models_dict
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def evaluate_all_models(self):
        """Evaluate all model variants"""
        results = {}
        
        for variant_name, model in self.models.items():
            print(f"\nEvaluating {variant_name}...")
            model.eval()
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(self.test_loader):
                    if len(batch) == 3:  # With masks
                        images, masks, labels = batch
                        images, masks, labels = images.to(self.device), masks.to(self.device), labels.to(self.device)
                        outputs = model(images, masks)
                    else:  # Without masks
                        images, labels = batch
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = model(images)
                    
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())
            
            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
            
            results[variant_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': all_preds,
                'labels': all_labels
            }
        
        return results
    
    def create_comparison_plots(self, results):
        """Create comparison plots for all models"""
        # Performance comparison
        variants = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results[variant][metric] for variant in variants]
            axes[i].bar(variants, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
```

## Implementation Schedule

### Week 1: Setup and Data Preparation
- [ ] Create directory structure
- [ ] Copy and adapt smart splitting from 28/06
- [ ] Copy and adapt mask generation from 28/06
- [ ] Test data pipeline

### Week 2: Model Development
- [ ] Implement hybrid model architecture
- [ ] Create model variants
- [ ] Implement weighted masking for Model D
- [ ] Test model forward passes

### Week 3: Training Pipeline
- [ ] Create training pipeline
- [ ] Implement comparative trainer
- [ ] Create configuration files
- [ ] Test training on small dataset

### Week 4: Full Training and Evaluation
- [ ] Train all 4 model variants
- [ ] Evaluate and compare results
- [ ] Create visualizations and reports
- [ ] Document findings

## Expected Outcomes

1. **Model A**: Baseline with masks, should perform well on road-focused classification
2. **Model B**: Test augmentation effectiveness without masks
3. **Model C**: Best of both worlds - augmentation + masks
4. **Model D**: Novel weighted masking approach - may provide best balance

## Success Metrics

- **Accuracy**: >85% on test set
- **F1-Score**: >0.80 weighted average
- **Generalization**: Consistent performance across different road types
- **Efficiency**: Training time <6 hours per model on RTX 4070 Ti Super

This plan combines the proven data splitting strategy from 28/06 with the successful architecture from 10/05, while introducing systematic comparison of different input strategies. 