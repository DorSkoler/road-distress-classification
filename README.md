# Road Distress Classification Project

## Project Overview
This project implements a road distress classification system using deep learning. The system uses road masks to focus on relevant areas during training and inference, improving classification accuracy for three main classes: Damage, Occlusion, and Crop.

## Data Preprocessing Pipeline

### 1. Road Mask Generation
- Used U-Net with ResNet34 backbone for road segmentation
- Training configuration:
  - Batch size: 4
  - Learning rate: 1e-4
  - Epochs: 20
  - Loss: Combined Dice + BCE loss
  - Image size: 256x256
- Model saved in `checkpoints/best_model.pth`

### 2. Road Isolation
- Implemented in `preprocessing/isolate_road.py`
- Features:
  - Multiple segmentation methods (U-Net, DeepLabV3, DeepLabV3+, SegFormer)
  - Classical CV fallback for robustness
  - Advanced polygon-based segmentation
  - Mask refinement with morphological operations
  - Image preprocessing with CLAHE and bilateral filtering

### 3. Data Organization
- Images: `filtered/` directory
- Masks: `filtered_masks/` directory
- JSON annotations: `tagged_json/` directory
- Data split into train/val/test sets

## Model Architecture

### Road Distress Model with Masks
```python
class RoadDistressModelWithMasks(nn.Module):
    def __init__(self, num_classes=3):
        self.backbone = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=num_classes,
            activation=None
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
```

Key features:
- EfficientNet-B3 backbone (12M parameters)
- Uses road masks to focus on relevant areas
- Multi-label classification
- Binary cross-entropy loss

## Training Process

### Configuration
- Batch size: 64
- Learning rate: 1e-3 with warmup
- Optimizer: AdamW
- Scheduler: OneCycleLR
- Mixed precision training
- Early stopping (patience=10)
- Weight decay: 0.01
- Gradient clipping: 1.0

### Training Features
- Uses road masks to focus on relevant areas
- Multi-label classification
- Validation after each epoch
- Saves best model based on validation accuracy
- Tracks training and validation metrics

## Evaluation Results

### Previous Model (Without Masks)
- Overall accuracy: ~81.35%
- Less robust to background variations
- More prone to false positives

### Current Model (With Masks)
Overall Accuracy: 88.99%

1. Damage Detection:
   - Accuracy: 73.60%
   - Precision: 82.61%
   - Recall: 28.30%

2. Occlusion Detection:
   - Accuracy: 93.96%
   - Precision: 86.23%
   - Recall: 79.33%

3. Crop Detection:
   - Accuracy: 99.43%
   - Precision: 99.21%
   - Recall: 88.73%

Key Observations:
- Overall accuracy improved by ~7.64% with masks
- Crop detection shows best performance
- Damage detection has lowest recall, suggesting missed detections
- Occlusion detection shows balanced precision-recall tradeoff

## Next Steps for Model Improvement

1. Hyperparameter Optimization
   - Learning rate search (current: 1e-3)
   - Batch size optimization (current: 64)
   - Test different optimizers
   - Experiment with loss functions

2. Data Augmentation
   - Add weather-specific augmentations
   - Implement task-specific augmentations
   - Test different augmentation parameters

3. Model Architecture
   - Test attention mechanisms
   - Experiment with transformer-based architectures
   - Try ensemble methods

4. Training Strategies
   - Implement curriculum learning
   - Test different batch sizes
   - Experiment with mixup/cutmix

5. Evaluation
   - Conduct detailed error analysis
   - Implement model interpretability
   - Study failure cases

## Project Structure
```
road-distress-classification/
├── filtered/              # Original images
├── filtered_masks/        # Generated road masks
├── tagged_json/          # JSON annotations
├── experiments/          # Training logs and checkpoints
├── evaluation_results/   # Model evaluation results
├── preprocessing/        # Road segmentation code
├── checkpoints/         # Model checkpoints
└── road_dataset_pipeline/
    └── scripts/
        ├── train_with_masks.py
        └── evaluate_model.py
```

## Usage

1. Training:
```bash
python train_with_masks.py
```

2. Evaluation:
```bash
python evaluate_model.py
```

## Dependencies
- PyTorch
- segmentation-models-pytorch
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- scikit-learn 