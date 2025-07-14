# CLAHE-Enhanced Models E, F, G, H

This document describes the new CLAHE-enhanced model variants that have been integrated into the existing trainer architecture.

## Overview

Four new model variants have been added to support CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing:

| Model | CLAHE | Masks | Mask Weight | Augmentation | Description |
|-------|-------|-------|-------------|--------------|-------------|
| Model E | ✅ | ✅ | 1.0 (Full) | ❌ | CLAHE + Full masks, no augmentation |
| Model F | ✅ | ✅ | 0.5 (Partial) | ❌ | CLAHE + Partial masks, no augmentation |
| Model G | ✅ | ✅ | 1.0 (Full) | ✅ | CLAHE + Full masks + augmentation |
| Model H | ✅ | ✅ | 0.5 (Partial) | ✅ | CLAHE + Partial masks + augmentation |

## Key Features

### ✨ **Preserved Trainer Architecture**
- **No changes required** to the existing trainer code
- Models integrate seamlessly through the existing factory pattern
- Configuration-driven approach maintains flexibility

### 🎯 **CLAHE Integration**
- Automatic loading of optimized CLAHE parameters from `clahe_params.json`
- Fallback to default parameters if specific image parameters not found
- Applied only to original images (not augmented versions)

### 🔧 **Model Variants**

#### **Model E: CLAHE + Full Masks**
```yaml
# Model E focuses on road surface with enhanced contrast
use_clahe: true
mask_weight: 1.0        # Full road masking
use_augmentation: false # No augmentation for baseline
```

#### **Model F: CLAHE + Partial Masks**
```yaml
# Model F learns from both road and non-road areas
use_clahe: true
mask_weight: 0.5        # Partial masking (50% non-road weight)
use_augmentation: false # No augmentation for baseline
```

#### **Model G: CLAHE + Full Masks + Augmentation**
```yaml
# Model G combines all techniques for robust training
use_clahe: true
mask_weight: 1.0        # Full road masking
use_augmentation: true  # Data augmentation enabled
```

#### **Model H: CLAHE + Partial Masks + Augmentation**
```yaml
# Model H balances road/non-road learning with robustness
use_clahe: true
mask_weight: 0.5        # Partial masking
use_augmentation: true  # Data augmentation enabled
```

## How to Use

### 🚀 **Quick Start**

```bash
# Train a single model
cd experiments/2025-07-05_hybrid_training
python run_model_training.py --model model_e

# Train all CLAHE models
python run_model_training.py --model all

# Train with custom config
python run_model_training.py --model model_f --config path/to/custom_config.yaml
```

### 📁 **Required Files**

1. **CLAHE Parameters**: `clahe_params.json` (should be in project root)
2. **Model Configs**: Located in `config/model_[e,f,g,h]_config.yaml`
3. **Training Data**: Existing Coryell dataset structure

### ⚙️ **Configuration Files**

Each model has its own configuration file:

- `config/model_e_config.yaml` - Model E settings
- `config/model_f_config.yaml` - Model F settings  
- `config/model_g_config.yaml` - Model G settings
- `config/model_h_config.yaml` - Model H settings

## Architecture Integration

### 🏗️ **Model Factory Pattern**

The new models integrate through the existing `ModelVariantFactory`:

```python
# Automatic model creation based on variant
model = create_model('model_e')  # Creates CLAHE + full masks model
model = create_model('model_f')  # Creates CLAHE + partial masks model
```

### 📊 **Dataset Integration**

The `HybridRoadDataset` automatically handles CLAHE preprocessing:

```python
# CLAHE applied automatically based on variant configuration
dataset = create_dataset('train', config, 'model_e')
# → Loads with CLAHE preprocessing and full masking
```

### 🔍 **CLAHE Processing Pipeline**

1. **Load Image**: Original image loaded from Coryell dataset
2. **Check Variant**: If model uses CLAHE (E, F, G, H), apply preprocessing
3. **Parameter Lookup**: Get optimized CLAHE parameters for specific image
4. **Apply CLAHE**: Enhance contrast using parameters
5. **Standard Pipeline**: Continue with normal preprocessing (RGB conversion, masking, etc.)

## Expected Performance

### 🎯 **Model Comparison Goals**

- **Model E vs A**: Test impact of CLAHE on full-masked training
- **Model F vs D**: Compare CLAHE enhancement with partial masking  
- **Model G vs C**: Evaluate CLAHE with augmentation and full masks
- **Model H vs D**: Test optimal combination of all techniques

### 📈 **Performance Metrics**

The trainer will log:
- Overall accuracy
- Per-class metrics (damage, occlusion, crop)
- Training convergence speed
- Loss curves and validation metrics

## CLAHE Parameters

### 📋 **Parameter Structure**

```json
{
  "coryell/Co Rd 246/img/000_31.471559_-97.717786.png": {
    "clip_limit": 3.0,
    "tile_grid_size": [8, 8]
  }
}
```

### 🔧 **Default Fallback**

If image-specific parameters aren't found:
```python
default_params = {
    'clip_limit': 3.0,
    'tile_grid_x': 8, 
    'tile_grid_y': 8
}
```

## Advanced Usage

### 🎛️ **Custom Training**

```python
from training.trainer import HybridTrainer

# Initialize trainer with specific config
trainer = HybridTrainer(
    config_path="config/model_e_config.yaml",
    variant="model_e"
)

# Start training
trainer.train()
```

### 📊 **Monitoring Training**

```bash
# View TensorBoard logs
tensorboard --logdir results/model_e/logs

# Check training progress
tail -f results/model_e/logs/training.log
```

### 🔍 **Testing Models**

```python
# Test model creation
from models.hybrid_model import create_model

for variant in ['model_e', 'model_f', 'model_g', 'model_h']:
    model = create_model(variant)
    print(f"{variant}: {model.get_model_info()}")
```

## Troubleshooting

### ❌ **Common Issues**

1. **CLAHE params not found**: Ensure `clahe_params.json` is in project root
2. **Config file missing**: Check that model config files exist in `config/`
3. **Import errors**: Verify you're running from the experiment directory

### 🔧 **Debugging**

```python
# Test CLAHE processing
from data.dataset import HybridRoadDataset

dataset = HybridRoadDataset('train', config, 'model_e')
print(f"CLAHE enabled: {dataset.use_clahe}")
print(f"CLAHE params loaded: {len(dataset.clahe_params)}")
```

## Summary

The new CLAHE-enhanced models (E, F, G, H) seamlessly integrate with your existing trainer architecture while adding powerful contrast enhancement capabilities. The configuration-driven approach ensures:

- ✅ **No trainer modifications needed**
- ✅ **Automatic CLAHE processing**
- ✅ **Optimized parameters per image**
- ✅ **Flexible model configurations**
- ✅ **Enhanced logging and monitoring**

Ready to train? Start with:
```bash
python run_model_training.py --model model_e
``` 