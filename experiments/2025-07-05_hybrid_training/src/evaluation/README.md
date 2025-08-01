# Evaluation Module

This module contains evaluation utilities for the hybrid road distress classification models, with **unified support** for both standard and CLAHE-enhanced models.

## Files Overview

### Core Evaluation
- **`model_evaluator.py`** - **Unified evaluator** with automatic CLAHE detection and dynamic optimization
- **`metrics_calculator.py`** - Metrics computation utilities

### Additional Files
- **`comparison_runner.py`** - Existing comparison utilities
- **`visualization.py`** - Existing visualization utilities

## Key Features

### 🎯 **Dynamic CLAHE Optimization**
CLAHE-trained models (E, F, G, H) are evaluated using the same dynamic optimization pipeline they were trained with:

```python
# For each test image:
1. Run batch_clahe_optimization.py to find optimal parameters
2. Apply optimal CLAHE using found parameters  
3. Feed enhanced image to model for evaluation
4. Ensure train/test preprocessing consistency
```

### 🔍 **Automatic CLAHE Detection**
```python
clahe_variants = ['model_e', 'model_f', 'model_g', 'model_h']
uses_clahe = variant in clahe_variants

if uses_clahe:
    test_dataset.use_dynamic_clahe_optimization = True
```

### 🛡️ **Robust Fallback System**
- **Primary**: Dynamic optimization using `batch_clahe_optimization.py`
- **Fallback**: Pre-computed parameters from `clahe_params.json`
- **Emergency**: Default CLAHE parameters (clip_limit=3.0, tile_grid=(8,8))

## Usage

All evaluation functionality is now available through the unified `ModelEvaluator` class.

### Examples

```python
from evaluation.model_evaluator import ModelEvaluator

# Initialize unified evaluator
evaluator = ModelEvaluator('config/base_config.yaml', 'results')

# Evaluate single model (automatic CLAHE detection)
results = evaluator.evaluate_model('model_h')  # CLAHE model
results = evaluator.evaluate_model('model_b')  # Standard model

# Evaluate all available models
all_results = evaluator.evaluate_all_models()

# Evaluate only CLAHE models  
clahe_results = evaluator.evaluate_clahe_models()

# Check evaluation details
dynamic_clahe = results['evaluation_config']['dynamic_clahe_optimization']
print(f"Dynamic CLAHE used: {dynamic_clahe}")
```

## Model Variants

| Model | CLAHE | Dynamic Optimization | Mask Weight | Augmentation |
|-------|-------|---------------------|-------------|--------------|
| Model A | ❌ | ❌ | Various | ❌ |
| Model B | ❌ | ❌ | Various | ✅ |
| Model C | ❌ | ❌ | Various | ✅ |
| Model D | ❌ | ❌ | Various | ✅ |
| **Model E** | ✅ | **✅** | 1.0 | ❌ |
| **Model F** | ✅ | **✅** | 0.5 | ❌ |
| **Model G** | ✅ | **✅** | 1.0 | ✅ |
| **Model H** | ✅ | **✅** | 0.5 | ✅ |

## Expected Benefits

### 🎯 **Improved Accuracy**
- **Train/test consistency** achieved for CLAHE models
- **Optimal CLAHE parameters** per test image
- **True model performance** without preprocessing mismatch

### 🔬 **Fair Evaluation**
- **Proper comparison** between CLAHE and non-CLAHE models
- **Consistent preprocessing** matching training pipeline
- **Detailed evaluation reports** with preprocessing info

### 🚀 **Production Readiness**
- **Same optimization** as training time
- **Robust fallback** system for reliability
- **Enhanced monitoring** and logging

## Dependencies

For full functionality, install:
```bash
pip3 install --break-system-packages scikit-learn matplotlib seaborn
```

## Integration Points

### Dataset Enhancement
The `HybridRoadDataset` in `../data/dataset.py` has been enhanced with:
- `apply_clahe()` method with dual modes
- `apply_dynamic_clahe_optimization()` for real-time optimization
- `apply_precomputed_clahe()` for static parameters
- `use_dynamic_clahe_optimization` configuration flag

### Batch Optimization Integration
The `batch_clahe_optimization.py` in the project root has been enhanced to:
- Accept numpy arrays directly (not just file paths)
- Support memory-based optimization during evaluation
- Provide optimal parameter extraction and application

## Testing

Run the test suite to verify integration:
```bash
python3 test_dynamic_clahe.py
```

Expected output:
```
🚀 Testing Dynamic CLAHE Integration
============================================================
   CLAHE Detection: ✅ PASSED
   Dynamic CLAHE Integration: ✅ PASSED  
   Batch CLAHE Optimizer: ✅ PASSED
   CLAHE-Aware Evaluator: ✅ PASSED

🎯 Summary: 4/4 tests passed
🎉 All tests passed! Dynamic CLAHE integration is working correctly.
```

---

**This evaluation module ensures that CLAHE-trained models are evaluated using the same dynamic CLAHE optimization pipeline they were trained with, providing fair and accurate performance assessment.**