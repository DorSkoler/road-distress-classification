# Road Distress Classification Multi-Model Ensemble Pipeline

A comprehensive multi-model inference pipeline for road distress classification featuring an ensemble of best-performing models with advanced Grad-CAM visualization and an intuitive web interface. This pipeline processes images of any size and provides detailed multi-model analysis with real attention mapping.

## 🌟 Key Features

### **🤖 Multi-Model Ensemble**
- **Model B**: Best performing model (Macro F1: 0.806) - No masks, no CLAHE
- **Model H**: Enhanced model (Macro F1: 0.781) - CLAHE + partial masks + augmentation
- **Ensemble Decision**: Weighted combination of both models for superior accuracy
- **Adjustable Thresholds**: Per-class confidence thresholds (damage, occlusion, crop)

### **🎯 Advanced Grad-CAM Visualization**
- **Individual Model Attention**: View Model B or Model H attention separately
- **Combined Attention**: Weighted average of both models' attention maps
- **Class-Specific Grad-CAM**: Focus on damage, occlusion, or crop attention
- **Real Model Insights**: Shows actual neural network attention, not artificial patterns

### **🎨 Modern Web Interface**
- **Dark Theme**: Professional UI with high contrast and modern gradients
- **Real-Time Processing**: Live inference with progress indicators
- **Interactive Visualizations**: Tabbed views, scalable images, and detailed charts
- **Batch Processing**: Handle multiple images simultaneously
- **Download Results**: JSON exports and visualization downloads

### **📊 Comprehensive Analysis**
- **3-Step Pipeline**: Model B → Model H → Ensemble Decision
- **Detailed Metrics**: Probabilities, predictions, and confidence scores
- **Model Comparison**: Side-by-side performance analysis
- **Flexible Input**: Any image size and format (JPG, PNG, BMP, TIFF)

## 🚀 Quick Start

### 1. Installation

```bash
cd road-distress-classification/inference_pipeline
pip install -r requirements.txt
```

### 2. Launch Web Interface

```bash
# Start the interactive web UI
streamlit run app.py

# Or use the launcher script
python launch_ui.py
```

Navigate to `http://localhost:8501` to access the full-featured web interface.

### 3. Command Line Usage

```bash
# Single image inference
python run_inference.py --image path/to/your/image.jpg --output results/

# Batch processing
python run_inference.py --batch path/to/images/ --output results/
```

## 🏗️ Architecture Overview

### Multi-Model Ensemble System

```
Input Image
    ↓
┌─────────────────────────────────────┐
│         Image Preprocessing         │
│    (Resize, Normalize, Augment)     │
└─────────────────────────────────────┘
    ↓                    ↓
┌─────────────┐    ┌─────────────┐
│   Model B   │    │   Model H   │
│  (No masks  │    │ (CLAHE +    │
│  No CLAHE)  │    │  Masks)     │
└─────────────┘    └─────────────┘
    ↓                    ↓
┌─────────────────────────────────────┐
│        Ensemble Combination         │
│     (Weighted Average: 50/50)       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│       Final Predictions             │
│   + Grad-CAM Visualizations        │
└─────────────────────────────────────┘
```

## 📱 Web Interface Features

### **🎯 Multi-Model Grad-CAM Options**
- **🅱️ Model B Only**: View attention from the best-performing model
- **🅷 Model H Only**: View attention from the enhanced model
- **🎯 Combined**: Weighted average of both models' attention maps

### **⚖️ Adjustable Thresholds**
- **Damage Threshold**: Customize sensitivity for damage detection
- **Occlusion Threshold**: Adjust for vegetation/obstruction detection
- **Crop Threshold**: Fine-tune edge/boundary detection

### **🎨 Visualization Options**
- **Clean Heatmap Overlay**: Attention map overlaid on original image
- **Pure Confidence Map**: Raw attention visualization
- **Side by Side**: Original vs. processed comparison

### **📊 Analysis Pipeline**
1. **Step 1**: Model B Analysis with individual predictions
2. **Step 2**: Model H Analysis with individual predictions  
3. **Step 3**: Ensemble Decision with combined results
4. **Comparison Table**: Performance metrics across all models

## 🔧 Python API Usage

### Basic Multi-Model Inference

```python
from src import EnsembleInferenceEngine, HeatmapGenerator

# Initialize multi-model engine
engine = EnsembleInferenceEngine()
heatmap_gen = HeatmapGenerator()

# Run individual model predictions
model_b_results = engine.predict_single_model("image.jpg", "model_b")
model_h_results = engine.predict_single_model("image.jpg", "model_h")

# Get ensemble prediction
ensemble_results = engine.predict_ensemble("image.jpg")

print(f"Model B Damage: {model_b_results['class_results']['damage']['probability']:.1%}")
print(f"Model H Damage: {model_h_results['class_results']['damage']['probability']:.1%}")
print(f"Ensemble Damage: {ensemble_results['ensemble_results']['class_results']['damage']['probability']:.1%}")
```

### Advanced Grad-CAM Visualization

```python
# Generate Grad-CAM for specific models
gradcam_b, _ = engine.get_damage_confidence_map(
    "image.jpg", 
    method="gradcam", 
    target_class="damage",
    model_name="model_b"
)

gradcam_h, _ = engine.get_damage_confidence_map(
    "image.jpg", 
    method="gradcam", 
    target_class="damage", 
    model_name="model_h"
)

# Combined Grad-CAM from both models
combined_gradcam, _ = engine.get_damage_confidence_map(
    "image.jpg", 
    method="gradcam", 
    target_class="damage",
    model_name="combined"
)

# Create visualizations
clean_heatmap = heatmap_gen.create_clean_heatmap(image, combined_gradcam)
```

### Threshold Customization

```python
# Set custom thresholds
custom_thresholds = {
    'damage': 0.7,      # Higher sensitivity for damage
    'occlusion': 0.3,   # Lower sensitivity for vegetation
    'crop': 0.4         # Medium sensitivity for boundaries
}

engine.update_thresholds(custom_thresholds)
results = engine.predict_ensemble("image.jpg")
```

## 📁 Output Structure

### Multi-Model Results Format

```json
{
  "filename": "road_image.jpg",
  "model_b_results": {
    "probabilities": [0.856, 0.124, 0.089],
    "predictions": [true, false, false],
    "class_results": {
      "damage": {
        "probability": 0.856,
        "prediction": true,
        "confidence": 0.856,
        "threshold": 0.5
      }
    },
    "overall_confidence": 0.847,
    "inference_time": 0.034
  },
  "model_h_results": {
    "probabilities": [0.723, 0.156, 0.098],
    "predictions": [true, false, false],
    "class_results": {
      "damage": {
        "probability": 0.723,
        "prediction": true,
        "confidence": 0.723,
        "threshold": 0.5
      }
    },
    "overall_confidence": 0.789,
    "inference_time": 0.041
  },
  "ensemble_results": {
    "ensemble_results": {
      "probabilities": [0.790, 0.140, 0.094],
      "predictions": [true, false, false],
      "class_results": {
        "damage": {
          "probability": 0.790,
          "prediction": true,
          "confidence": 0.790,
          "threshold": 0.5
        }
      },
      "overall_confidence": 0.818,
      "model_weights": [0.5, 0.5]
    }
  }
}
```

## 🎯 Model Performance

### Model B (Primary)
- **Macro F1**: 0.806
- **Architecture**: EfficientNet-B3
- **Training**: No masks, no CLAHE, augmentation only
- **Strengths**: Highest overall performance, robust feature extraction

### Model H (Enhanced)  
- **Macro F1**: 0.781
- **Architecture**: EfficientNet-B3
- **Training**: CLAHE + partial masks + augmentation
- **Strengths**: Enhanced contrast sensitivity, edge detection

### Ensemble Performance
- **Combined Accuracy**: Superior to individual models
- **Robust Predictions**: Weighted decision making
- **Confidence Calibration**: Improved reliability through averaging

## 🛠️ Configuration

### Model Configuration (`config.yaml`)

```yaml
models:
  model_b:
    checkpoint_path: "../experiments/2025-07-05_hybrid_training/results/model_b/checkpoints/best_model.pth"
    architecture: "efficientnet_b3"
    macro_f1: 0.806
    description: "No masks, no CLAHE, augmentation only"
  
  model_h:
    checkpoint_path: "../experiments/2025-07-05_hybrid_training/results/model_h/checkpoints/best_model.pth"
    architecture: "efficientnet_b3"
    macro_f1: 0.781
    description: "CLAHE + partial masks + augmentation"

ensemble:
  enabled: true
  default_weights: [0.5, 0.5]

inference:
  thresholds:
    damage: 0.5
    occlusion: 0.5
    crop: 0.5
```

## 🔬 Advanced Features

### Grad-CAM Implementation
- **Layer Targeting**: Uses final convolutional layer for attention
- **Gradient Computation**: Backpropagation through class-specific outputs
- **Attention Fusion**: Combines gradients with feature maps
- **Multi-Class Support**: Individual attention maps per class

### Ensemble Strategy
- **Weighted Averaging**: Configurable model weights
- **Threshold Application**: Per-class decision boundaries
- **Confidence Aggregation**: Combined uncertainty estimation

### Performance Optimization
- **Model Caching**: Streamlit resource caching for fast reloads
- **Batch Processing**: Efficient multi-image handling
- **Memory Management**: Optimized tensor operations
- **GPU Acceleration**: Automatic CUDA detection and usage

## 🚀 Deployment Options

### Local Development
```bash
# Development server
streamlit run app.py --server.port 8501

# Production server
streamlit run app.py --server.port 80 --server.address 0.0.0.0
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

### Cloud Platforms
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Container deployment
- **AWS/GCP**: Scalable cloud hosting
- **Azure**: Enterprise integration

## 🧪 Testing

### Unit Tests
```bash
# Run test suite
python -m pytest tests/

# Test specific components
python -m pytest tests/test_ensemble_engine.py
python -m pytest tests/test_gradcam.py
```

### UI Testing
```bash
# Test UI components
python test_modern_ui.py

# Restart UI with fresh cache
python restart_ui.py
```

## 📈 Future Enhancements

### Planned Features
- [ ] **Video Processing**: Real-time road distress detection in video streams
- [ ] **Mobile App**: React Native mobile application
- [ ] **API Server**: RESTful API for integration
- [ ] **Model Versioning**: A/B testing and model comparison
- [ ] **Active Learning**: Continuous model improvement
- [ ] **Edge Deployment**: Optimized models for edge devices

### Research Directions
- [ ] **Transformer Models**: Vision Transformer integration
- [ ] **Self-Supervised Learning**: Unlabeled data utilization
- [ ] **Federated Learning**: Distributed model training
- [ ] **Explainable AI**: Enhanced interpretability methods

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd road-distress-classification/inference_pipeline

# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **EfficientNet**: Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks.
- **Grad-CAM**: Selvaraju, R. R., et al. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization.
- **Streamlit**: For providing an excellent framework for ML web applications.

## 📞 Support

For questions, issues, or contributions:
- 📧 Email: [your-email@domain.com]
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/discussions)

---

🛣️ **Built with ❤️ for safer roads through AI-powered infrastructure monitoring** 🛣️