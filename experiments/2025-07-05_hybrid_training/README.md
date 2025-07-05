# 2025-07-05 Hybrid Training Experiment

## Overview

This experiment combines the best aspects of previous road distress classification experiments:
- **Data Splitting Strategy**: Smart road-wise splitting from 2025-06-28 experiment
- **Model Architecture**: Successful UNet + EfficientNet-B3 from 2025-05-10 experiment
- **Cross-Platform Compatibility**: Works seamlessly on both Mac and Windows

## Model Variants

| Model | Description | Input Components | Masking Strategy |
|-------|-------------|------------------|------------------|
| **Model A** | Pictures + Masks | Original images + road masks | Full masking (zero non-road) |
| **Model B** | Pictures + Augmentation | Original + augmented images | No masking |
| **Model C** | Pictures + Augmentation + Masks | Original + augmented images + masks | Full masking |
| **Model D** | Pictures + Augmentation + 50% Masks | Original + augmented images + masks | Weighted masking (50% weight to non-road) |

## Cross-Platform Compatibility

This experiment is designed to work on:
- ✅ **Windows 10/11** (CUDA GPU support)
- ✅ **macOS** (Apple Silicon MPS support)
- ✅ **Linux** (CUDA GPU support)

### Platform-Specific Features

#### Windows
- CUDA GPU acceleration
- PowerShell command support
- Windows path handling
- 8 default data loading workers

#### macOS
- Apple Silicon MPS acceleration
- Bash command support
- Unix path handling
- 6 default data loading workers (optimized for thermal throttling)

#### Linux
- CUDA GPU acceleration
- Bash command support
- Unix path handling
- 8 default data loading workers

## Prerequisites

### All Platforms
- Python 3.8 or higher
- At least 8GB RAM
- 10GB free disk space
- Internet connection for package installation

### Windows Specific
- NVIDIA GPU (optional but recommended)
- CUDA 11.8+ (if using GPU)
- Visual Studio Build Tools (for some packages)

### Mac Specific
- macOS 11+ (for MPS support)
- Apple Silicon M1/M2 (for MPS acceleration)
- Xcode Command Line Tools

## Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository and navigate to experiment directory
cd road-distress-classification/experiments/2025-07-05_hybrid_training

# Run the automated setup script
python scripts/setup.py
```

The setup script will:
- ✅ Detect your platform automatically
- ✅ Check Python version compatibility
- ✅ Install all required packages
- ✅ Create necessary directories
- ✅ Configure platform-specific settings
- ✅ Verify GPU acceleration availability
- ✅ Create sample configuration files

### Option 2: Manual Setup

#### Step 1: Install Dependencies

```bash
# Install requirements
pip install -r requirements.txt
```

#### Step 2: Create Directories

**Windows (PowerShell):**
```powershell
mkdir data/splits, data/masks, data/augmented, results/model_a, results/model_b, results/model_c, results/model_d, logs, checkpoints
```

**Mac/Linux (Bash):**
```bash
mkdir -p data/{splits,masks,augmented} results/{model_a,model_b,model_c,model_d} logs checkpoints
```

#### Step 3: Configure Platform Settings

Copy and modify the configuration:
```bash
cp config/base_config.yaml config/user_config.yaml
# Edit user_config.yaml with your specific settings
```

## Configuration

### Base Configuration
The `config/base_config.yaml` contains all default settings with cross-platform compatibility:

```yaml
# Platform auto-detection
platform:
  os: "auto"  # Automatically detects Windows/Mac/Linux
  
# Hardware auto-configuration  
hardware:
  device: "auto"  # Automatically selects best device (cuda/mps/cpu)
  
# Dataset configuration
dataset:
  num_workers: null  # Auto-detects optimal number based on platform
```

### Platform-Specific Overrides

The configuration automatically adjusts based on your platform:

| Setting | Windows | Mac | Linux |
|---------|---------|-----|-------|
| Device | CUDA → CPU | MPS → CPU | CUDA → CPU |
| Workers | 8 | 6 | 8 |
| Paths | Windows-style | Unix-style | Unix-style |
| Line Endings | CRLF | LF | LF |

### Custom Configuration

Create `config/user_config.yaml` to override any settings:

```yaml
# Override data path if needed
dataset:
  coryell_path: "/custom/path/to/coryell/data"
  
# Override hardware settings
hardware:
  device: "cpu"  # Force CPU usage
  
# Override training settings
training:
  batch_size: 32  # Reduce for lower memory systems
```

## Usage

### Step 1: Prepare Data Splits

```bash
# Run smart data splitter (works on all platforms)
python src/data/smart_splitter.py
```

This will create road-wise splits in `data/splits/`:
- `train_images.txt`
- `val_images.txt` 
- `test_images.txt`
- `split_statistics.json`
- `split_visualization.png`

### Step 2: Generate Road Masks (if needed)

```bash
# Generate road masks for all images
python src/data/mask_generator.py
```

### Step 3: Create Augmented Data (if needed)

```bash
# Generate augmented images
python src/data/augmentation.py
```

### Step 4: Train Models

Train individual model variants:

```bash
# Model A: Pictures + Masks
python src/training/train_single.py --variant model_a

# Model B: Pictures + Augmentation  
python src/training/train_single.py --variant model_b

# Model C: Pictures + Augmentation + Masks
python src/training/train_single.py --variant model_c

# Model D: Pictures + Augmentation + 50% Masks
python src/training/train_single.py --variant model_d
```

Or train all variants comparatively:

```bash
# Train all 4 variants and compare
python src/training/train_comparative.py
```

### Step 5: Evaluate Results

```bash
# Evaluate all trained models
python src/training/evaluate_models.py

# Generate comparison visualizations
python src/training/create_comparisons.py
```

## Platform-Specific Tips

### Windows Users

1. **GPU Memory Issues**: If you encounter CUDA out of memory errors:
   ```yaml
   # In user_config.yaml
   dataset:
     batch_size: 32  # Reduce from 64
   ```

2. **Path Issues**: Use forward slashes in config files:
   ```yaml
   dataset:
     coryell_path: "../../data/coryell"  # Not "..\\..\\data\\coryell"
   ```

3. **PowerShell Execution Policy**:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

### Mac Users

1. **Apple Silicon MPS**: Ensure you have PyTorch with MPS support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Memory Management**: Mac systems may need reduced batch sizes:
   ```yaml
   # In user_config.yaml  
   dataset:
     batch_size: 32
     num_workers: 4  # Reduce for thermal management
   ```

3. **Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

### Linux Users

1. **CUDA Setup**: Ensure CUDA is properly installed:
   ```bash
   nvidia-smi  # Check GPU status
   ```

2. **Permissions**: You may need to set file permissions:
   ```bash
   chmod +x scripts/*.py
   ```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Check if packages are installed
pip list | grep torch
pip list | grep opencv

# Reinstall if needed
pip install --upgrade torch torchvision opencv-python
```

#### GPU Not Detected
```python
# Test GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
```

#### Path Issues
```bash
# Check current directory
pwd  # Unix
cd   # Windows

# Verify data path exists
ls ../../data/coryell  # Unix
dir ..\..\data\coryell  # Windows
```

#### Memory Issues
- Reduce batch size in configuration
- Reduce number of workers
- Close other applications
- Use CPU instead of GPU if needed

### Platform-Specific Issues

#### Windows
- **DLL Load Errors**: Install Visual Studio Build Tools
- **Permission Denied**: Run as Administrator or use `--user` flag with pip
- **Path Length Limit**: Enable long path support in Windows

#### Mac
- **MPS Errors**: Add `PYTORCH_ENABLE_MPS_FALLBACK=1` to environment
- **Thermal Throttling**: Reduce batch size and workers
- **Permission Issues**: Use `sudo` for system-wide installations

#### Linux
- **CUDA Version Mismatch**: Ensure PyTorch CUDA version matches system CUDA
- **Permission Issues**: Use `sudo` for system directories
- **Display Issues**: Set `DISPLAY` environment variable for GUI

## Monitoring and Logging

### TensorBoard (All Platforms)
```bash
# Start TensorBoard
tensorboard --logdir logs

# Open browser to http://localhost:6006
```

### System Monitoring

**Windows:**
```powershell
# Monitor GPU usage
nvidia-smi -l 1

# Monitor CPU/Memory
Get-Process python
```

**Mac:**
```bash
# Monitor system resources
top -pid $(pgrep python)

# Monitor GPU (if available)
sudo powermetrics --samplers gpu_power -n 1
```

**Linux:**
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor CPU/Memory
htop
```

## Expected Results

### Performance Targets
- **Training Time**: 2-4 hours per model on GPU
- **Memory Usage**: 4-8GB GPU memory, 8-16GB RAM
- **Accuracy**: >85% on test set
- **F1-Score**: >0.80 weighted average

### Output Files
```
results/
├── model_a/
│   ├── checkpoints/
│   ├── logs/
│   ├── predictions/
│   └── visualizations/
├── model_b/
├── model_c/
├── model_d/
└── comparison/
    ├── performance_comparison.png
    ├── confusion_matrices.png
    └── model_analysis.json
```

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black src/
isort src/
```

### Adding New Platforms
1. Update `platform_utils.py` with new platform detection
2. Add platform-specific configuration in `base_config.yaml`
3. Test thoroughly on the new platform
4. Update documentation

## Support

### Getting Help
1. Check this README for common issues
2. Review the logs in `logs/` directory
3. Check GPU/CPU availability and memory usage
4. Verify data paths and permissions

### Reporting Issues
When reporting issues, please include:
- Platform (Windows/Mac/Linux)
- Python version
- GPU information (if applicable)
- Error messages and logs
- Configuration files

## License

This experiment is part of the road distress classification research project. Please refer to the main project license for usage terms. 