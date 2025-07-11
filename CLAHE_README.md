# CLAHE Parameter Optimization for Road Distress Analysis

This system automatically finds the optimal CLAHE (Contrast Limited Adaptive Histogram Equalization) parameters for road distress detection in drone images. It's specifically designed to enhance visibility of road damage, occlusions, and improve overall image quality for analysis.

## 🎯 Purpose

For road distress analysis (detecting damage, occlusions, crops), optimal CLAHE parameters are crucial:
- **Damage detection**: Enhanced edge visibility for cracks, potholes
- **Occlusion handling**: Better contrast in shadowed areas  
- **Quality preservation**: Avoid over-enhancement artifacts

## 📋 Requirements

Install dependencies:
```bash
pip install -r clahe_requirements.txt
```

Or manually:
```bash
pip install opencv-python numpy matplotlib scikit-learn
```

## 🚀 Quick Start

### Command Line Usage

Basic optimization:
```bash
python optimize_clahe_for_road_distress.py --image your_road_image.jpg
```

With detailed results and visualizations:
```bash
python optimize_clahe_for_road_distress.py --image your_road_image.jpg --save-results
```

Custom output directory:
```bash
python optimize_clahe_for_road_distress.py --image your_road_image.jpg --save-results --output-dir my_results
```

### Programmatic Usage

```python
from optimize_clahe_for_road_distress import CLAHEOptimizer

# Initialize and optimize
optimizer = CLAHEOptimizer("road_image.jpg")
best_config = optimizer.optimize()

# Get optimal parameters
clip_limit = best_config['clip_limit']
tile_grid_size = best_config['tile_grid_size']

# Apply to new images
enhanced_image = optimizer.apply_best_clahe("another_road_image.jpg")

# Save detailed results
optimizer.save_results("results_folder")
```

## 📊 What Gets Evaluated

The optimizer tests **30 parameter combinations** (6 clip limits × 5 tile grid sizes) and evaluates each using metrics specifically relevant to road distress analysis:

### Core Metrics

1. **Edge Quality (35% weight)** - Crucial for crack detection
   - Canny edge density and coherence
   - Higher weight for road damage visibility

2. **Contrast Enhancement (25% weight)** - Important for shadows/occlusions  
   - Local standard deviation improvement
   - Helps with uneven lighting conditions

3. **Texture Preservation (20% weight)** - Road surface analysis
   - Laplacian variance for texture quality
   - Maintains road surface details

4. **Overall Quality (30% weight)** - General image quality
   - BRISQUE-like quality assessment
   - Prevents over-enhancement

5. **Noise Artifacts (-10% weight)** - Penalty for artifacts
   - Bilateral filtering to detect noise
   - Negative weight to penalize over-enhancement

### Parameter Ranges Tested

- **Clip Limits**: [1.0, 2.0, 3.0, 4.0, 5.0, 8.0]
- **Tile Grid Sizes**: [(4,4), (6,6), (8,8), (12,12), (16,16)]

## 📁 Output Files

When using `--save-results`, you get:

```
clahe_optimization_results/
├── clahe_optimization_results.json    # All configurations and scores
├── clahe_comparison.png               # Top 6 configurations visual
├── metrics_comparison.png             # Detailed metrics analysis  
├── before_after_comparison.png        # Side-by-side comparison
├── rank_1_clip_X.X_grid_YxY.png      # Top 5 enhanced images
├── rank_2_clip_X.X_grid_YxY.png
└── ...
```

## 🎛️ Understanding Results

### Example Output
```
Best configuration:
  Clip Limit: 3.0
  Tile Grid Size: (8, 8)
  Composite Score: 0.7245

Detailed metrics:
  Edge Quality: 0.0851
  Contrast Enhancement: 1.2341  
  Texture Preservation: 45.6789
  Noise Artifacts: 12.3456
  Overall Quality: -0.0234

Recommended CLAHE parameters:
  cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
```

### Using Results in Your Code

```python
import cv2

# Apply optimized CLAHE to any image
def enhance_road_image(image_path, clip_limit=3.0, tile_grid_size=(8, 8)):
    img = cv2.imread(image_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)
    
    lab[:, :, 0] = enhanced_l
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

## 💡 Best Practices

### For Road Distress Analysis:

1. **Use representative images**: Test on images with typical lighting/damage conditions
2. **Consider image size**: Larger images may benefit from larger tile grid sizes
3. **Validate results**: Check enhanced images visually for your specific use case
4. **Batch processing**: Use optimized parameters across similar road images

### Parameter Guidelines:

- **Low clip limit (1.0-3.0)**: Conservative enhancement, good for high-quality images
- **Medium clip limit (3.0-5.0)**: Balanced enhancement, good for most cases  
- **High clip limit (5.0-8.0)**: Aggressive enhancement, for very dark/low-contrast images

- **Small grids (4×4-6×6)**: Fine-grained local adaptation
- **Large grids (12×12-16×16)**: Broader region adaptation

## 🔧 Customization

### Modify Evaluation Weights

Edit the `calculate_composite_score` method in `CLAHEOptimizer`:

```python
weights = {
    'edge_quality': 0.40,          # Increase for more edge focus
    'contrast_enhancement': 0.30,   # Increase for shadow handling
    'texture_preservation': 0.15,   # Adjust for surface detail importance
    'noise_artifacts': -0.15,       # More negative = stricter noise penalty
    'overall_quality': 0.30
}
```

### Add Custom Metrics

Add your own evaluation function to the `evaluate_clahe_config` method:

```python
def your_custom_metric(self, image: np.ndarray) -> float:
    # Your custom evaluation logic
    return score
```

## 📖 Examples

See `example_clahe_usage.py` for detailed usage examples including:
- Programmatic optimization
- Batch processing multiple images
- Integration with existing pipelines

## ⚡ Performance

- **Runtime**: ~5-15 seconds for 30 configurations on typical images
- **Memory**: Moderate (keeps enhanced images in memory during optimization)
- **Scalability**: Linear with number of parameter combinations

## 🔍 Technical Details

- **Color space**: LAB (better than RGB for luminance enhancement)
- **Edge detection**: Canny with adaptive thresholds
- **Quality assessment**: BRISQUE-inspired metrics
- **Optimization**: Exhaustive search over parameter grid

Perfect for drone road imagery analysis where optimal contrast enhancement is critical for damage detection! 