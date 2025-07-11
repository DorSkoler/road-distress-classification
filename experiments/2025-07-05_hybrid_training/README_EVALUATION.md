# Model Evaluation and Comparison System

This directory contains a comprehensive evaluation and comparison system for the trained road distress classification models.

## 🎯 Overview

The evaluation system provides:
- **Comprehensive metrics**: Accuracy, F1, Precision, Recall, AUC, and more
- **Model comparison**: Side-by-side analysis of all model variants
- **Visualizations**: Charts, plots, and confusion matrices
- **Recommendations**: Best model selection for different use cases
- **Detailed reports**: Text summaries and CSV rankings

## 📁 System Architecture

```
src/evaluation/
├── __init__.py                 # Module initialization
├── metrics_calculator.py       # Comprehensive metrics calculation
├── model_evaluator.py         # Individual model evaluation
├── comparison_runner.py        # Full comparison orchestration
└── visualization.py            # Plotting and charts

run_comparison.py              # Main execution script
```

## 🚀 Quick Start

### 1. Run Complete Comparison

```bash
# Compare all models
python run_comparison.py

# Compare specific models
python run_comparison.py --variants model_a model_b

# Verbose logging
python run_comparison.py --verbose
```

### 2. Check Results

Results are saved to `results/comparison/`:
- `summary_report.txt` - Comprehensive text report
- `model_rankings.csv` - Sortable ranking table
- `plots/` - All visualizations
- `evaluation_results.json` - Complete detailed data

## 📊 Generated Outputs

### Visualizations
1. **Metric Comparison**: Bar charts of key performance metrics
2. **Confusion Matrices**: Per-class confusion matrices for each model
3. **Performance Radar**: Multi-metric radar chart comparison
4. **Training Efficiency**: Performance vs training time scatter plot
5. **Class Performance**: Per-class F1, precision, recall comparison
6. **Summary Table**: Visual ranking table

### Reports
1. **Summary Report**: Human-readable analysis with recommendations
2. **Rankings CSV**: Sortable table for further analysis
3. **Detailed JSON**: Complete metrics and metadata

## 🔍 Available Metrics

### Overall Metrics
- **Macro F1**: Unweighted average F1 across classes
- **Macro Precision/Recall**: Unweighted averages
- **Hamming Accuracy**: Per-label accuracy (most relevant for multi-label)
- **Exact Match Accuracy**: All labels must be correct
- **Micro F1/Precision/Recall**: Weighted by support

### Per-Class Metrics
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Specificity**: True negatives / (true negatives + false positives)
- **AUC**: Area under ROC curve (if probabilities available)

### Training Efficiency
- **Training Time**: Hours spent training
- **Efficiency Score**: F1 score per training hour
- **Convergence**: Best epoch and early stopping behavior
- **Overfitting**: Validation/training loss ratio

## 🎯 Understanding Results

### Model Rankings
Models are ranked by **Macro F1** score, which gives equal weight to all classes:
- **High Macro F1** (>0.80): Excellent performance
- **Medium Macro F1** (0.60-0.80): Good performance
- **Low Macro F1** (<0.60): Needs improvement

### Class-Specific Analysis
Each model's performance varies by class:
- **Damage Detection**: Often most challenging due to subtle patterns
- **Occlusion Detection**: Usually easier with clear visual patterns
- **Crop Detection**: Typically highest performance (binary decision)

### Training Efficiency
Consider both performance and training time:
- **High Efficiency**: Good F1 score with short training time
- **Diminishing Returns**: Very long training with minimal gain
- **Overfitting**: High train performance, poor validation

## 🛠 Advanced Usage

### Evaluate Single Model

```python
from evaluation.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator('config/base_config.yaml')
results = evaluator.evaluate_model('model_a')
print(results['summary_table'])
```

### Custom Comparison

```python
from evaluation.comparison_runner import ComparisonRunner

runner = ComparisonRunner('config/base_config.yaml')
results = runner.run_full_comparison(['model_a', 'model_c'])
```

### Generate Only Visualizations

```python
from evaluation.visualization import create_comparison_plots

# Assuming you have results dictionary
create_comparison_plots(results, 'output/plots/')
```

## 📋 Troubleshooting

### Common Issues

1. **Missing Checkpoints**
   ```
   Error: Checkpoint not found: results/model_a/checkpoints/best_checkpoint.pth
   ```
   - Ensure models are fully trained
   - Check checkpoint file names match expectations

2. **Configuration Errors**
   ```
   Error: Configuration file not found: config/base_config.yaml
   ```
   - Verify config file path
   - Ensure config file is valid YAML

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'evaluation'
   ```
   - Run from the experiment directory
   - Ensure `src/` is in Python path

4. **Memory Issues**
   ```
   CUDA out of memory
   ```
   - Reduce batch size in config
   - Evaluate fewer models at once

### Performance Tips

1. **Speed up evaluation**: Reduce test batch size if memory constrained
2. **Parallel evaluation**: Models are evaluated sequentially (safe for GPU memory)
3. **Selective comparison**: Use `--variants` to compare only specific models

## 📈 Interpreting Recommendations

### Best Overall Model
- Highest Macro F1 score
- Balanced performance across all classes
- Good for general-purpose deployment

### Most Efficient Model  
- Best F1 score per training hour
- Good for resource-constrained scenarios
- Consider when training time matters

### Class-Specific Recommendations
- **Best for Damage**: Highest damage detection F1
- **Best for Occlusion**: Highest occlusion detection F1  
- **Best for Crop**: Highest crop classification F1

Use class-specific recommendations when you have a primary use case focus.

## 🔧 Customization

### Adding New Metrics

Edit `metrics_calculator.py` to add custom metrics:

```python
def _calculate_custom_metrics(self, predictions, labels):
    # Add your custom metric calculation
    return {'custom_metric': value}
```

### Custom Visualizations

Edit `visualization.py` to add new plots:

```python
def _plot_custom_analysis(results, output_dir):
    # Add your custom visualization
    pass
```

### Modify Reports

Edit `comparison_runner.py` to customize report format:

```python
def _create_summary_report(self, evaluation_results, comparison_analysis):
    # Customize report content and format
    pass
```

This evaluation system provides comprehensive analysis to help you select the best model for your specific road distress classification needs! 