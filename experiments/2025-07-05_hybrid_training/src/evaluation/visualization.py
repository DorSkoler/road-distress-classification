#!/usr/bin/env python3
"""
Visualization utilities for model comparison.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_comparison_plots(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Create comprehensive comparison plots for all models.
    
    Args:
        results: Dictionary with results for all models
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract valid results (excluding failed models)
    valid_results = {k: v for k, v in results.items() 
                    if k != 'evaluation_summary' and 'error' not in v}
    
    if not valid_results:
        logger.warning("No valid results to plot")
        return
    
    logger.info(f"Creating comparison plots for {len(valid_results)} models")
    
    # Create individual plots
    _plot_metric_comparison(valid_results, output_dir)
    _plot_confusion_matrices(valid_results, output_dir)
    _plot_roc_curves(valid_results, output_dir)
    _plot_performance_radar(valid_results, output_dir)
    _plot_training_efficiency(valid_results, output_dir)
    _plot_class_performance(valid_results, output_dir)
    
    logger.info(f"All comparison plots saved to {output_dir}")

def _plot_metric_comparison(results: Dict[str, Any], output_dir: Path) -> None:
    """Plot comparison of key metrics across models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    models = list(results.keys())
    metrics_to_plot = ['macro_f1', 'macro_precision', 'macro_recall', 'hamming_accuracy']
    metric_labels = ['Macro F1', 'Macro Precision', 'Macro Recall', 'Hamming Accuracy']
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        values = [results[model]['test_metrics'][metric] for model in models]
        colors = sns.color_palette("husl", len(models))
        
        bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(label, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='x', rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def _plot_confusion_matrices(results: Dict[str, Any], output_dir: Path) -> None:
    """Plot confusion matrices for each model and class."""
    class_names = ['damage', 'occlusion', 'crop']
    n_models = len(results)
    
    fig, axes = plt.subplots(n_models, 3, figsize=(12, 4 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Confusion Matrices by Model and Class', fontsize=16, fontweight='bold')
    
    for model_idx, (model_name, model_results) in enumerate(results.items()):
        confusion_matrices = model_results['test_metrics']['confusion_matrices']
        
        for class_idx, class_name in enumerate(class_names):
            ax = axes[model_idx, class_idx]
            cm = confusion_matrices[class_name]
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=[f'Not {class_name}', class_name],
                       yticklabels=[f'Not {class_name}', class_name])
            
            ax.set_title(f'{model_name}: {class_name.capitalize()}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def _plot_roc_curves(results: Dict[str, Any], output_dir: Path) -> None:
    """Plot ROC curves for each class."""
    class_names = ['damage', 'occlusion', 'crop']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('ROC Curves by Class', fontsize=16, fontweight='bold')
    
    colors = sns.color_palette("husl", len(results))
    
    for class_idx, class_name in enumerate(class_names):
        ax = axes[class_idx]
        
        for model_idx, (model_name, model_results) in enumerate(results.items()):
            # Check if AUC is available
            auc_key = f'{class_name}_auc'
            if auc_key in model_results['test_metrics']:
                auc = model_results['test_metrics'][auc_key]
                
                # Plot a placeholder curve (would need actual ROC data for real curve)
                # For now, just show AUC values
                ax.text(0.6, 0.4 - model_idx * 0.1, 
                       f'{model_name}: AUC = {auc:.3f}',
                       color=colors[model_idx], fontweight='bold')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{class_name.capitalize()} ROC Curves')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def _plot_performance_radar(results: Dict[str, Any], output_dir: Path) -> None:
    """Create radar chart comparing models across multiple metrics."""
    metrics = ['macro_precision', 'macro_recall', 'macro_f1', 'hamming_accuracy']
    metric_labels = ['Precision', 'Recall', 'F1', 'Accuracy']
    
    # Set up radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = sns.color_palette("husl", len(results))
    
    for model_idx, (model_name, model_results) in enumerate(results.items()):
        values = [model_results['test_metrics'][metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, 
               color=colors[model_idx], markersize=8)
        ax.fill(angles, values, alpha=0.25, color=colors[model_idx])
    
    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    plt.title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(output_dir / 'performance_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

def _plot_training_efficiency(results: Dict[str, Any], output_dir: Path) -> None:
    """Plot training time vs performance efficiency."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    models = []
    training_times = []
    f1_scores = []
    colors = sns.color_palette("husl", len(results))
    
    for model_name, model_results in results.items():
        model_info = model_results.get('model_info', {})
        training_time = model_info.get('training_time_hours', 0)
        f1_score = model_results['test_metrics']['macro_f1']
        
        models.append(model_name)
        training_times.append(training_time)
        f1_scores.append(f1_score)
    
    # Create scatter plot
    scatter = ax.scatter(training_times, f1_scores, s=200, c=colors, alpha=0.7, edgecolors='black')
    
    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(model, (training_times[i], f1_scores[i]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontweight='bold', fontsize=12)
    
    ax.set_xlabel('Training Time (hours)', fontweight='bold')
    ax.set_ylabel('Macro F1 Score', fontweight='bold')
    ax.set_title('Training Efficiency: Performance vs Time', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()

def _plot_class_performance(results: Dict[str, Any], output_dir: Path) -> None:
    """Plot per-class performance comparison."""
    class_names = ['damage', 'occlusion', 'crop']
    metrics = ['precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Per-Class Performance Comparison', fontsize=16, fontweight='bold')
    
    models = list(results.keys())
    x_pos = np.arange(len(models))
    bar_width = 0.25
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        for class_idx, class_name in enumerate(class_names):
            values = [results[model]['test_metrics'][f'{class_name}_{metric}'] 
                     for model in models]
            
            offset = (class_idx - 1) * bar_width
            bars = ax.bar(x_pos + offset, values, bar_width, 
                         label=class_name.capitalize(), 
                         color=colors[class_idx], alpha=0.8,
                         edgecolor='black', linewidth=1)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} by Class')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table_plot(results: Dict[str, Any], output_dir: Path) -> None:
    """Create a summary table as an image."""
    # Prepare data for table
    models = [k for k in results.keys() if k != 'evaluation_summary' and 'error' not in results[k]]
    
    if not models:
        return
    
    # Extract key metrics
    table_data = []
    for model in models:
        metrics = results[model]['test_metrics']
        model_info = results[model].get('model_info', {})
        
        row = [
            model,
            f"{metrics.get('macro_f1', 0):.3f}",
            f"{metrics.get('macro_precision', 0):.3f}",
            f"{metrics.get('macro_recall', 0):.3f}",
            f"{metrics.get('hamming_accuracy', 0):.3f}",
            f"{metrics.get('exact_match_accuracy', 0):.3f}",
            f"{model_info.get('training_time_hours', 0):.2f}h"
        ]
        table_data.append(row)
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    headers = ['Model', 'Macro F1', 'Precision', 'Recall', 'Hamming Acc', 'Exact Match', 'Train Time']
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colColours=['lightblue'] * len(headers))
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_text_props(weight='bold')
    
    plt.title('Model Comparison Summary Table', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close() 