#!/usr/bin/env python3
"""
Create all visualizations for the final paper based on the LaTeX document
Each visualization is saved as a separate image in the images directory
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as patches
from datetime import datetime
import pandas as pd
import os

# Set style for publication quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11

# Create output directory
output_dir = "road-distress-classification/finalPaper/mlds_final_project_template/images"
os.makedirs(output_dir, exist_ok=True)

def save_plot(filename, bbox_inches='tight', dpi=300):
    """Save plot with consistent settings"""
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches=bbox_inches, dpi=dpi, 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {filepath}")

# 1. Class Distribution Visualization
def create_class_distribution():
    """Create individual pie charts for each classification task"""
    
    # Damage Distribution
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    damage_data = [5971, 12202]
    damage_labels = ['Damaged\n(32.9%)', 'Not Damaged\n(67.1%)']
    colors = ['#FF6B6B', '#4ECDC4']
    
    wedges, texts, autotexts = ax.pie(damage_data, labels=damage_labels, colors=colors,
                                     autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
    ax.set_title('Damage Classification Distribution\n(18,173 total images)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add total count annotation
    ax.text(0, -1.4, f'Total Images: {sum(damage_data):,}', 
            ha='center', va='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    save_plot('damage_distribution.png')
    
    # Occlusion Distribution
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    occlusion_data = [3476, 14697]
    occlusion_labels = ['Occluded\n(19.1%)', 'Not Occluded\n(80.9%)']
    colors = ['#FFB347', '#87CEEB']
    
    wedges, texts, autotexts = ax.pie(occlusion_data, labels=occlusion_labels, colors=colors,
                                     autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
    ax.set_title('Occlusion Classification Distribution\n(18,173 total images)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.text(0, -1.4, f'Total Images: {sum(occlusion_data):,}', 
            ha='center', va='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    save_plot('occlusion_distribution.png')
    
    # Crop Distribution
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    crop_data = [778, 17395]
    crop_labels = ['Cropped\n(4.3%)', 'Not Cropped\n(95.7%)']
    colors = ['#DDA0DD', '#98FB98']
    
    wedges, texts, autotexts = ax.pie(crop_data, labels=crop_labels, colors=colors,
                                     autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
    ax.set_title('Crop Classification Distribution\n(18,173 total images)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.text(0, -1.4, f'Total Images: {sum(crop_data):,}', 
            ha='center', va='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    save_plot('crop_distribution.png')

# 2. Dataset Split Visualization
def create_dataset_split():
    """Create dataset split visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    splits = ['Training', 'Validation', 'Test']
    counts = [10901, 3640, 3632]
    percentages = [60.0, 20.0, 20.0]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(splits, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add percentage labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'{count:,}\n({pct}%)', ha='center', va='bottom',
                fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Split Distribution\n(Total: 18,173 images)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(counts) * 1.15)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    save_plot('dataset_split.png')

# 3. Model Performance Comparison
def create_model_comparison():
    """Create model performance comparison"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    
    models = ['Model B', 'Model H']
    macro_f1 = [0.806, 0.781]
    training_time = [1.26, 2.99]
    epochs = [21, 37]
    colors = ['#3498db', '#e74c3c']
    
    # Macro F1 Score
    bars1 = ax1.bar(models, macro_f1, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Macro F1 Score', fontweight='bold')
    ax1.set_title('Macro F1 Performance', fontweight='bold')
    ax1.set_ylim(0.75, 0.82)
    for bar, score in zip(bars1, macro_f1):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Training Time
    bars2 = ax2.bar(models, training_time, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Training Time (hours)', fontweight='bold')
    ax2.set_title('Training Duration', fontweight='bold')
    for bar, time in zip(bars2, training_time):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{time:.2f}h', ha='center', va='bottom', fontweight='bold')
    
    # Epochs
    bars3 = ax3.bar(models, epochs, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Training Epochs', fontweight='bold')
    ax3.set_title('Epochs to Convergence', fontweight='bold')
    for bar, epoch in zip(bars3, epochs):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{epoch}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Individual Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_plot('model_comparison.png')

# 4. Per-Class Threshold Results
def create_threshold_results():
    """Create per-class threshold optimization results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    classes = ['Damage', 'Occlusion', 'Crop']
    thresholds = [0.50, 0.40, 0.49]
    precisions = [0.54, 0.80, 0.99]
    recalls = [0.66, 0.75, 0.86]
    accuracies = [0.79, 0.93, 0.99]
    colors = ['#FF6B6B', '#FFB347', '#DDA0DD']
    
    # Thresholds
    bars1 = ax1.bar(classes, thresholds, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Threshold Value', fontweight='bold')
    ax1.set_title('Optimized Per-Class Thresholds', fontweight='bold')
    ax1.set_ylim(0, 0.6)
    for bar, thresh in zip(bars1, thresholds):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{thresh:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Precision
    bars2 = ax2.bar(classes, precisions, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision by Class', fontweight='bold')
    ax2.set_ylim(0, 1.1)
    for bar, prec in zip(bars2, precisions):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{prec:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Recall
    bars3 = ax3.bar(classes, recalls, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Recall', fontweight='bold')
    ax3.set_title('Recall by Class', fontweight='bold')
    ax3.set_ylim(0, 1.1)
    for bar, rec in zip(bars3, recalls):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{rec:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Accuracy
    bars4 = ax4.bar(classes, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Accuracy', fontweight='bold')
    ax4.set_title('Accuracy by Class', fontweight='bold')
    ax4.set_ylim(0, 1.1)
    for bar, acc in zip(bars4, accuracies):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{acc:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Balanced Per-Class Threshold Optimization Results', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_plot('threshold_results.png')

# 5. ROC AUC and Average Precision
def create_roc_ap_comparison():
    """Create ROC AUC and Average Precision comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    classes = ['Damage', 'Occlusion', 'Crop']
    roc_auc = [0.80, 0.94, 0.98]
    avg_precision = [0.61, 0.81, 0.93]
    colors = ['#FF6B6B', '#FFB347', '#DDA0DD']
    
    # ROC AUC
    bars1 = ax1.bar(classes, roc_auc, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('ROC AUC Score', fontweight='bold')
    ax1.set_title('ROC AUC by Class', fontweight='bold')
    ax1.set_ylim(0, 1.1)
    for bar, score in zip(bars1, roc_auc):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Average Precision
    bars2 = ax2.bar(classes, avg_precision, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Average Precision', fontweight='bold')
    ax2.set_title('Average Precision by Class', fontweight='bold')
    ax2.set_ylim(0, 1.1)
    for bar, score in zip(bars2, avg_precision):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Global Performance Metrics Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_plot('roc_ap_comparison.png')

# 6. Operational Performance Visualization
def create_operational_performance():
    """Create operational performance per 1000 images"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    classes = ['Damage', 'Occlusion', 'Crop']
    alerts_generated = [291, 146, 38]
    cases_missed = [82, 39, 6]
    colors = ['#FF6B6B', '#FFB347', '#DDA0DD']
    
    # Alerts Generated
    bars1 = ax1.bar(classes, alerts_generated, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Alerts Generated (per 1000 images)', fontweight='bold')
    ax1.set_title('Alert Generation Rate', fontweight='bold')
    for bar, alerts in zip(bars1, alerts_generated):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{alerts}', ha='center', va='bottom', fontweight='bold')
    
    # Cases Missed
    bars2 = ax2.bar(classes, cases_missed, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Cases Missed (per 1000 images)', fontweight='bold')
    ax2.set_title('Miss Rate', fontweight='bold')
    for bar, missed in zip(bars2, cases_missed):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{missed}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Operational Performance Analysis (per 1000 images)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_plot('operational_performance.png')

if __name__ == "__main__":
    print("Creating visualizations for final paper...")
    
    create_class_distribution()
    print("✓ Class distribution charts created")
    
    create_dataset_split()
    print("✓ Dataset split visualization created")
    
    create_model_comparison()
    print("✓ Model comparison charts created")
    
    create_threshold_results()
    print("✓ Threshold optimization results created")
    
    create_roc_ap_comparison()
    print("✓ ROC AUC and AP comparison created")
    
    create_operational_performance()
    print("✓ Operational performance visualization created")
    
    print(f"\nAll basic visualizations saved to: {output_dir}")
