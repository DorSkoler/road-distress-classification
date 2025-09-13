#!/usr/bin/env python3
"""
Create advanced visualizations for the final paper
Complex diagrams, timelines, and detailed analysis charts
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Arrow
import matplotlib.patches as patches
from datetime import datetime, timedelta
import pandas as pd
import os
from matplotlib.sankey import Sankey

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

# 7. Ensemble Architecture Diagram
def create_ensemble_architecture():
    """Create detailed ensemble architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Input Image
    input_box = FancyBboxPatch((4, 8.5), 2, 0.8, boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 8.9, 'Input Image\n256×256×3', ha='center', va='center', 
            fontweight='bold', fontsize=12)
    
    # Model B branch
    modelb_box = FancyBboxPatch((1, 6.5), 2.5, 1.2, boxstyle="round,pad=0.1", 
                                facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(modelb_box)
    ax.text(2.25, 7.1, 'Model B\nEfficientNet-B3\n+ Augmentation\n(Pure Features)', 
            ha='center', va='center', fontweight='bold', fontsize=10, color='white')
    
    # Model H branch
    modelh_box = FancyBboxPatch((6.5, 6.5), 2.5, 1.2, boxstyle="round,pad=0.1", 
                                facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(modelh_box)
    ax.text(7.75, 7.1, 'Model H\nEfficientNet-B3\n+ CLAHE + Masks\n(Enhanced Preproc)', 
            ha='center', va='center', fontweight='bold', fontsize=10, color='white')
    
    # Ensemble Averaging
    ensemble_box = FancyBboxPatch((3.5, 4.5), 3, 1, boxstyle="round,pad=0.1", 
                                  facecolor='#f39c12', edgecolor='black', linewidth=2)
    ax.add_patch(ensemble_box)
    ax.text(5, 5, 'Ensemble Averaging\n(0.5 Model B + 0.5 Model H)', 
            ha='center', va='center', fontweight='bold', fontsize=11, color='white')
    
    # Per-Class Thresholds
    threshold_box = FancyBboxPatch((2.5, 2.5), 5, 1.2, boxstyle="round,pad=0.1", 
                                   facecolor='#9b59b6', edgecolor='black', linewidth=2)
    ax.add_patch(threshold_box)
    ax.text(5, 3.1, 'Per-Class Thresholds\nDamage: 0.50 | Occlusion: 0.40 | Crop: 0.49\n(Balanced Optimization)', 
            ha='center', va='center', fontweight='bold', fontsize=11, color='white')
    
    # Final Output
    output_box = FancyBboxPatch((4, 0.5), 2, 0.8, boxstyle="round,pad=0.1", 
                                facecolor='#2ecc71', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 0.9, 'Final Predictions\n3-Class Output', ha='center', va='center', 
            fontweight='bold', fontsize=11, color='white')
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # Input to models
    ax.annotate('', xy=(2.25, 7.7), xytext=(4.5, 8.5), arrowprops=arrow_props)
    ax.annotate('', xy=(7.75, 7.7), xytext=(5.5, 8.5), arrowprops=arrow_props)
    
    # Models to ensemble
    ax.annotate('', xy=(4.2, 5.5), xytext=(2.25, 6.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5.8, 5.5), xytext=(7.75, 6.5), arrowprops=arrow_props)
    
    # Ensemble to thresholds
    ax.annotate('', xy=(5, 3.7), xytext=(5, 4.5), arrowprops=arrow_props)
    
    # Thresholds to output
    ax.annotate('', xy=(5, 1.3), xytext=(5, 2.5), arrowprops=arrow_props)
    
    # Add performance annotations
    ax.text(0.5, 6, 'F1: 0.806\n21 epochs', ha='center', va='center', 
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax.text(9.5, 6, 'F1: 0.781\n37 epochs', ha='center', va='center', 
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Final performance
    ax.text(8.5, 1, 'Final Accuracy:\n92.0%', ha='center', va='center', 
            fontsize=11, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.8))
    
    plt.title('Two-Model Ensemble Architecture with Per-Class Threshold Optimization', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    save_plot('ensemble_architecture.png')

# 8. Experimental Timeline
def create_experimental_timeline():
    """Create experimental evolution timeline"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Timeline data from paper
    phases = [
        ("2025-04-08", "Initial Development", "Project setup, data preprocessing, exploratory analysis", "#3498db"),
        ("2025-04-27", "Model Training Foundation", "EfficientNet-B3, ResNet50, basic training pipelines", "#e74c3c"),
        ("2025-05-10", "Mask-Enhanced Training", "U-Net integration, road masks, +7.64% improvement", "#2ecc71"),
        ("2025-06-28", "Smart Data Splitting", "Road-wise splits, 97.36% mask success, zero leakage", "#f39c12"),
        ("2025-07-05", "Hybrid Training Approach", "Models A-H variants, cross-platform compatibility", "#9b59b6"),
        ("2025-08-01", "Ensemble Breakthrough", "Per-class thresholds, 63.3%→92% accuracy jump", "#e67e22")
    ]
    
    y_positions = np.arange(len(phases))
    colors = [phase[3] for phase in phases]
    
    # Create horizontal timeline
    for i, (date, title, description, color) in enumerate(phases):
        # Main timeline bar
        ax.barh(i, 1, height=0.6, color=color, alpha=0.7, edgecolor='black')
        
        # Phase title
        ax.text(0.05, i, f"{title}\n({date})", ha='left', va='center', 
                fontweight='bold', fontsize=11, color='white')
        
        # Description on the right
        ax.text(1.1, i, description, ha='left', va='center', 
                fontsize=10, wrap=True)
    
    # Breakthrough annotation
    ax.annotate('BREAKTHROUGH:\n+28.7% Accuracy', 
                xy=(0.5, 5), xytext=(1.5, 5.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=12, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.5, len(phases) - 0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.title('Experimental Evolution Timeline\nSystematic Methodology Leading to Breakthrough', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    save_plot('experimental_timeline.png')

# 9. Accuracy Breakthrough Visualization
def create_accuracy_breakthrough():
    """Create dramatic accuracy improvement visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    approaches = ['Uniform\nThresholds\n(0.5)', 'Optimized\nPer-Class\nThresholds']
    accuracies = [63.3, 92.0]
    colors = ['#e74c3c', '#2ecc71']
    
    # Create bars with emphasis on the breakthrough
    bars = ax.bar(approaches, accuracies, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=2, width=0.6)
    
    # Add dramatic styling
    bars[1].set_height(92.0)
    bars[1].set_color('#2ecc71')
    
    # Add value labels with emphasis
    ax.text(0, 63.3 + 2, '63.3%', ha='center', va='bottom', 
            fontsize=14, fontweight='bold')
    ax.text(1, 92.0 + 2, '92.0%\n(+28.7%)', ha='center', va='bottom', 
            fontsize=14, fontweight='bold', color='green')
    
    # Add breakthrough arrow and annotation
    ax.annotate('BREAKTHROUGH!', xy=(1, 92), xytext=(1.3, 85),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'),
                fontsize=16, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.9))
    
    # Styling
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('The Per-Class Threshold Breakthrough\nTransforming Multi-Label Classification Performance', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add improvement percentage as background text
    ax.text(0.5, 30, '+28.7%\nImprovement', ha='center', va='center', 
            fontsize=24, fontweight='bold', alpha=0.3, color='green',
            rotation=15)
    
    plt.tight_layout()
    save_plot('accuracy_breakthrough.png')

# 10. Alternative Threshold Strategies
def create_threshold_strategies():
    """Create comparison of different threshold strategies"""
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
    
    classes = ['Damage', 'Occlusion', 'Crop']
    colors = ['#FF6B6B', '#FFB347', '#DDA0DD']
    
    # Balanced Mode (Current)
    balanced_precision = [0.54, 0.80, 0.99]
    balanced_recall = [0.66, 0.75, 0.86]
    balanced_thresholds = [0.50, 0.40, 0.49]
    
    # High-Recall Mode
    high_recall_precision = [0.32, 0.52, 0.78]
    high_recall_recall = [0.90, 0.91, 0.90]
    high_recall_thresholds = [0.12, 0.10, 0.25]
    
    # High-Precision Mode  
    high_precision_precision = [0.80, 0.90, 0.90]
    high_precision_recall = [0.19, 0.49, 0.87]
    high_precision_thresholds = [0.89, 0.64, 0.38]
    
    # Balanced Mode Charts
    ax1.bar(classes, balanced_thresholds, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Balanced Mode\nThresholds', fontweight='bold')
    ax1.set_ylabel('Threshold', fontweight='bold')
    for i, v in enumerate(balanced_thresholds):
        ax1.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.bar(classes, balanced_precision, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Balanced Mode\nPrecision', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    for i, v in enumerate(balanced_precision):
        ax2.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.bar(classes, balanced_recall, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_title('Balanced Mode\nRecall', fontweight='bold')
    ax3.set_ylabel('Recall', fontweight='bold')
    for i, v in enumerate(balanced_recall):
        ax3.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # High-Recall Mode Charts
    ax4.bar(classes, high_recall_thresholds, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_title('High-Recall Mode\nThresholds', fontweight='bold')
    ax4.set_ylabel('Threshold', fontweight='bold')
    for i, v in enumerate(high_recall_thresholds):
        ax4.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax5.bar(classes, high_recall_precision, color=colors, alpha=0.8, edgecolor='black')
    ax5.set_title('High-Recall Mode\nPrecision', fontweight='bold')
    ax5.set_ylabel('Precision', fontweight='bold')
    for i, v in enumerate(high_recall_precision):
        ax5.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # High-Precision comparison in ax6
    width = 0.25
    x = np.arange(len(classes))
    
    ax6.bar(x - width, high_precision_precision, width, label='High-Precision P', 
            color=['#FF6B6B', '#FFB347', '#DDA0DD'], alpha=0.8, edgecolor='black')
    ax6.bar(x, high_precision_recall, width, label='High-Precision R', 
            color=['#FF6B6B', '#FFB347', '#DDA0DD'], alpha=0.6, edgecolor='black')
    ax6.bar(x + width, high_recall_recall, width, label='High-Recall R', 
            color=['#FF6B6B', '#FFB347', '#DDA0DD'], alpha=0.4, edgecolor='black')
    
    ax6.set_title('Strategy Comparison\n(P vs R)', fontweight='bold')
    ax6.set_ylabel('Score', fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(classes)
    ax6.legend(fontsize=9)
    ax6.set_ylim(0, 1.0)
    
    # Set consistent y-limits
    for ax in [ax1, ax4]:
        ax.set_ylim(0, 1.0)
    for ax in [ax2, ax3, ax5]:
        ax.set_ylim(0, 1.0)
    
    plt.suptitle('Alternative Threshold Strategies Comparison\nBalanced vs High-Recall vs High-Precision Modes', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_plot('threshold_strategies_comparison.png')

# 11. Individual Model Performance Details
def create_individual_model_details():
    """Create detailed individual model performance comparison"""
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))
    
    classes = ['Damage', 'Occlusion', 'Crop']
    colors = ['#FF6B6B', '#FFB347', '#DDA0DD']
    
    # Model B Performance
    model_b_precision = [63.6, 80.1, 97.5]
    model_b_recall = [65.8, 80.5, 96.3]
    model_b_f1 = [64.7, 80.3, 96.9]
    
    # Model H Performance
    model_h_precision = [57.8, 81.7, 97.4]
    model_h_recall = [64.3, 73.8, 94.4]
    model_h_f1 = [60.9, 77.6, 95.9]
    
    # Model B Charts
    ax1.bar(classes, model_b_precision, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Model B - Precision (%)', fontweight='bold')
    ax1.set_ylabel('Precision (%)', fontweight='bold')
    for i, v in enumerate(model_b_precision):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.bar(classes, model_b_recall, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Model B - Recall (%)', fontweight='bold')
    ax2.set_ylabel('Recall (%)', fontweight='bold')
    for i, v in enumerate(model_b_recall):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax3.bar(classes, model_b_f1, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_title('Model B - F1 Score (%)', fontweight='bold')
    ax3.set_ylabel('F1 Score (%)', fontweight='bold')
    for i, v in enumerate(model_b_f1):
        ax3.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Model H Charts
    ax4.bar(classes, model_h_precision, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_title('Model H - Precision (%)', fontweight='bold')
    ax4.set_ylabel('Precision (%)', fontweight='bold')
    for i, v in enumerate(model_h_precision):
        ax4.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax5.bar(classes, model_h_recall, color=colors, alpha=0.8, edgecolor='black')
    ax5.set_title('Model H - Recall (%)', fontweight='bold')
    ax5.set_ylabel('Recall (%)', fontweight='bold')
    for i, v in enumerate(model_h_recall):
        ax5.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax6.bar(classes, model_h_f1, color=colors, alpha=0.8, edgecolor='black')
    ax6.set_title('Model H - F1 Score (%)', fontweight='bold')
    ax6.set_ylabel('F1 Score (%)', fontweight='bold')
    for i, v in enumerate(model_h_f1):
        ax6.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Set consistent y-limits
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_ylim(0, 105)
    
    plt.suptitle('Individual Model Performance Comparison\nModel B (Pure Features) vs Model H (Enhanced Preprocessing)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_plot('individual_model_details.png')

if __name__ == "__main__":
    print("Creating advanced visualizations for final paper...")
    
    create_ensemble_architecture()
    print("✓ Ensemble architecture diagram created")
    
    create_experimental_timeline()
    print("✓ Experimental timeline visualization created")
    
    create_accuracy_breakthrough()
    print("✓ Accuracy breakthrough visualization created")
    
    create_threshold_strategies()
    print("✓ Threshold strategies comparison created")
    
    create_individual_model_details()
    print("✓ Individual model performance details created")
    
    print(f"\nAll advanced visualizations saved to: {output_dir}")
