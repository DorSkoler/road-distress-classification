#!/usr/bin/env python3
"""
Create simple comparative performance visualization - ROC AUC and Average Precision only
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Output directory
output_dir = "mlds_final_project_template/images"
os.makedirs(output_dir, exist_ok=True)

def create_comparative_performance():
    """Create single focused comparative performance chart"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Data from the original analysis
    classes = ['Damage', 'Occlusion', 'Crop']
    roc_auc = [0.80, 0.94, 0.98]
    avg_precision = [0.61, 0.81, 0.93]
    colors = ['#FF6B6B', '#FFB347', '#DDA0DD']
    
    # Create grouped bar chart
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, roc_auc, width, label='ROC AUC Score', 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, avg_precision, width, label='Average Precision Score', 
                   color=colors, alpha=0.6, edgecolor='black', linewidth=2, hatch='///')
    
    # Styling
    ax.set_ylabel('Performance Score', fontweight='bold', fontsize=14)
    ax.set_xlabel('Classification Task', fontweight='bold', fontsize=14)
    ax.set_title('Global Performance Metrics - Comparative Analysis\nROC AUC and Average Precision Scores', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    # Add value labels
    for bar, score in zip(bars1, roc_auc):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    for bar, score in zip(bars2, avg_precision):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add performance interpretation
    ax.text(0.5, -0.12, 
            'ROC AUC measures overall class separability; Average Precision focuses on positive class performance.\n' +
            'Higher scores indicate better classification capability and model reliability.',
            transform=ax.transAxes, ha='center', va='top', fontsize=11, style='italic',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save with the same name as original to replace it
    filepath = os.path.join(output_dir, 'performance_metrics_analysis.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved comparative performance chart: {filepath}")

if __name__ == "__main__":
    print("Creating focused comparative performance visualization...")
    create_comparative_performance()
    print("✓ Comparative performance analysis completed!")
