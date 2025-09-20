#!/usr/bin/env python3
"""
Create comparative performance visualization only - focused on ROC AUC and Average Precision comparison
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for publication quality figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# Create output directory
output_dir = "road-distress-classification/finalPaper/mlds_final_project_template/images"
os.makedirs(output_dir, exist_ok=True)

def save_plot(filename):
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {filename}")

def create_comparative_performance_only():
    """Create focused comparative performance visualization with ROC AUC and Average Precision"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    classes = ['Damage', 'Occlusion', 'Crop']
    colors = ['#FF6B6B', '#FFB347', '#DDA0DD']
    
    # ROC AUC Scores
    roc_auc = [0.80, 0.94, 0.98]
    bars1 = ax1.bar(classes, roc_auc, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
    ax1.set_ylabel('ROC AUC Score', fontweight='bold', fontsize=16)
    ax1.set_xlabel('Classification Task', fontweight='bold', fontsize=16)
    ax1.set_title('ROC AUC Performance Comparison\nArea Under Receiver Operating Characteristic Curve', 
                  fontweight='bold', fontsize=18)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add performance quality annotations with values
    for bar, score in zip(bars1, roc_auc):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=16)
    
    # Add quality indicators
    quality_zones = [(0.0, 0.6, 'Poor', 'lightcoral'), (0.6, 0.8, 'Fair', 'lightyellow'), 
                     (0.8, 0.9, 'Good', 'lightgreen'), (0.9, 1.0, 'Excellent', 'lightblue')]
    for start, end, label, color in quality_zones:
        ax1.axhspan(start, end, alpha=0.1, color=color)
    
    # Average Precision Scores
    avg_precision = [0.61, 0.81, 0.93]
    bars2 = ax2.bar(classes, avg_precision, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
    ax2.set_ylabel('Average Precision Score', fontweight='bold', fontsize=16)
    ax2.set_xlabel('Classification Task', fontweight='bold', fontsize=16)
    ax2.set_title('Average Precision Comparison\nArea Under Precision-Recall Curve', fontweight='bold', fontsize=18)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars2, avg_precision):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=16)
    
    # Add quality zones for average precision as well
    for start, end, label, color in quality_zones:
        ax2.axhspan(start, end, alpha=0.1, color=color)
    
    # Add explanatory text
    fig.text(0.5, 0.02, 
             'ROC AUC measures overall classification performance across all thresholds. Average Precision focuses on positive class performance.\n' +
             'Higher scores indicate better class separability and detection capability. Crop detection shows excellent performance,\n' +
             'while damage detection presents the most challenging classification task due to class imbalance and subtle visual differences.',
             ha='center', va='bottom', fontsize=12, style='italic',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Global Performance Metrics - Comparative Analysis\nROC AUC and Average Precision Scores Across Classification Tasks', 
                 fontsize=20, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.85)
    save_plot('performance_metrics_analysis.png')

def create_combined_comparative_metrics():
    """Create a single comprehensive comparative visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    classes = ['Damage\nDetection', 'Occlusion\nDetection', 'Crop\nDetection']
    roc_auc = [0.80, 0.94, 0.98]
    avg_precision = [0.61, 0.81, 0.93]
    
    x = np.arange(len(classes))
    width = 0.35
    
    # Create grouped bar chart
    bars1 = ax.bar(x - width/2, roc_auc, width, label='ROC AUC Score', 
                   color=['#FF6B6B', '#FFB347', '#DDA0DD'], alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, avg_precision, width, label='Average Precision Score', 
                   color=['#FF6B6B', '#FFB347', '#DDA0DD'], alpha=0.6, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Performance Score (0.0 - 1.0)', fontweight='bold', fontsize=16)
    ax.set_xlabel('Classification Task', fontweight='bold', fontsize=16)
    ax.set_title('Comparative Performance Analysis\nROC AUC vs Average Precision Across All Classification Tasks', 
                 fontweight='bold', fontsize=18, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(loc='upper left', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    # Add value labels on bars
    for bar, score in zip(bars1, roc_auc):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=13)
    
    for bar, score in zip(bars2, avg_precision):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=13)
    
    # Add performance quality zones as horizontal bands
    ax.axhspan(0.0, 0.6, alpha=0.05, color='red', label='Poor Performance')
    ax.axhspan(0.6, 0.8, alpha=0.05, color='orange', label='Fair Performance') 
    ax.axhspan(0.8, 0.9, alpha=0.05, color='yellow', label='Good Performance')
    ax.axhspan(0.9, 1.0, alpha=0.05, color='green', label='Excellent Performance')
    
    # Add performance indicators
    performance_indicators = ['Good\n(0.80)', 'Excellent\n(0.94, 0.81)', 'Outstanding\n(0.98, 0.93)']
    for i, indicator in enumerate(performance_indicators):
        ax.text(i, 0.15, indicator, ha='center', va='center', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='black'))
    
    # Add explanatory text
    ax.text(0.5, -0.15, 
            'Key Insights: ROC AUC measures overall discriminative ability across all thresholds.\n' +
            'Average Precision emphasizes performance on the positive (minority) class.\n' +
            'Crop classification achieves near-perfect performance, while damage detection remains challenging due to class imbalance.',
            transform=ax.transAxes, ha='center', va='top', fontsize=12, style='italic',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    save_plot('comparative_performance_metrics.png')

if __name__ == "__main__":
    print("Creating focused comparative performance visualizations...")
    print(f"Output directory: {output_dir}")
    
    try:
        # Create the original format but with comparative focus
        create_comparative_performance_only()
        print("✓ Comparative performance analysis (2-panel) created")
        
        # Create a single comprehensive comparison
        create_combined_comparative_metrics()
        print("✓ Combined comparative metrics visualization created")
        
        print("\n" + "="*60)
        print("Comparative performance visualizations completed!")
        print("Files created:")
        print("- performance_metrics_analysis.png (replaces original)")
        print("- comparative_performance_metrics.png (new combined view)")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
