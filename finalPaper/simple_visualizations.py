#!/usr/bin/env python3
"""
Simple visualizations for the final paper - minimal dependencies
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
output_dir = os.path.join(os.getcwd(), "mlds_final_project_template", "images")
os.makedirs(output_dir, exist_ok=True)

def save_plot(filename):
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {filename}")

# 1. Class Distribution - Combined View
def create_class_distributions():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Damage Distribution
    damage_data = [5971, 12202]
    damage_labels = ['Damaged\n(32.9%)', 'Not Damaged\n(67.1%)']
    ax1.pie(damage_data, labels=damage_labels, autopct='%1.1f%%', startangle=90,
           colors=['#FF6B6B', '#4ECDC4'])
    ax1.set_title('Damage Classification\n(18,173 images)')
    
    # Occlusion Distribution  
    occlusion_data = [3476, 14697]
    occlusion_labels = ['Occluded\n(19.1%)', 'Not Occluded\n(80.9%)']
    ax2.pie(occlusion_data, labels=occlusion_labels, autopct='%1.1f%%', startangle=90,
           colors=['#FFB347', '#87CEEB'])
    ax2.set_title('Occlusion Classification\n(18,173 images)')
    
    # Crop Distribution
    crop_data = [778, 17395] 
    crop_labels = ['Cropped\n(4.3%)', 'Not Cropped\n(95.7%)']
    ax3.pie(crop_data, labels=crop_labels, autopct='%1.1f%%', startangle=90,
           colors=['#DDA0DD', '#98FB98'])
    ax3.set_title('Crop Classification\n(18,173 images)')
    
    plt.suptitle('Class Distribution Analysis Across All Classification Tasks', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_plot('class_distributions.png')

# 2. Dataset Split
def create_dataset_split():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    splits = ['Training\n(60%)', 'Validation\n(20%)', 'Test\n(20%)']
    counts = [10901, 3640, 3632]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(splits, counts, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Number of Images')
    ax.set_title('Dataset Split Distribution (Total: 18,173 images)')
    plt.tight_layout()
    save_plot('dataset_split.png')

# 3. Model Performance Comparison
def create_model_performance():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Model B\n(Pure Features)', 'Model H\n(Enhanced Preprocessing)']
    f1_scores = [0.806, 0.781]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(models, f1_scores, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('Individual Model Performance Comparison')
    ax.set_ylim(0.75, 0.82)
    plt.tight_layout()
    save_plot('model_performance.png')

# 4. Threshold Results
def create_threshold_results():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    classes = ['Damage', 'Occlusion', 'Crop']
    thresholds = [0.50, 0.40, 0.49]
    precisions = [0.54, 0.80, 0.99]
    recalls = [0.66, 0.75, 0.86]
    accuracies = [0.79, 0.93, 0.99]
    colors = ['#FF6B6B', '#FFB347', '#DDA0DD']
    
    # Thresholds
    ax1.bar(classes, thresholds, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Optimized Thresholds')
    ax1.set_ylabel('Threshold Value')
    for i, v in enumerate(thresholds):
        ax1.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Precision
    ax2.bar(classes, precisions, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Precision by Class')
    ax2.set_ylabel('Precision')
    for i, v in enumerate(precisions):
        ax2.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Recall
    ax3.bar(classes, recalls, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_title('Recall by Class')
    ax3.set_ylabel('Recall')
    for i, v in enumerate(recalls):
        ax3.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Accuracy
    ax4.bar(classes, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_title('Accuracy by Class')
    ax4.set_ylabel('Accuracy')
    for i, v in enumerate(accuracies):
        ax4.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Per-Class Threshold Optimization Results', fontweight='bold')
    plt.tight_layout()
    save_plot('threshold_optimization.png')

# 5. ROC AUC and AP
def create_performance_metrics():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    classes = ['Damage', 'Occlusion', 'Crop']
    roc_auc = [0.80, 0.94, 0.98]
    avg_precision = [0.61, 0.81, 0.93]
    colors = ['#FF6B6B', '#FFB347', '#DDA0DD']
    
    # ROC AUC
    ax1.bar(classes, roc_auc, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('ROC AUC by Class')
    ax1.set_ylabel('ROC AUC Score')
    for i, v in enumerate(roc_auc):
        ax1.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Average Precision
    ax2.bar(classes, avg_precision, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Average Precision by Class') 
    ax2.set_ylabel('Average Precision')
    for i, v in enumerate(avg_precision):
        ax2.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Global Performance Metrics', fontweight='bold')
    plt.tight_layout()
    save_plot('performance_metrics.png')

# 6. Breakthrough Visualization
def create_breakthrough():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    approaches = ['Standard\nUniform Thresholds', 'Optimized\nPer-Class Thresholds']
    accuracies = [63.3, 92.0]
    colors = ['#e74c3c', '#2ecc71']
    
    bars = ax.bar(approaches, accuracies, color=colors, alpha=0.8, edgecolor='black')
    
    # Add dramatic styling
    ax.text(0, 65, '63.3%', ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax.text(1, 94, '92.0%\n(+28.7%)', ha='center', va='bottom', 
            fontweight='bold', fontsize=14, color='darkgreen')
    
    ax.set_ylabel('Overall Accuracy (%)')
    ax.set_title('The Per-Class Threshold Breakthrough\n+28.7% Accuracy Improvement')
    ax.set_ylim(0, 100)
    
    # Add improvement arrow
    ax.annotate('', xy=(1, 85), xytext=(0.3, 70),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(0.6, 77, '+28.7%\nImprovement!', ha='center', va='center',
            fontweight='bold', color='green', fontsize=12)
    
    plt.tight_layout()
    save_plot('accuracy_breakthrough.png')

# 7. Ensemble Architecture (Simplified)
def create_ensemble_diagram():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Draw boxes
    # Input
    input_rect = plt.Rectangle((4, 6.5), 2, 1, fill=True, facecolor='lightblue', 
                              edgecolor='black', linewidth=2)
    ax.add_patch(input_rect)
    ax.text(5, 7, 'Input Image\n256×256×3', ha='center', va='center', fontweight='bold')
    
    # Model B
    modelb_rect = plt.Rectangle((1, 4.5), 2.5, 1.5, fill=True, facecolor='#3498db', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(modelb_rect)
    ax.text(2.25, 5.25, 'Model B\nEfficientNet-B3\n(Pure Features)', 
            ha='center', va='center', fontweight='bold', color='white')
    
    # Model H
    modelh_rect = plt.Rectangle((6.5, 4.5), 2.5, 1.5, fill=True, facecolor='#e74c3c', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(modelh_rect)
    ax.text(7.75, 5.25, 'Model H\nEfficientNet-B3\n(CLAHE + Masks)', 
            ha='center', va='center', fontweight='bold', color='white')
    
    # Ensemble
    ensemble_rect = plt.Rectangle((3.5, 2.5), 3, 1, fill=True, facecolor='#f39c12', 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(ensemble_rect)
    ax.text(5, 3, 'Ensemble (0.5 + 0.5)', ha='center', va='center', 
            fontweight='bold', color='white')
    
    # Thresholds
    threshold_rect = plt.Rectangle((2.5, 0.5), 5, 1.2, fill=True, facecolor='#9b59b6', 
                                  edgecolor='black', linewidth=2)
    ax.add_patch(threshold_rect)
    ax.text(5, 1.1, 'Per-Class Thresholds\nDam:0.50 | Occ:0.40 | Crop:0.49', 
            ha='center', va='center', fontweight='bold', color='white')
    
    # Draw arrows
    ax.arrow(4.5, 6.5, -1.8, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(5.5, 6.5, 1.8, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(2.25, 4.5, 1.5, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(7.75, 4.5, -1.5, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(5, 2.5, 0, -0.6, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    plt.title('Two-Model Ensemble Architecture', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot('ensemble_architecture.png')

if __name__ == "__main__":
    print(f"Creating visualizations in: {output_dir}")
    
    try:
        create_class_distributions()
        print("✓ Class distributions created")
        
        create_dataset_split()
        print("✓ Dataset split created")
        
        create_model_performance()
        print("✓ Model performance created")
        
        create_threshold_results()
        print("✓ Threshold results created")
        
        create_performance_metrics()
        print("✓ Performance metrics created")
        
        create_breakthrough()
        print("✓ Breakthrough visualization created")
        
        create_ensemble_diagram()
        print("✓ Ensemble diagram created")
        
        print("\nAll visualizations completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
