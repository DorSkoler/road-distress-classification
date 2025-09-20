#!/usr/bin/env python3
"""
Supplementary visualizations for the final paper
Additional charts for experimental timeline, ensemble architecture, and strategy comparisons
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Set style for publication quality figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11

# Create output directory
output_dir = os.path.join(os.getcwd(), "mlds_final_project_template", "images")
os.makedirs(output_dir, exist_ok=True)

def save_plot(filename):
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {filename}")

# 1. Experimental Evolution Timeline
def create_experimental_timeline():
    """Create detailed experimental evolution timeline"""
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    
    # Timeline data from paper with more details
    phases = [
        {
            'date': '2025-04-08',
            'title': 'Initial Development',
            'description': 'Project setup, data preprocessing,\nexploratory analysis',
            'outcomes': 'Established project architecture\nand data processing pipelines',
            'color': '#3498db'
        },
        {
            'date': '2025-04-27', 
            'title': 'Model Training Foundation',
            'description': 'EfficientNet-B3, ResNet50,\nbasic training pipelines',
            'outcomes': 'Baseline performance metrics\nestablished',
            'color': '#e74c3c'
        },
        {
            'date': '2025-05-10',
            'title': 'Mask-Enhanced Training',
            'description': 'U-Net integration,\nroad masks implementation',
            'outcomes': '88.99% accuracy achieved\n(+7.64% improvement)',
            'color': '#2ecc71'
        },
        {
            'date': '2025-06-28',
            'title': 'Smart Data Splitting',
            'description': 'Road-wise splits, quality filtering,\nA/B testing framework',
            'outcomes': '97.36% mask success rate\nZero data leakage',
            'color': '#f39c12'
        },
        {
            'date': '2025-07-05',
            'title': 'Hybrid Training Approach',
            'description': 'Models A-H variants,\ncross-platform compatibility',
            'outcomes': 'Models B & H identified\nas top performers',
            'color': '#9b59b6'
        },
        {
            'date': '2025-08-01',
            'title': 'Ensemble Results',
            'description': 'Per-class threshold optimization,\nensemble design',
            'outcomes': '63.3% → 92% accuracy\n(+28.7% improvement)',
            'color': '#e67e22'
        }
    ]
    
    # Create timeline
    y_positions = np.arange(len(phases))[::-1]  # Reverse for chronological order
    
    # Main timeline bars with enhanced styling
    for i, phase in enumerate(phases):
        y_pos = y_positions[i]
        
        # Main phase bar with gradient effect
        phase_bar = FancyBboxPatch((0.5, y_pos-0.4), 9, 0.8, 
                                  boxstyle="round,pad=0.08", 
                                  facecolor=phase['color'], 
                                  edgecolor='black', 
                                  linewidth=2, alpha=0.9)
        ax.add_patch(phase_bar)
        
        # Phase title and date with proportional font
        ax.text(1, y_pos, f"{phase['title']}\n({phase['date']})", 
                ha='left', va='center', fontweight='bold', fontsize=11, color='white')
        
        # Description with proportional spacing
        ax.text(5, y_pos, phase['description'], 
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        
        # Enhanced outcomes box
        outcome_box = FancyBboxPatch((10, y_pos-0.35), 5, 0.7, 
                                    boxstyle="round,pad=0.08", 
                                    facecolor='white', 
                                    edgecolor=phase['color'], 
                                    linewidth=2, alpha=0.95)
        ax.add_patch(outcome_box)
        ax.text(12.5, y_pos, phase['outcomes'], 
                ha='center', va='center', fontsize=10, fontweight='bold', color=phase['color'])
        
        # Remove annotation for cleaner look
    
    # Enhanced timeline connector line
    ax.plot([0.3, 0.3], [y_positions[-1]-0.5, y_positions[0]+0.5], 
            'k-', linewidth=4, alpha=0.6)
    
    # Add proportional progress markers
    for i, y_pos in enumerate(y_positions):
        ax.plot([0.2, 0.4], [y_pos, y_pos], 'ko-', markersize=8, linewidth=2)
        ax.text(0.1, y_pos, f'{i+1}', ha='center', va='center', fontsize=10, 
                fontweight='bold', color='white',
                bbox=dict(boxstyle="circle,pad=0.08", facecolor='black', alpha=0.8))
    
    # Enhanced styling
    ax.set_xlim(-0.2, 17)
    ax.set_ylim(-0.8, len(phases) - 0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Proportional column headers
    ax.text(5, len(phases) + 0.2, 'Experimental Phase & Activities', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.8))
    ax.text(12.5, len(phases) + 0.2, 'Key Outcomes & Results', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral', alpha=0.8))
    
    # Remove title to prevent overlap
    
    # Remove bottom annotation to clean up the visualization
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05, top=0.95)
    save_plot('experimental_timeline_detailed.png')

# 2. Ensemble Architecture Diagram
def create_ensemble_architecture_detailed():
    """Create detailed ensemble architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Input Layer
    input_box = FancyBboxPatch((4.5, 8.5), 3, 1, boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(6, 9, 'Input Road Image\n256×256×3 RGB', ha='center', va='center', 
            fontweight='bold', fontsize=13)
    
    # Model B Branch (Left)
    modelb_preproc = FancyBboxPatch((0.5, 6.8), 2.5, 0.8, boxstyle="round,pad=0.1", 
                                   facecolor='#85C1E9', edgecolor='black', linewidth=1)
    ax.add_patch(modelb_preproc)
    ax.text(1.75, 7.2, 'Standard Preprocessing\nResize + Normalize', ha='center', va='center', 
            fontweight='bold', fontsize=10)
    
    modelb_box = FancyBboxPatch((0.5, 5.5), 2.5, 1.8, boxstyle="round,pad=0.1", 
                                facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(modelb_box)
    ax.text(1.75, 6.4, 'Model B\nEfficientNet-B3\n(Pure Features)\n\n• No masking\n• Standard augmentation\n• F1: 0.806', 
            ha='center', va='center', fontweight='bold', fontsize=10, color='white')
    
    # Model H Branch (Right)
    modelh_preproc = FancyBboxPatch((9, 6.8), 2.5, 0.8, boxstyle="round,pad=0.1", 
                                   facecolor='#F1948A', edgecolor='black', linewidth=1)
    ax.add_patch(modelh_preproc)
    ax.text(10.25, 7.2, 'Enhanced Preprocessing\nCLAHE + Road Masking', ha='center', va='center', 
            fontweight='bold', fontsize=10)
    
    modelh_box = FancyBboxPatch((9, 5.5), 2.5, 1.8, boxstyle="round,pad=0.1", 
                                facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(modelh_box)
    ax.text(10.25, 6.4, 'Model H\nEfficientNet-B3\n(Enhanced Preprocessing)\n\n• CLAHE enhancement\n• Partial road masking\n• F1: 0.781', 
            ha='center', va='center', fontweight='bold', fontsize=10, color='white')
    
    # Ensemble Averaging Layer
    ensemble_box = FancyBboxPatch((4, 3.5), 4, 1.2, boxstyle="round,pad=0.1", 
                                  facecolor='#f39c12', edgecolor='black', linewidth=2)
    ax.add_patch(ensemble_box)
    ax.text(6, 4.1, 'Ensemble Averaging Layer\n(0.5 × Model B) + (0.5 × Model H)\nCombines complementary strengths', 
            ha='center', va='center', fontweight='bold', fontsize=12, color='white')
    
    # Per-Class Threshold Layer
    threshold_box = FancyBboxPatch((3, 1.5), 6, 1.5, boxstyle="round,pad=0.1", 
                                   facecolor='#9b59b6', edgecolor='black', linewidth=2)
    ax.add_patch(threshold_box)
    ax.text(6, 2.25, 'Per-Class Decision Thresholds\n(Optimized for Balanced Performance)\n\nDamage: τ=0.50 | Occlusion: τ=0.40 | Crop: τ=0.49', 
            ha='center', va='center', fontweight='bold', fontsize=11, color='white')
    
    # Final Output
    output_box = FancyBboxPatch((4.5, 0.2), 3, 0.8, boxstyle="round,pad=0.1", 
                                facecolor='#2ecc71', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(6, 0.6, 'Final Multi-Label Predictions\n92% Overall Accuracy', ha='center', va='center', 
            fontweight='bold', fontsize=12, color='white')
    
    # Connection arrows with labels
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # Input to preprocessing
    ax.annotate('', xy=(1.75, 7.6), xytext=(5.2, 8.5), arrowprops=arrow_props)
    ax.annotate('', xy=(10.25, 7.6), xytext=(6.8, 8.5), arrowprops=arrow_props)
    
    # Preprocessing to models
    ax.annotate('', xy=(1.75, 7.3), xytext=(1.75, 6.8), arrowprops=arrow_props)
    ax.annotate('', xy=(10.25, 7.3), xytext=(10.25, 6.8), arrowprops=arrow_props)
    
    # Models to ensemble
    ax.annotate('', xy=(4.5, 4.5), xytext=(1.75, 5.5), arrowprops=arrow_props)
    ax.annotate('', xy=(7.5, 4.5), xytext=(10.25, 5.5), arrowprops=arrow_props)
    
    # Ensemble to thresholds
    ax.annotate('', xy=(6, 3.0), xytext=(6, 3.5), arrowprops=arrow_props)
    
    # Thresholds to output
    ax.annotate('', xy=(6, 1.0), xytext=(6, 1.5), arrowprops=arrow_props)
    
    # Add performance annotations
    ax.text(0.2, 4.5, 'Training:\n21 epochs\n1.26 hours', ha='center', va='center', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax.text(11.8, 4.5, 'Training:\n37 epochs\n2.99 hours', ha='center', va='center', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add architectural insight
    ax.text(6, 9.7, 'Two-Model Ensemble Architecture\nComplementary Approaches for Optimal Performance', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add key insight box
    insight_text = ("Key Innovation: Combining pure feature learning (Model B) with enhanced preprocessing (Model H)\n"
                   "achieves better performance than either approach alone through complementary strengths.")
    ax.text(6, -0.5, insight_text, ha='center', va='center', fontsize=11, style='italic',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    save_plot('ensemble_architecture_detailed.png')

# 3. Alternative Threshold Strategies Comparison
def create_threshold_strategies_comparison():
    """Create comprehensive threshold strategies comparison"""
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
    
    classes = ['Damage', 'Occlusion', 'Crop']
    colors = ['#FF6B6B', '#FFB347', '#DDA0DD']
    
    # Strategy 1: Balanced Mode (Current Approach)
    balanced_thresholds = [0.50, 0.40, 0.49]
    balanced_precision = [0.54, 0.80, 0.99]
    balanced_recall = [0.66, 0.75, 0.86]
    
    # Strategy 2: High-Recall Mode (Safety-First)
    high_recall_thresholds = [0.12, 0.10, 0.25]
    high_recall_precision = [0.32, 0.52, 0.78]
    high_recall_recall = [0.90, 0.91, 0.90]
    
    # Strategy 3: High-Precision Mode (Quality-First)
    high_precision_thresholds = [0.89, 0.64, 0.38]
    high_precision_precision = [0.80, 0.90, 0.90]
    high_precision_recall = [0.19, 0.49, 0.87]
    
    # Balanced Mode Visualization
    bars1 = ax1.bar(classes, balanced_precision, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Precision Score', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Classification Task', fontweight='bold', fontsize=12)
    ax1.set_title('Balanced Mode Precision\n(General Monitoring)', fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, prec, thresh in zip(bars1, balanced_precision, balanced_thresholds):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                f'{prec:.2f}\n(τ={thresh:.2f})', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    bars2 = ax2.bar(classes, balanced_recall, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Recall Score', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Classification Task', fontweight='bold', fontsize=12)
    ax2.set_title('Balanced Mode Recall\n(General Monitoring)', fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, rec in zip(bars2, balanced_recall):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                f'{rec:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # High-Recall Mode Visualization
    bars3 = ax3.bar(classes, high_recall_precision, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Precision Score', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Classification Task', fontweight='bold', fontsize=12)
    ax3.set_title('High-Recall Mode Precision\n(Safety-First Strategy)', fontweight='bold')
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, prec, thresh in zip(bars3, high_recall_precision, high_recall_thresholds):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                f'{prec:.2f}\n(τ={thresh:.2f})', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    bars4 = ax4.bar(classes, high_recall_recall, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Recall Score', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Classification Task', fontweight='bold', fontsize=12)
    ax4.set_title('High-Recall Mode Recall\n(~90% Detection Target)', fontweight='bold')
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, rec in zip(bars4, high_recall_recall):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                f'{rec:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # High-Precision Mode Visualization
    bars5 = ax5.bar(classes, high_precision_precision, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax5.set_ylabel('Precision Score', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Classification Task', fontweight='bold', fontsize=12)
    ax5.set_title('High-Precision Mode Precision\n(Quality-First Strategy)', fontweight='bold')
    ax5.set_ylim(0, 1.1)
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, prec, thresh in zip(bars5, high_precision_precision, high_precision_thresholds):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                f'{prec:.2f}\n(τ={thresh:.2f})', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Comparative Analysis
    strategies = ['Balanced\nMode', 'High-Recall\nMode', 'High-Precision\nMode']
    overall_performance = [0.75, 0.61, 0.72]  # F1-weighted average across classes
    strategy_colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars6 = ax6.bar(strategies, overall_performance, color=strategy_colors, alpha=0.8, 
                    edgecolor='black', linewidth=2)
    ax6.set_ylabel('Weighted F1 Score', fontweight='bold', fontsize=12)
    ax6.set_xlabel('Threshold Strategy', fontweight='bold', fontsize=12)
    ax6.set_title('Overall Strategy Performance\n(Weighted F1 Across All Classes)', fontweight='bold')
    ax6.set_ylim(0, 1.0)
    ax6.grid(True, alpha=0.3, axis='y')
    
    use_cases = ['General\nMonitoring', 'Safety\nAudits', 'Quality\nControl']
    for bar, perf, use_case in zip(bars6, overall_performance, use_cases):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{perf:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax6.text(bar.get_x() + bar.get_width()/2., 0.05,
                use_case, ha='center', va='center', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    plt.suptitle('Alternative Threshold Strategies Comparison\nAdaptive Approaches for Different Operational Requirements', 
                 fontsize=18, fontweight='bold')
    
    # Add strategy explanations
    fig.text(0.5, 0.02, 
             'Balanced Mode: Optimal for general road monitoring | High-Recall Mode: Minimizes missed cases for safety audits | High-Precision Mode: Reduces false alarms for automated processing',
             ha='center', va='bottom', fontsize=11, style='italic',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    save_plot('threshold_strategies_comparison.png')

# 4. Individual Model Performance Breakdown
def create_individual_model_breakdown():
    """Create detailed individual model performance breakdown"""
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
    
    classes = ['Damage', 'Occlusion', 'Crop']
    colors = ['#FF6B6B', '#FFB347', '#DDA0DD']
    
    # Model B Performance Data
    model_b_precision = [63.6, 80.1, 97.5]
    model_b_recall = [65.8, 80.5, 96.3]
    model_b_f1 = [64.7, 80.3, 96.9]
    
    # Model H Performance Data
    model_h_precision = [57.8, 81.7, 97.4]
    model_h_recall = [64.3, 73.8, 94.4]
    model_h_f1 = [60.9, 77.6, 95.9]
    
    # Model B Performance Charts
    bars1 = ax1.bar(classes, model_b_precision, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Precision (%)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Classification Task', fontweight='bold', fontsize=12)
    ax1.set_title('Model B - Precision Performance\n(Pure Feature Learning)', fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, prec in zip(bars1, model_b_precision):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                f'{prec:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    bars2 = ax2.bar(classes, model_b_recall, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Recall (%)', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Classification Task', fontweight='bold', fontsize=12)
    ax2.set_title('Model B - Recall Performance\n(Pure Feature Learning)', fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, rec in zip(bars2, model_b_recall):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                f'{rec:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    bars3 = ax3.bar(classes, model_b_f1, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_ylabel('F1 Score (%)', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Classification Task', fontweight='bold', fontsize=12)
    ax3.set_title('Model B - F1 Score Performance\n(Macro F1: 80.6%)', fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, f1 in zip(bars3, model_b_f1):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                f'{f1:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Model H Performance Charts
    bars4 = ax4.bar(classes, model_h_precision, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Precision (%)', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Classification Task', fontweight='bold', fontsize=12)
    ax4.set_title('Model H - Precision Performance\n(Enhanced Preprocessing)', fontweight='bold')
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, prec in zip(bars4, model_h_precision):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                f'{prec:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    bars5 = ax5.bar(classes, model_h_recall, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax5.set_ylabel('Recall (%)', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Classification Task', fontweight='bold', fontsize=12)
    ax5.set_title('Model H - Recall Performance\n(Enhanced Preprocessing)', fontweight='bold')
    ax5.set_ylim(0, 105)
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, rec in zip(bars5, model_h_recall):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                f'{rec:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    bars6 = ax6.bar(classes, model_h_f1, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax6.set_ylabel('F1 Score (%)', fontweight='bold', fontsize=12)
    ax6.set_xlabel('Classification Task', fontweight='bold', fontsize=12)
    ax6.set_title('Model H - F1 Score Performance\n(Macro F1: 78.1%)', fontweight='bold')
    ax6.set_ylim(0, 105)
    ax6.grid(True, alpha=0.3, axis='y')
    for bar, f1 in zip(bars6, model_h_f1):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                f'{f1:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.suptitle('Individual Model Performance Breakdown\nComplementary Strengths Leading to Ensemble Success', 
                 fontsize=18, fontweight='bold')
    
    # Add model characteristics
    fig.text(0.25, 0.02, 
             'Model B Characteristics:\n• Fast convergence (21 epochs)\n• Efficient training (1.26 hours)\n• Strong pure feature extraction',
             ha='center', va='bottom', fontsize=10, style='italic',
             bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.7))
    
    fig.text(0.75, 0.02, 
             'Model H Characteristics:\n• Longer convergence (37 epochs)\n• Enhanced preprocessing (2.99 hours)\n• CLAHE + road masking benefits',
             ha='center', va='bottom', fontsize=10, style='italic',
             bbox=dict(boxstyle="round,pad=0.4", facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    save_plot('individual_model_breakdown.png')

if __name__ == "__main__":
    print(f"Creating supplementary visualizations...")
    print(f"Output directory: {output_dir}")
    
    try:
        create_experimental_timeline()
        print("✓ Detailed experimental timeline created")
        
        create_ensemble_architecture_detailed()
        print("✓ Detailed ensemble architecture diagram created")
        
        create_threshold_strategies_comparison()
        print("✓ Threshold strategies comparison created")
        
        create_individual_model_breakdown()
        print("✓ Individual model performance breakdown created")
        
        print("\n" + "="*70)
        print("All supplementary visualizations completed successfully!")
        print("Additional charts created:")
        print("- Comprehensive experimental timeline with outcomes")
        print("- Detailed ensemble architecture with component flow")
        print("- Alternative threshold strategies for different use cases")
        print("- Individual model performance breakdown and analysis")
        print("="*70)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
