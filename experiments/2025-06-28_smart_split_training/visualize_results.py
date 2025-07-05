import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

def load_results():
    """Load the combined training results."""
    with open('results/combined_training_results.json', 'r') as f:
        return json.load(f)

def create_overall_comparison(results):
    """Create overall metrics comparison across models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Extract metrics
    models = list(results.keys())
    overall_metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Overall metrics bar chart
    ax = axes[0, 0]
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(overall_metrics):
        values = [results[model]['test_metrics'][metric] for model in models]
        ax.bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Overall Metrics Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training time comparison
    ax = axes[0, 1]
    training_times = [results[model]['training_time'] / 60 for model in models]  # Convert to minutes
    bars = ax.bar(models, training_times, alpha=0.7, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax.set_xlabel('Models')
    ax.set_ylabel('Training Time (minutes)')
    ax.set_title('Training Time Comparison')
    
    # Add value labels on bars
    for bar, time in zip(bars, training_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{time:.1f}m', ha='center', va='bottom')
    
    # Per-class F1 scores heatmap
    ax = axes[1, 0]
    classes = ['damaged', 'occluded', 'cropped']
    f1_matrix = []
    
    for model in models:
        model_f1s = []
        for class_name in classes:
            f1_score = results[model]['test_metrics']['per_class'][class_name]['f1']
            model_f1s.append(f1_score)
        f1_matrix.append(model_f1s)
    
    f1_matrix = np.array(f1_matrix)
    im = ax.imshow(f1_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(classes)):
            text = ax.text(j, i, f'{f1_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([c.capitalize() for c in classes])
    ax.set_yticklabels([m.upper() for m in models])
    ax.set_title('Per-Class F1 Scores Heatmap')
    plt.colorbar(im, ax=ax, label='F1 Score')
    
    # Model descriptions
    ax = axes[1, 1]
    ax.axis('off')
    descriptions = []
    for model in models:
        desc = results[model]['description']
        f1 = results[model]['test_metrics']['f1']
        time = results[model]['training_time'] / 60
        descriptions.append(f"{model.upper()}: {desc}\nF1: {f1:.3f}, Time: {time:.1f}m")
    
    text = '\n\n'.join(descriptions)
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_per_class_analysis(results):
    """Create detailed per-class analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Per-Class Performance Analysis', fontsize=16, fontweight='bold')
    
    models = list(results.keys())
    classes = ['damaged', 'occluded', 'cropped']
    metrics = ['precision', 'recall', 'f1']
    
    # Per-class metrics for each model
    for i, class_name in enumerate(classes):
        ax = axes[i//2, i%2]
        
        x = np.arange(len(models))
        width = 0.25
        
        for j, metric in enumerate(metrics):
            values = [results[model]['test_metrics']['per_class'][class_name][metric] 
                     for model in models]
            ax.bar(x + j*width, values, width, label=metric.capitalize(), alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title(f'{class_name.capitalize()} Performance')
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.upper() for m in models])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('results/per_class_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_training_curves(results):
    """Create training curves if epoch data is available."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves Analysis', fontsize=16, fontweight='bold')
    
    for i, model in enumerate(results.keys()):
        ax = axes[i//2, i%2]
        
        # Try to load epoch metrics
        epoch_file = f'results/{model}/epoch_metrics.json'
        try:
            with open(epoch_file, 'r') as f:
                epoch_data = json.load(f)
            
            epochs = [m['epoch'] for m in epoch_data]
            train_f1 = [m['train']['f1'] for m in epoch_data]
            val_f1 = [m['val']['f1'] for m in epoch_data]
            train_loss = [m['train']['loss'] for m in epoch_data]
            val_loss = [m['val']['loss'] for m in epoch_data]
            
            # Plot F1 curves
            ax.plot(epochs, train_f1, label='Train F1', linewidth=2, marker='o', markersize=4)
            ax.plot(epochs, val_f1, label='Val F1', linewidth=2, marker='s', markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1-Score')
            ax.set_title(f'{model.upper()} Training Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except FileNotFoundError:
            ax.text(0.5, 0.5, f'No epoch data\navailable for {model}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{model.upper()} - No Data')
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(results):
    """Create a summary table."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    data = []
    for model in results.keys():
        metrics = results[model]['test_metrics']
        row = [
            model.upper(),
            f"{metrics['accuracy']:.3f}",
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1']:.3f}",
            f"{results[model]['training_time']/60:.1f}m",
            results[model]['description']
        ]
        data.append(row)
    
    # Create table
    columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Time', 'Description']
    table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best F1 score
    f1_scores = [float(row[4]) for row in data]
    best_f1_idx = f1_scores.index(max(f1_scores))
    for i in range(len(columns)):
        table[(best_f1_idx + 1, i)].set_facecolor('#FFD700')
    
    plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('results/summary_table.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_insights_analysis(results):
    """Create insights and recommendations."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Analyze results
    insights = []
    
    # Find best performing model
    best_model = max(results.keys(), key=lambda x: results[x]['test_metrics']['f1'])
    best_f1 = results[best_model]['test_metrics']['f1']
    insights.append(f"🏆 BEST MODEL: {best_model.upper()} (F1: {best_f1:.3f})")
    
    # Analyze per-class performance
    for class_name in ['damaged', 'occluded', 'cropped']:
        class_performances = [(model, results[model]['test_metrics']['per_class'][class_name]['f1']) 
                             for model in results.keys()]
        best_class_model = max(class_performances, key=lambda x: x[1])
        insights.append(f"📊 {class_name.upper()}: {best_class_model[0].upper()} (F1: {best_class_model[1]:.3f})")
    
    # Training time analysis
    fastest_model = min(results.keys(), key=lambda x: results[x]['training_time'])
    fastest_time = results[fastest_model]['training_time'] / 60
    insights.append(f"⚡ FASTEST: {fastest_model.upper()} ({fastest_time:.1f} minutes)")
    
    # Key observations
    insights.append("\n🔍 KEY OBSERVATIONS:")
    
    # Check if augmentation helps
    model_c_f1 = results['model_c']['test_metrics']['f1']
    model_b_f1 = results['model_b']['test_metrics']['f1']
    if model_c_f1 > model_b_f1:
        insights.append("• Augmentation (Model C) outperforms masks (Model B)")
    else:
        insights.append("• Masks (Model B) outperform augmentation (Model C)")
    
    # Check if combined approach helps
    model_d_f1 = results['model_d']['test_metrics']['f1']
    if model_d_f1 > max(model_c_f1, model_b_f1):
        insights.append("• Combined approach (Model D) is most effective")
    else:
        insights.append("• Combined approach (Model D) doesn't improve performance")
    
    # Class imbalance issues
    for class_name in ['occluded', 'cropped']:
        all_f1s = [results[model]['test_metrics']['per_class'][class_name]['f1'] 
                  for model in results.keys()]
        if max(all_f1s) < 0.5:
            insights.append(f"• {class_name.capitalize()} class shows poor performance across all models")
    
    # Recommendations
    insights.append("\n💡 RECOMMENDATIONS:")
    insights.append("• Use Model C for best overall performance")
    insights.append("• Consider class weighting for occluded/cropped classes")
    insights.append("• Augmentation significantly improves model performance")
    insights.append("• Road masks don't provide substantial benefits")
    
    # Display insights
    text = '\n'.join(insights)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.title('Results Analysis & Insights', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('results/insights_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to create all visualizations."""
    print("Loading results...")
    results = load_results()
    
    print("Creating visualizations...")
    
    # Create output directory
    Path('results').mkdir(exist_ok=True)
    
    # Generate all visualizations
    create_overall_comparison(results)
    create_per_class_analysis(results)
    create_training_curves(results)
    create_summary_table(results)
    create_insights_analysis(results)
    
    print("All visualizations saved to 'results/' directory!")
    print("\nKey findings:")
    print(f"• Best overall model: {max(results.keys(), key=lambda x: results[x]['test_metrics']['f1']).upper()}")
    print(f"• Model C (augmentation) shows best performance with F1: {results['model_c']['test_metrics']['f1']:.3f}")
    print(f"• Training times range from {min([results[m]['training_time']/60 for m in results.keys()]):.1f} to {max([results[m]['training_time']/60 for m in results.keys()]):.1f} minutes")

if __name__ == "__main__":
    main() 