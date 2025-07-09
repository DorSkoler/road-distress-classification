#!/usr/bin/env python3
"""
Analyze improvements from class balance and regularization fixes.

This script compares the original vs improved training results to validate
that the fixes for class imbalance and overfitting are working.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(results_dir):
    """Load training results from directory."""
    results = {}
    
    for model_dir in results_dir.glob("model_*"):
        if model_dir.is_dir():
            model_name = model_dir.name
            summary_file = model_dir / "training_summary.json"
            
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    results[model_name] = data
    
    return results

def analyze_class_balance_improvement(original_results, improved_results):
    """Analyze if class balance improvements are working."""
    print("🔍 CLASS BALANCE ANALYSIS")
    print("=" * 50)
    
    for model in ['model_a', 'model_b', 'model_c', 'model_d']:
        if model in original_results and model in improved_results:
            orig = original_results[model].get('best_metrics', {})
            impr = improved_results[model].get('best_metrics', {})
            
            print(f"\n{model.upper()}:")
            print("                 Original  →  Improved")
            
            # Per-class F1 scores
            for class_name in ['damage', 'occlusion', 'crop']:
                orig_f1 = orig.get(f'{class_name}_f1', 0)
                impr_f1 = impr.get(f'{class_name}_f1', 0)
                change = impr_f1 - orig_f1
                arrow = "↗️" if change > 0 else "↘️" if change < 0 else "→"
                print(f"  {class_name:10} F1:  {orig_f1:.3f}  →  {impr_f1:.3f} {arrow} ({change:+.3f})")
            
            # Overall metrics
            orig_macro = orig.get('f1_macro', orig.get('weighted_f1', 0))
            impr_macro = impr.get('f1_macro', impr.get('weighted_f1', 0))
            change = impr_macro - orig_macro
            arrow = "↗️" if change > 0 else "↘️" if change < 0 else "→"
            print(f"  {'Macro F1':10}:  {orig_macro:.3f}  →  {impr_macro:.3f} {arrow} ({change:+.3f})")

def analyze_overfitting_reduction(original_results, improved_results):
    """Analyze if overfitting reduction is working."""
    print("\n\n🔍 OVERFITTING ANALYSIS")
    print("=" * 50)
    
    for model in ['model_a', 'model_b', 'model_c', 'model_d']:
        if model in original_results and model in improved_results:
            orig = original_results[model].get('training_results', {})
            impr = improved_results[model].get('training_results', {})
            
            print(f"\n{model.upper()}:")
            
            # Training vs validation loss gap (proxy for overfitting)
            orig_train_loss = orig.get('final_train_loss', 0)
            orig_val_loss = orig.get('final_val_loss', 0)
            orig_gap = abs(orig_val_loss - orig_train_loss)
            
            impr_train_loss = impr.get('final_train_loss', 0)
            impr_val_loss = impr.get('final_val_loss', 0)
            impr_gap = abs(impr_val_loss - impr_train_loss)
            
            gap_change = impr_gap - orig_gap
            arrow = "↗️" if gap_change > 0 else "↘️" if gap_change < 0 else "→"
            
            print(f"  Train/Val Gap:  {orig_gap:.3f}  →  {impr_gap:.3f} {arrow}")
            
            # Early stopping epoch (earlier = less overfitting)
            orig_epoch = orig.get('best_epoch', 0) + 1
            impr_epoch = impr.get('best_epoch', 0) + 1
            epoch_change = impr_epoch - orig_epoch
            arrow = "↗️" if epoch_change > 0 else "↘️" if epoch_change < 0 else "→"
            
            print(f"  Best Epoch:     {orig_epoch:3d}      →  {impr_epoch:3d}      {arrow}")

def create_comparison_plots(original_results, improved_results):
    """Create comparison plots."""
    print("\n\n📊 Creating comparison plots...")
    
    # Prepare data for plotting
    models = ['model_a', 'model_b', 'model_c', 'model_d']
    
    # F1 scores comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Per-class F1 comparison
    class_names = ['damage', 'occlusion', 'crop']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, class_name in enumerate(class_names):
        ax = axes[0, i] if i < 2 else axes[1, 0]
        
        orig_scores = []
        impr_scores = []
        model_labels = []
        
        for model in models:
            if model in original_results and model in improved_results:
                orig = original_results[model].get('best_metrics', {})
                impr = improved_results[model].get('best_metrics', {})
                
                orig_scores.append(orig.get(f'{class_name}_f1', 0))
                impr_scores.append(impr.get(f'{class_name}_f1', 0))
                model_labels.append(model.replace('model_', 'M'))
        
        x = np.arange(len(model_labels))
        width = 0.35
        
        ax.bar(x - width/2, orig_scores, width, label='Original', alpha=0.7, color='lightcoral')
        ax.bar(x + width/2, impr_scores, width, label='Improved', alpha=0.7, color=colors[i])
        
        ax.set_title(f'{class_name.capitalize()} F1 Score')
        ax.set_xlabel('Model')
        ax.set_ylabel('F1 Score')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    # Macro F1 comparison
    ax = axes[1, 1]
    orig_macro = []
    impr_macro = []
    model_labels = []
    
    for model in models:
        if model in original_results and model in improved_results:
            orig = original_results[model].get('best_metrics', {})
            impr = improved_results[model].get('best_metrics', {})
            
            orig_macro.append(orig.get('f1_macro', orig.get('weighted_f1', 0)))
            impr_macro.append(impr.get('f1_macro', impr.get('weighted_f1', 0)))
            model_labels.append(model.replace('model_', 'M'))
    
    x = np.arange(len(model_labels))
    width = 0.35
    
    ax.bar(x - width/2, orig_macro, width, label='Original', alpha=0.7, color='lightcoral')
    ax.bar(x + width/2, impr_macro, width, label='Improved', alpha=0.7, color='gold')
    
    ax.set_title('Macro F1 Score')
    ax.set_xlabel('Model')
    ax.set_ylabel('Macro F1 Score')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('results_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved results_comparison.png")

def main():
    """Main analysis function."""
    print("🔬 ANALYZING TRAINING IMPROVEMENTS")
    print("=" * 70)
    
    # Load results
    original_dir = Path("results")
    improved_dir = Path("results_improved")  # Where improved results would be saved
    
    # For now, simulate this since we just created the configs
    print("📁 Looking for results directories:")
    print(f"  Original:  {original_dir} {'✓' if original_dir.exists() else '❌'}")
    print(f"  Improved:  {improved_dir} {'✓' if improved_dir.exists() else '❌'}")
    
    if original_dir.exists():
        original_results = load_results(original_dir)
        print(f"\n📊 Loaded {len(original_results)} original models")
        
        # Show what we expect to see with improvements
        print("\n🎯 EXPECTED IMPROVEMENTS WITH NEW CONFIGURATION:")
        print("=" * 60)
        
        print("\n🔧 Applied fixes:")
        print("  ✓ Class weights [1.0, 2.2, 9.3] for damage/occlusion/crop")
        print("  ✓ Increased dropout: 0.5 → 0.7")
        print("  ✓ Increased weight decay: 0.02 → 0.05")
        print("  ✓ Added label smoothing: 0.1")
        print("  ✓ Reduced early stopping: 10 → 5 epochs")
        print("  ✓ Switched to macro F1 evaluation")
        print("  ✓ Reduced augmentation samples: 3 → 2")
        
        print("\n📈 Expected outcomes:")
        print("  • CROP class F1: Should increase significantly (was severely underrepresented)")
        print("  • OCCLUSION class F1: Should increase moderately")
        print("  • DAMAGE class F1: May decrease slightly (less overfit to majority class)")
        print("  • Macro F1: Should be lower but more honest (no class size bias)")
        print("  • Training curves: Should be more stable, less overfitting")
        print("  • Best epoch: Should occur earlier (better regularization)")
        
        print("\n🚀 TO RUN IMPROVED TRAINING:")
        print("  python run_improved_training.py")
        
        if improved_dir.exists():
            improved_results = load_results(improved_dir)
            print(f"📊 Loaded {len(improved_results)} improved models")
            
            # Run actual analysis
            analyze_class_balance_improvement(original_results, improved_results)
            analyze_overfitting_reduction(original_results, improved_results)
            create_comparison_plots(original_results, improved_results)
        
    else:
        print("\n❌ No original results found. Run training first:")
        print("  python -m src.training.trainer --variant model_a")

if __name__ == "__main__":
    main() 