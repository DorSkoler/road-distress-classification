import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_metrics(experiment_dir):
    """Load metrics from a model's experiment directory"""
    metrics_path = os.path.join(experiment_dir, 'metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def load_task_accuracies(experiment_dir):
    """Load per-task accuracies from a model's experiment directory"""
    accuracies_path = os.path.join(experiment_dir, 'best_task_accuracies.json')
    with open(accuracies_path, 'r') as f:
        accuracies = json.load(f)
    return accuracies

def plot_training_curves(metrics_dict, save_path):
    """Plot training and validation curves for all models"""
    plt.figure(figsize=(15, 10))
    
    # Plot loss curves
    plt.subplot(2, 2, 1)
    for model_name, metrics in metrics_dict.items():
        plt.plot(metrics['train_loss'], label=f'{model_name} Train')
        plt.plot(metrics['val_loss'], label=f'{model_name} Val')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy curves
    plt.subplot(2, 2, 2)
    for model_name, metrics in metrics_dict.items():
        plt.plot(metrics['train_acc'], label=f'{model_name} Train')
        plt.plot(metrics['val_acc'], label=f'{model_name} Val')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate schedules
    plt.subplot(2, 2, 3)
    for model_name, metrics in metrics_dict.items():
        plt.plot(metrics['learning_rate'], label=model_name)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_task_accuracies(accuracies_dict, save_path):
    """Plot per-task accuracies for all models"""
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting
    models = list(accuracies_dict.keys())
    tasks = ['damage_acc', 'occlusion_acc', 'crop_acc']
    task_names = ['Damage', 'Occlusion', 'Crop']
    
    x = range(len(models))
    width = 0.25
    
    for i, (task, task_name) in enumerate(zip(tasks, task_names)):
        values = [accuracies_dict[model][task] for model in models]
        plt.bar([xi + i*width for xi in x], values, width, label=task_name)
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Task Validation Accuracies')
    plt.xticks([xi + width for xi in x], models)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Create visualization directory
    vis_dir = 'visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load metrics from both models
    metrics_dict = {}
    accuracies_dict = {}
    
    for model_name in ['efficientnet_b3_enhanced_20250427_210718']:
        experiment_dir = os.path.join('experiments', model_name)
        if os.path.exists(experiment_dir):
            metrics_dict[model_name] = load_metrics(experiment_dir)
            accuracies_dict[model_name] = load_task_accuracies(experiment_dir)
    
    # Plot training curves
    plot_training_curves(metrics_dict, os.path.join(vis_dir, 'training_curves.png'))
    
    # Plot task accuracies
    plot_task_accuracies(accuracies_dict, os.path.join(vis_dir, 'task_accuracies.png'))
    
    # Load and display final results
    results_path = 'experiments/training_results.csv'
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        print("\nFinal Model Comparison:")
        print(results_df.to_string(index=False))
    
    print(f"\nVisualizations saved to {vis_dir}/")

if __name__ == '__main__':
    main() 