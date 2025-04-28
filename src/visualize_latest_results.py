import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_metrics(experiment_dir):
    """Load metrics from the experiment directory"""
    metrics_path = os.path.join(experiment_dir, 'metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def plot_training_progress(metrics, save_dir):
    """Plot training progress including loss, accuracy, and learning rate"""
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot loss curves
    axes[0].plot(metrics['train_loss'], label='Training Loss', color='blue')
    axes[0].plot(metrics['val_loss'], label='Validation Loss', color='red')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy curves
    axes[1].plot(metrics['train_acc'], label='Training Accuracy', color='blue')
    axes[1].plot(metrics['val_acc'], label='Validation Accuracy', color='red')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot learning rate curve
    axes[2].plot(metrics['lr'], color='green')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].grid(True)
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_progress.png'), dpi=300)
    plt.close()

def plot_epoch_comparison(metrics, save_dir):
    """Plot comparison of metrics at different epochs"""
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot loss comparison
    axes[0, 0].plot(epochs, metrics['train_loss'], label='Training Loss', color='blue')
    axes[0, 0].plot(epochs, metrics['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Loss Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot accuracy comparison
    axes[0, 1].plot(epochs, metrics['train_acc'], label='Training Accuracy', color='blue')
    axes[0, 1].plot(epochs, metrics['val_acc'], label='Validation Accuracy', color='red')
    axes[0, 1].set_title('Accuracy Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot learning rate schedule
    axes[1, 0].plot(epochs, metrics['lr'], color='green')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale('log')
    
    # Plot best epoch markers
    best_val_acc_epoch = np.argmax(metrics['val_acc']) + 1
    best_val_loss_epoch = np.argmin(metrics['val_loss']) + 1
    
    axes[0, 1].axvline(x=best_val_acc_epoch, color='green', linestyle='--', 
                      label=f'Best Val Acc: {metrics["val_acc"][best_val_acc_epoch-1]:.2f}%')
    axes[0, 0].axvline(x=best_val_loss_epoch, color='purple', linestyle='--',
                      label=f'Best Val Loss: {metrics["val_loss"][best_val_loss_epoch-1]:.4f}')
    
    # Add model summary
    axes[1, 1].text(0.1, 0.5, 
                   f"Model: EfficientNet-B3 Enhanced\n"
                   f"Best Validation Accuracy: {max(metrics['val_acc']):.2f}%\n"
                   f"Best Validation Loss: {min(metrics['val_loss']):.4f}\n"
                   f"Final Training Accuracy: {metrics['train_acc'][-1]:.2f}%\n"
                   f"Final Validation Accuracy: {metrics['val_acc'][-1]:.2f}%\n"
                   f"Total Epochs: {len(metrics['train_loss'])}",
                   fontsize=12)
    axes[1, 1].set_title('Model Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'epoch_comparison.png'), dpi=300)
    plt.close()

def main():
    # Configuration
    experiment_dir = 'experiments/efficientnet_b3_enhanced_20250427_210718'
    output_dir = 'visualization_results'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load metrics
        metrics = load_metrics(experiment_dir)
        print("Successfully loaded metrics from experiment directory")
        
        # Generate plots
        plot_training_progress(metrics, output_dir)
        print("Generated training progress plot")
        
        plot_epoch_comparison(metrics, output_dir)
        print("Generated epoch comparison plot")
        
        # Create summary report
        with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
            f.write("EfficientNet-B3 Enhanced Training Summary\n")
            f.write("=======================================\n\n")
            f.write(f"Best Validation Accuracy: {max(metrics['val_acc']):.2f}%\n")
            f.write(f"Best Validation Loss: {min(metrics['val_loss']):.4f}\n")
            f.write(f"Final Training Accuracy: {metrics['train_acc'][-1]:.2f}%\n")
            f.write(f"Final Validation Accuracy: {metrics['val_acc'][-1]:.2f}%\n")
            f.write(f"Total Epochs: {len(metrics['train_loss'])}\n")
            f.write(f"Final Learning Rate: {metrics['lr'][-1]:.6f}\n")
        
        print(f"\nVisualizations and summary saved to {output_dir}/")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main() 