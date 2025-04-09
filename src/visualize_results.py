import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from data_loader import RoadDistressDataset
from model import RoadDistressModel

def load_training_history(checkpoint_dir):
    """
    Load the training history from the checkpoint directory
    
    Args:
        checkpoint_dir (str): Directory containing the checkpoints
    
    Returns:
        dict: Training history
    """
    # Load the best model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    # Extract training history
    history = {
        'epoch': checkpoint['epoch'],
        'train_loss': checkpoint['train_loss'],
        'train_acc': checkpoint['train_acc'],
        'val_loss': checkpoint['val_loss'],
        'val_acc': checkpoint['val_acc']
    }
    
    return history, checkpoint

def plot_training_curves(history, save_dir):
    """
    Plot training and validation curves
    
    Args:
        history (dict): Training history
        save_dir (str): Directory to save the plots
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot learning rate curve
    if 'lr' in history:
        axes[1, 0].plot(history['lr'], label='Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot model architecture
    axes[1, 1].text(0.1, 0.5, 
                   f"Model: ResNet50\n"
                   f"Best Epoch: {history['epoch']}\n"
                   f"Best Train Acc: {history['train_acc']:.2f}%\n"
                   f"Best Val Acc: {history['val_acc']:.2f}%",
                   fontsize=12)
    axes[1, 1].set_title('Model Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on the given data loader
    
    Args:
        model (nn.Module): The model to evaluate
        data_loader (DataLoader): Data loader for evaluation
        device (torch.device): Device to evaluate on
    
    Returns:
        tuple: (predictions, labels, probabilities)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Plot confusion matrix
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Damaged', 'Damaged'],
                yticklabels=['Not Damaged', 'Damaged'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path):
    """
    Plot ROC curve
    
    Args:
        y_true (np.ndarray): True labels
        y_prob (np.ndarray): Predicted probabilities
        save_path (str): Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, save_path):
    """
    Plot precision-recall curve
    
    Args:
        y_true (np.ndarray): True labels
        y_prob (np.ndarray): Predicted probabilities
        save_path (str): Path to save the plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    # Configuration
    data_dir = 'organized_dataset'
    checkpoint_dir = 'checkpoints'
    output_dir = 'visualization_results'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training history
    try:
        history, checkpoint = load_training_history(checkpoint_dir)
        print(f"Loaded training history from epoch {history['epoch']}")
    except Exception as e:
        print(f"Error loading training history: {e}")
        return
    
    # Plot training curves
    plot_training_curves(history, output_dir)
    print(f"Training curves saved to {os.path.join(output_dir, 'training_curves.png')}")
    
    # Load model and data
    model = RoadDistressModel(num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create data loaders
    val_dataset = RoadDistressDataset(data_dir, split='val', transform_type='val')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Evaluate model
    y_pred, y_true, y_prob = evaluate_model(model, val_loader, device)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, os.path.join(output_dir, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")
    
    # Plot ROC curve
    plot_roc_curve(y_true, y_prob, os.path.join(output_dir, 'roc_curve.png'))
    print(f"ROC curve saved to {os.path.join(output_dir, 'roc_curve.png')}")
    
    # Plot precision-recall curve
    plot_precision_recall_curve(y_true, y_prob, os.path.join(output_dir, 'precision_recall_curve.png'))
    print(f"Precision-recall curve saved to {os.path.join(output_dir, 'precision_recall_curve.png')}")
    
    # Create a summary report
    with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write("Road Distress Classification Model Evaluation\n")
        f.write("===========================================\n\n")
        f.write(f"Model: ResNet50\n")
        f.write(f"Best Epoch: {history['epoch']}\n")
        f.write(f"Best Train Accuracy: {history['train_acc']:.2f}%\n")
        f.write(f"Best Validation Accuracy: {history['val_acc']:.2f}%\n\n")
        
        # Calculate metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        f.write("Validation Metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
    
    print(f"Evaluation summary saved to {os.path.join(output_dir, 'evaluation_summary.txt')}")
    print("\nVisualization complete!")

if __name__ == '__main__':
    main() 