import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import torch
from torch.utils.data import DataLoader
from model import RoadDistressModel
from data_loader import RoadDistressDataset
import json

def load_training_history(history_path):
    """Load training history from JSON file."""
    with open(history_path, 'r') as f:
        return json.load(f)

def plot_training_curves(history, save_dir='plots'):
    """Plot training and validation curves."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()

    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_curves.png'))
    plt.close()

    # Plot learning rate if available
    if 'lr' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['lr'], label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'lr_schedule.png'))
        plt.close()

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set and return predictions and metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, save_dir='plots'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_prob, save_dir='plots'):
    """Plot and save ROC curve."""
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
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, save_dir='plots'):
    """Plot and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'))
    plt.close()

def main():
    # Load model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RoadDistressModel()
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)
    
    # Load test dataset
    test_dataset = RoadDistressDataset('test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load training history
    history = load_training_history('training_history.json')
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Generate all plots
    plot_training_curves(history)
    
    # Evaluate model
    y_pred, y_true, y_prob = evaluate_model(model, test_loader, device)
    
    # Plot evaluation metrics
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_prob)
    plot_precision_recall_curve(y_true, y_prob)
    
    # Print evaluation summary
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    print("\nModel Evaluation Summary:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")

if __name__ == '__main__':
    main() 