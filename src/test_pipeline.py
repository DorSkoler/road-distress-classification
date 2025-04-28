import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

from data_loader import RoadDistressDataset
from model import RoadDistressModel

def load_model_and_data(model_path, device):
    """Load the trained model and prepare test data"""
    # Load model configuration
    config_path = os.path.join(os.path.dirname(model_path), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model
    model = RoadDistressModel(**config['model_config']).to(device)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Create test dataset and dataloader
    test_dataset = RoadDistressDataset(
        data_dir='organized_dataset',
        split='test',
        transform_type='val'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return model, test_loader

def evaluate_model(model, test_loader, device):
    """Evaluate model performance and generate visualizations"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Get predictions
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating model"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics for each task
    task_names = ['Damage', 'Occlusion', 'Crop']
    metrics = {}
    
    for i, task in enumerate(task_names):
        # Calculate basic metrics
        accuracy = np.mean(all_labels[:, i] == all_preds[:, i]) * 100
        precision = np.sum((all_preds[:, i] == 1) & (all_labels[:, i] == 1)) / np.sum(all_preds[:, i] == 1) * 100
        recall = np.sum((all_preds[:, i] == 1) & (all_labels[:, i] == 1)) / np.sum(all_labels[:, i] == 1) * 100
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate ROC and PR curves
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        pr_auc = auc(recall_curve, precision_curve)
        
        metrics[task] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
    
    # Generate confusion matrices
    plt.figure(figsize=(15, 5))
    for i, task in enumerate(task_names):
        plt.subplot(1, 3, i+1)
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {task}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('visualization_results/confusion_matrices.png')
    plt.close()
    
    # Generate ROC curves
    plt.figure(figsize=(10, 8))
    for i, task in enumerate(task_names):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        plt.plot(fpr, tpr, label=f'{task} (AUC = {metrics[task]["roc_auc"]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('visualization_results/roc_curves.png')
    plt.close()
    
    # Generate Precision-Recall curves
    plt.figure(figsize=(10, 8))
    for i, task in enumerate(task_names):
        precision, recall, _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        plt.plot(recall, precision, label=f'{task} (AUC = {metrics[task]["pr_auc"]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.savefig('visualization_results/precision_recall_curves.png')
    plt.close()
    
    # Print detailed metrics
    print("\nDetailed Metrics:")
    print("=" * 50)
    for task, task_metrics in metrics.items():
        print(f"\n{task}:")
        print(f"Accuracy: {task_metrics['accuracy']:.2f}%")
        print(f"Precision: {task_metrics['precision']:.2f}%")
        print(f"Recall: {task_metrics['recall']:.2f}%")
        print(f"F1 Score: {task_metrics['f1']:.2f}%")
        print(f"ROC AUC: {task_metrics['roc_auc']:.3f}")
        print(f"PR AUC: {task_metrics['pr_auc']:.3f}")
    
    # Save metrics
    with open('visualization_results/test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def main():
    # Create visualization results directory
    os.makedirs('visualization_results', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and data
    model_path = 'experiments/efficientnet_b3_enhanced_20250427_230032/best_model.pth'
    model, test_loader = load_model_and_data(model_path, device)
    
    # Evaluate model and generate visualizations
    metrics = evaluate_model(model, test_loader, device)
    
    print("\nEvaluation completed! Results saved in visualization_results/")

if __name__ == '__main__':
    main() 