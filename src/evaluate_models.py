import os
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from model import RoadDistressModel
from data_loader import RoadDistressDataset

def evaluate_model(model, test_loader, device, model_name):
    """Evaluate a single model on the test dataset"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'Evaluating {model_name}'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics for each task
    task_names = ['damage', 'occlusion', 'crop']
    metrics = {}
    
    for i, task in enumerate(task_names):
        y_true = all_labels[:, i]
        y_pred = all_predictions[:, i]
        
        # Calculate accuracy
        accuracy = np.mean(y_true == y_pred) * 100
        
        # Get classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        metrics[task] = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'] * 100,
            'recall': report['weighted avg']['recall'] * 100,
            'f1': report['weighted avg']['f1-score'] * 100
        }
    
    # Calculate overall accuracy
    overall_accuracy = np.mean(all_predictions == all_labels) * 100
    
    return metrics, overall_accuracy

def plot_confusion_matrices(predictions, labels, model_name, save_dir):
    """Plot confusion matrices for each task"""
    task_names = ['damage', 'occlusion', 'crop']
    
    plt.figure(figsize=(15, 5))
    for i, task in enumerate(task_names):
        plt.subplot(1, 3, i+1)
        cm = confusion_matrix(labels[:, i], predictions[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{task.capitalize()} - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrices.png'))
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataset and loader
    test_dataset = RoadDistressDataset(
        data_dir='organized_dataset',
        split='test',
        transform_type='val'  # Use validation transforms for testing
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Create visualization directory
    vis_dir = 'visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    
    # Evaluate both models
    results = {}
    model_configs = [
        ('efficientnet_b3', 'efficientnet_b3'),
        ('resnet50', 'resnet50')
    ]
    
    for model_name, backbone_type in model_configs:
        # Load model
        model = RoadDistressModel(
            num_classes=3,
            pretrained=False,  # We'll load our trained weights
            backbone_type=backbone_type
        ).to(device)
        
        # Load trained weights
        model_path = os.path.join('experiments', model_name, 'best_model.pth')
        model.load_state_dict(torch.load(model_path))
        
        # Evaluate
        metrics, overall_acc = evaluate_model(model, test_loader, device, model_name)
        results[model_name] = {
            'metrics': metrics,
            'overall_accuracy': overall_acc
        }
        
        # Plot confusion matrices
        plot_confusion_matrices(
            np.vstack([(torch.sigmoid(model(images.to(device))) > 0.5).float().cpu().numpy() 
                      for images, _ in test_loader]),
            np.vstack([labels.numpy() for _, labels in test_loader]),
            model_name,
            vis_dir
        )
    
    # Save results
    results_path = 'experiments/test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print results
    print("\nTest Results:")
    print("=" * 50)
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"Overall Accuracy: {result['overall_accuracy']:.2f}%")
        print("\nPer-task metrics:")
        for task, metrics in result['metrics'].items():
            print(f"\n{task.capitalize()}:")
            print(f"  Accuracy: {metrics['accuracy']:.2f}%")
            print(f"  Precision: {metrics['precision']:.2f}%")
            print(f"  Recall: {metrics['recall']:.2f}%")
            print(f"  F1-score: {metrics['f1']:.2f}%")
    
    print(f"\nResults saved to {results_path}")
    print(f"Confusion matrices saved to {vis_dir}/")

if __name__ == '__main__':
    main() 