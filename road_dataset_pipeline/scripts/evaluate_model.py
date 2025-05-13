import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
from datetime import datetime
import pandas as pd

class RoadDatasetWithMasks:
    def __init__(self, image_dir, mask_dir, json_dir, split='test'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.json_dir = json_dir
        self.split = split
        
        # Get all image-mask pairs for the specified split
        self.samples = []
        img_split_dir = os.path.join(image_dir, split)
        mask_split_dir = os.path.join(mask_dir, split)
        
        if not os.path.exists(img_split_dir):
            print(f"Warning: {split} directory not found at {img_split_dir}")
            return
            
        for img_name in os.listdir(img_split_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                base_name = os.path.splitext(img_name)[0]
                mask_name = f"{base_name}_mask.png"
                json_name = f"{base_name}.json"
                
                if os.path.exists(os.path.join(mask_split_dir, mask_name)):
                    # Load JSON data
                    json_path = os.path.join(json_dir, json_name)
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            img_data = json.load(f)
                        
                        # Initialize labels
                        labels = torch.zeros(3)  # [Damage, Occlusion, Crop]
                        
                        # Extract labels from tags
                        for tag in img_data.get('tags', []):
                            if tag['name'] == 'Damage':
                                labels[0] = 1 if tag['value'] == 'Damaged' else 0
                            elif tag['name'] == 'Occlusion':
                                labels[1] = 1 if tag['value'] == 'Occluded' else 0
                            elif tag['name'] == 'Crop':
                                labels[2] = 1 if tag['value'] == 'Cropped' else 0
                        
                        self.samples.append({
                            'image': os.path.join(img_split_dir, img_name),
                            'mask': os.path.join(mask_split_dir, mask_name),
                            'labels': labels
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image and mask
        image = cv2.imread(sample['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        
        # Resize to model input size
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        return image, mask, sample['labels']

class RoadDistressModelWithMasks(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # EfficientNet-B3 backbone
        self.backbone = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=num_classes,
            activation=None
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, mask):
        # Get features from backbone
        features = self.backbone(x)  # [B, num_classes, H, W]
        
        # Apply mask to focus on road pixels
        mask = mask.expand_as(features)  # [B, num_classes, H, W]
        masked_features = features * mask
        
        # Get classification predictions
        return self.classifier(masked_features)

def evaluate_model(model_path, test_loader, device, save_dir):
    # Load model
    model = RoadDistressModelWithMasks(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Initialize metrics
    all_preds = []
    all_labels = []
    
    # Evaluation loop
    with torch.no_grad():
        for images, masks, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            outputs = model(images, masks)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            # Store predictions and ground truth
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate per-class metrics
    class_names = ['Damage', 'Occlusion', 'Crop']
    metrics = {
        'overall_accuracy': float(np.mean(all_preds == all_labels)),
        'per_class_metrics': {}
    }
    
    for i, name in enumerate(class_names):
        metrics['per_class_metrics'][name] = {
            'accuracy': float(np.mean(all_preds[:, i] == all_labels[:, i])),
            'precision': float(np.sum((all_preds[:, i] == 1) & (all_labels[:, i] == 1)) / (np.sum(all_preds[:, i] == 1) + 1e-10)),
            'recall': float(np.sum((all_preds[:, i] == 1) & (all_labels[:, i] == 1)) / (np.sum(all_labels[:, i] == 1) + 1e-10))
        }
    
    # Save metrics
    with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def create_evaluation_visualizations(metrics, save_dir):
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Per-class metrics bar plot
    plt.figure(figsize=(12, 6))
    class_names = list(metrics['per_class_metrics'].keys())
    metrics_names = ['accuracy', 'precision', 'recall']
    
    x = np.arange(len(class_names))
    width = 0.25
    
    for i, metric in enumerate(metrics_names):
        values = [metrics['per_class_metrics'][name][metric] for name in class_names]
        plt.bar(x + i*width, values, width, label=metric.capitalize())
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Per-class Evaluation Metrics')
    plt.xticks(x + width, class_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'per_class_metrics.png'))
    plt.close()
    
    # 2. Overall accuracy
    plt.figure(figsize=(8, 6))
    plt.bar(['Overall Accuracy'], [metrics['overall_accuracy']])
    plt.ylim(0, 1)
    plt.title('Overall Model Accuracy')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'overall_accuracy.png'))
    plt.close()

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'experiments/efficientnet_b3_with_masks_20250510_220123/best_model.pth'  # Update with your model path
    save_dir = 'evaluation_results'
    os.makedirs(save_dir, exist_ok=True)
    
    # Create test dataset and dataloader
    test_dataset = RoadDatasetWithMasks(
        image_dir='../filtered',
        mask_dir='../filtered_masks',
        json_dir='../tagged_json',
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Evaluating model on {len(test_dataset)} test samples...")
    
    # Evaluate model
    metrics = evaluate_model(model_path, test_loader, device, save_dir)
    
    # Create visualizations
    create_evaluation_visualizations(metrics, save_dir)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print("\nPer-class Metrics:")
    for class_name, class_metrics in metrics['per_class_metrics'].items():
        print(f"\n{class_name}:")
        print(f"  Accuracy: {class_metrics['accuracy']:.4f}")
        print(f"  Precision: {class_metrics['precision']:.4f}")
        print(f"  Recall: {class_metrics['recall']:.4f}")
    
    print(f"\nResults saved to: {save_dir}")

if __name__ == '__main__':
    main() 