import torch
import json
import yaml
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dual_input_model import DualInputRoadDistressClassifier, RoadDistressDataset, create_model_variant
from torch.utils.data import DataLoader


def load_best_checkpoint(checkpoint_path, config, variant_name):
    """Load the best model checkpoint."""
    # Use create_model_variant to ensure correct config for single/dual input
    model = create_model_variant(config, variant_name)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


def setup_test_data(config, variant_name):
    """Setup test data loader."""
    import json as pyjson
    # Load split information
    splits_dir = 'splits'
    with open(os.path.join(splits_dir, 'test_images.txt'), 'r') as f:
        test_images = [line.strip() for line in f.readlines()]

    # Construct full paths to actual image files and annotation files
    coryell_path = config.get('dataset', {}).get('coryell_path', '../../data/coryell')
    
    def construct_image_paths_and_ann_paths(image_list):
        img_paths = []
        ann_paths = []
        for img_path in image_list:
            road_name = img_path.split('/')[0]
            img_name = img_path.split('/')[1]
            img_full = os.path.join(coryell_path, road_name, 'img', f"{img_name}.png")
            ann_full = os.path.join(coryell_path, road_name, 'ann', f"{img_name}.json")
            img_paths.append(img_full)
            ann_paths.append(ann_full)
        return img_paths, ann_paths

    test_images, test_anns = construct_image_paths_and_ann_paths(test_images)

    # Load labels from annotation JSONs
    label_map = {'damaged': 0, 'occlusion': 1, 'cropped': 2}
    test_labels = []
    for ann_path in test_anns:
        with open(ann_path, 'r') as f:
            ann = pyjson.load(f)
        label_str = ann.get('label', 'damaged')
        test_labels.append(label_map.get(label_str, 0))

    # Setup mask paths if using masks
    test_masks = None
    variant_config = config.get('comparative_training', {}).get('variants', {}).get(variant_name, {})
    use_masks = variant_config.get('use_masks', False)

    if use_masks:
        masks_dir = 'masks'
        def construct_mask_paths(image_list, split_name):
            mask_paths = []
            for img_path in image_list:
                parts = img_path.split(os.sep)
                road_name = parts[-3]
                img_name = parts[-1].replace('.png', '')
                mask_path = os.path.join(masks_dir, split_name, road_name, f"{img_name}.png")
                mask_paths.append(mask_path)
            return mask_paths
        test_masks = construct_mask_paths(test_images, 'test')

    # Create dataset
    test_dataset = RoadDistressDataset(
        image_paths=test_images,
        labels=test_labels,
        mask_paths=test_masks,
        use_masks=use_masks
    )

    # Create data loader
    batch_size = config.get('training', {}).get('batch_size', 16)
    num_workers = config.get('system', {}).get('num_workers', 4)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader, test_labels


def generate_test_predictions(model, test_loader, device, use_masks=False):
    """Generate predictions on test set."""
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            if use_masks:
                images, masks, _ = batch
                images, masks = images.to(device), masks.to(device)
                outputs = model(images, masks)
            else:
                images, _ = batch
                images = images.to(device)
                outputs = model(images)
            
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return all_predictions, all_probabilities


def calculate_metrics(predictions, targets):
    """Calculate test metrics."""
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='macro', zero_division=0
    )
    
    # Per-class metrics
    class_names = ['damaged', 'occlusion', 'cropped']
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            targets, predictions, labels=[i], average='binary', zero_division=0
        )
        per_class_metrics[class_name] = {
            'precision': float(class_precision),
            'recall': float(class_recall),
            'f1': float(class_f1)
        }
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'per_class': per_class_metrics
    }
    
    return metrics


def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    variant_name = 'model_a'
    checkpoint_path = f'results/{variant_name}/checkpoints/best.pth'
    output_path = f'results/{variant_name}/test_predictions.json'
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading best checkpoint from: {checkpoint_path}")
    
    # Load model
    model, checkpoint = load_best_checkpoint(checkpoint_path, config, variant_name)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model loaded on {device}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best metric: {checkpoint['best_metric']:.4f}")
    
    # Setup test data
    print("Setting up test data...")
    test_loader, test_labels = setup_test_data(config, variant_name)
    
    variant_config = config.get('comparative_training', {}).get('variants', {}).get(variant_name, {})
    use_masks = variant_config.get('use_masks', False)
    
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Using masks: {use_masks}")
    
    # Generate predictions
    print("Generating test predictions...")
    predictions, probabilities = generate_test_predictions(model, test_loader, device, use_masks)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(predictions, test_labels)
    
    # Save results
    results = {
        'predictions': [int(p) for p in predictions],
        'targets': test_labels,
        'probabilities': [p.tolist() for p in probabilities],
        'metrics': metrics,
        'model_info': {
            'variant': variant_name,
            'checkpoint_epoch': checkpoint['epoch'],
            'best_metric': float(checkpoint['best_metric']),
            'use_masks': use_masks
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Test predictions saved to: {output_path}")
    print(f"\nTest Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    
    print(f"\nPer-class metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"  {class_name}: P={class_metrics['precision']:.4f}, R={class_metrics['recall']:.4f}, F1={class_metrics['f1']:.4f}")


if __name__ == "__main__":
    main() 