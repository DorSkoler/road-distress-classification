#!/usr/bin/env python3
"""
Test Model Confidence Script

Tests models B and H on test images to get confidence scores for occlusion, damage, and crop labels.
Start with a single image first, then expand to all test images.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import yaml
import json
import cv2

# Add the src directory to the Python path (adjusted for new location)
current_dir = Path(__file__).parent
src_dir = current_dir / ".." / "2025-07-05_hybrid_training" / "src"
sys.path.append(str(src_dir))

from models.hybrid_model import create_model
from data.dataset import create_dataset
from torch.utils.data import DataLoader

class ModelConfidenceTester:
    """Test model confidence scores for multiple models."""
    
    def __init__(self, base_config_path: str):
        """Initialize with base configuration."""
        self.base_config_path = Path(base_config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load base config
        with open(self.base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Load CLAHE parameters for preprocessing
        self.load_clahe_params()
    
    def load_model(self, variant: str, config_path: str = None) -> torch.nn.Module:
        """Load a specific model variant."""
        # Load variant-specific config if provided
        if config_path:
            with open(config_path, 'r') as f:
                variant_config = yaml.safe_load(f)
        else:
            variant_config = self.base_config
        
        # Create model
        model_config = variant_config.get('model', {})
        model = create_model(
            variant=variant,
            num_classes=model_config.get('num_classes', 3),
            dropout_rate=model_config.get('classifier', {}).get('dropout_rate', 0.5),
            encoder_name=model_config.get('encoder_name', 'efficientnet_b3'),
            encoder_weights=model_config.get('encoder_weights', 'imagenet')
        )
        
        # Try to load checkpoint (adjusted for new location)
        results_dir = Path(f"../2025-07-05_hybrid_training/results/model_{variant[-1]}")
        checkpoint_path = results_dir / "checkpoints" / "best_model.pth"
        
        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}")
            print("Using model with random weights")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def load_clahe_params(self):
        """Load CLAHE parameters from JSON file."""
        # Look for CLAHE params in the main project directory
        clahe_params_path = Path("../../clahe_params.json")
        
        if not clahe_params_path.exists():
            print(f"Warning: CLAHE parameters file not found: {clahe_params_path}")
            print("Using default CLAHE parameters for all images")
            self.clahe_params = {}
            return
        
        try:
            with open(clahe_params_path, 'r') as f:
                data = json.load(f)
            
            # Convert JSON data to expected format
            self.clahe_params = {}
            for image_path, param_data in data.items():
                if 'tile_grid_size' in param_data:
                    tile_grid_size = param_data['tile_grid_size']
                    tile_grid_x, tile_grid_y = tile_grid_size[0], tile_grid_size[1]
                else:
                    # Fallback values
                    tile_grid_x, tile_grid_y = 8, 8
                
                self.clahe_params[image_path] = {
                    'clip_limit': param_data.get('clip_limit', 3.0),
                    'tile_grid_x': tile_grid_x,
                    'tile_grid_y': tile_grid_y
                }
            
            print(f"✓ Loaded CLAHE parameters for {len(self.clahe_params)} images")
            
        except Exception as e:
            print(f"Error loading CLAHE parameters from {clahe_params_path}: {str(e)}")
            print("Using default CLAHE parameters for all images")
            self.clahe_params = {}
    
    def apply_clahe(self, image: np.ndarray, image_path: str) -> np.ndarray:
        """Apply CLAHE enhancement to image."""
        import cv2
        
        # Get CLAHE parameters for this image
        # Extract the relative path in the format expected by clahe_params.json
        path_obj = Path(image_path)
        
        # Convert to the format used in clahe_params.json: "Co Rd XXX\img\filename.png"
        if 'coryell' in path_obj.parts:
            coryell_idx = path_obj.parts.index('coryell')
            if coryell_idx + 1 < len(path_obj.parts):
                # Get road name and filename
                road_name = path_obj.parts[coryell_idx + 1]
                filename = path_obj.name
                # Format: "Co Rd XXX\img\filename.png" (with backslashes as in JSON)
                clahe_key = f"{road_name}\\img\\{filename}"
            else:
                clahe_key = str(path_obj.name)
        else:
            clahe_key = str(path_obj.name)
        
        # Try both formats if first doesn't work
        clahe_params = self.clahe_params.get(clahe_key)
        if clahe_params is None:
            # Try with forward slashes
            clahe_key_forward = clahe_key.replace('\\', '/')
            clahe_params = self.clahe_params.get(clahe_key_forward)
        
        if clahe_params is None:
            # Default parameters
            clahe_params = {
                'clip_limit': 3.0,
                'tile_grid_x': 8,
                'tile_grid_y': 8
            }
            print(f"Warning: Using default CLAHE params for {clahe_key}")
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=clahe_params['clip_limit'],
            tileGridSize=(clahe_params['tile_grid_x'], clahe_params['tile_grid_y'])
        )
        enhanced_l = clahe.apply(l_channel)
        
        # Reconstruct image
        lab[:, :, 0] = enhanced_l
        enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr
    
    def uses_clahe(self, variant: str) -> bool:
        """Check if a model variant uses CLAHE preprocessing."""
        clahe_variants = ['model_e', 'model_f', 'model_g', 'model_h']
        return variant in clahe_variants
    
    def test_single_image(self, image_path: str, models: dict) -> dict:
        """Test a single image with multiple models."""
        print(f"\nTesting image: {image_path}")
        
        results = {}
        
        for model_name, model_info in models.items():
            print(f"Testing with {model_name}...")
            
            model = model_info['model']
            variant = model_info['variant']
            
            # Load and preprocess image based on model requirements
            import cv2
            
            # Load image in BGR format (OpenCV default)
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Apply CLAHE preprocessing if the model uses it
            if self.uses_clahe(variant):
                print(f"  → Applying CLAHE preprocessing for {model_name}")
                image_bgr = self.apply_clahe(image_bgr, image_path)
            else:
                print(f"  → Using raw image for {model_name}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            image_size = self.base_config['dataset']['image_size']
            image_pil = Image.fromarray(image_rgb).resize(tuple(image_size))
            
            # Convert to tensor and normalize
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get model predictions
                outputs = model(image_tensor)
                probabilities = torch.sigmoid(outputs)
                
                # Extract confidence scores
                confidences = probabilities.cpu().numpy()[0]
                
                results[model_name] = {
                    'damage_confidence': float(confidences[0]),
                    'occlusion_confidence': float(confidences[1]), 
                    'crop_confidence': float(confidences[2]),
                    'predictions': {
                        'damage': bool(confidences[0] > 0.5),
                        'occlusion': bool(confidences[1] > 0.5),
                        'crop': bool(confidences[2] > 0.5)
                    }
                }
        
        return results
    
    def get_test_image_info(self, image_path: str) -> dict:
        """Get ground truth labels for a test image."""
        # Load test labels (adjusted for new location)
        test_labels_path = Path("../../test_labels.csv")
        if test_labels_path.exists():
            df = pd.read_csv(test_labels_path)
            
            # Find matching row (handle path format differences)
            image_name = str(Path(image_path).relative_to(Path("../../coryell")))
            # Try different formats
            for _, row in df.iterrows():
                if image_name in row['image_name'] or row['image_name'] in image_path:
                    return {
                        'image_name': row['image_name'],
                        'damage': bool(row['damage']),
                        'occlusion': bool(row['occlusion']),
                        'crop': bool(row['crop'])
                    }
        
        print(f"Warning: Could not find ground truth labels for {image_path}")
        return None

def test_all_images(tester, models, num_images=None):
    """Test all images or a subset of test images."""
    test_labels_path = Path("../../test_labels.csv")
    if not test_labels_path.exists():
        print(f"❌ Test labels file not found: {test_labels_path}")
        return
    
    df = pd.read_csv(test_labels_path)
    if num_images:
        df = df.head(num_images)
    
    print(f"\n🔍 Testing {len(df)} images...")
    
    all_results = []
    successful_tests = 0
    
    for idx, row in df.iterrows():
        image_name = row['image_name']
        test_image_path = Path("../../coryell") / image_name
        
        if not test_image_path.exists():
            print(f"⚠️  Image not found: {test_image_path}")
            continue
        
        try:
            # Get ground truth
            ground_truth = {
                'image_name': image_name,
                'damage': bool(row['damage']),
                'occlusion': bool(row['occlusion']),
                'crop': bool(row['crop'])
            }
            
            # Test with models
            results = tester.test_single_image(str(test_image_path), models)
            
            # Store results
            test_result = {
                'image_index': idx,
                'image_path': str(test_image_path),
                'ground_truth': ground_truth,
                'model_results': results
            }
            all_results.append(test_result)
            successful_tests += 1
            
            # Progress indicator
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(df)} images...")
                
        except Exception as e:
            print(f"❌ Error processing {test_image_path}: {e}")
    
    print(f"\n✅ Successfully tested {successful_tests}/{len(df)} images")
    
    # Save results
    output_data = {
        'test_summary': {
            'total_images_attempted': len(df),
            'successful_tests': successful_tests,
            'models_tested': list(models.keys())
        },
        'results': all_results,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open('all_images_confidence_test.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Results saved to: all_images_confidence_test.json")
    
    # Display summary statistics and find optimal threshold
    optimal_results = print_summary_statistics(all_results, models)
    
    # Save optimal threshold results if found
    if optimal_results:
        output_data['optimal_ensemble'] = optimal_results
        print(f"\n💾 Optimal ensemble results included in output file")

def find_optimal_ensemble_threshold(all_results, models):
    """Find optimal threshold for ensemble of Model B and Model H."""
    
    if 'Model_B' not in models or 'Model_H' not in models:
        print("⚠️  Both Model B and Model H required for ensemble analysis")
        return None
    
    print("\n🔍 FINDING OPTIMAL ENSEMBLE THRESHOLD:")
    print("=" * 80)
    
    # Test thresholds from 0.1 to 0.9 in steps of 0.05
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    best_overall_threshold = 0.5
    best_overall_accuracy = 0.0
    
    # Per-class optimal thresholds
    best_damage_threshold = 0.5
    best_occlusion_threshold = 0.5
    best_crop_threshold = 0.5
    
    best_damage_accuracy = 0.0
    best_occlusion_accuracy = 0.0
    best_crop_accuracy = 0.0
    
    threshold_results = []
    
    for threshold in thresholds:
        # Calculate ensemble predictions for this threshold
        correct_damage = 0
        correct_occlusion = 0
        correct_crop = 0
        total_predictions = 0
        
        for result in all_results:
            model_b_result = result['model_results']['Model_B']
            model_h_result = result['model_results']['Model_H']
            ground_truth = result['ground_truth']
            
            # Calculate ensemble confidence (average of both models)
            ensemble_damage = (model_b_result['damage_confidence'] + model_h_result['damage_confidence']) / 2
            ensemble_occlusion = (model_b_result['occlusion_confidence'] + model_h_result['occlusion_confidence']) / 2
            ensemble_crop = (model_b_result['crop_confidence'] + model_h_result['crop_confidence']) / 2
            
            # Make binary predictions using threshold
            pred_damage = ensemble_damage > threshold
            pred_occlusion = ensemble_occlusion > threshold
            pred_crop = ensemble_crop > threshold
            
            # Check against ground truth
            if pred_damage == ground_truth['damage']:
                correct_damage += 1
            if pred_occlusion == ground_truth['occlusion']:
                correct_occlusion += 1
            if pred_crop == ground_truth['crop']:
                correct_crop += 1
            
            total_predictions += 3  # 3 labels per image
        
        # Calculate accuracies
        total_samples = len(all_results)
        damage_accuracy = correct_damage / total_samples
        occlusion_accuracy = correct_occlusion / total_samples
        crop_accuracy = correct_crop / total_samples
        overall_accuracy = (correct_damage + correct_occlusion + correct_crop) / (total_samples * 3)
        
        threshold_results.append({
            'threshold': threshold,
            'overall_accuracy': overall_accuracy,
            'damage_accuracy': damage_accuracy,
            'occlusion_accuracy': occlusion_accuracy,
            'crop_accuracy': crop_accuracy
        })
        
        # Track best overall threshold
        if overall_accuracy > best_overall_accuracy:
            best_overall_accuracy = overall_accuracy
            best_overall_threshold = threshold
        
        # Track best per-class thresholds
        if damage_accuracy > best_damage_accuracy:
            best_damage_accuracy = damage_accuracy
            best_damage_threshold = threshold
        
        if occlusion_accuracy > best_occlusion_accuracy:
            best_occlusion_accuracy = occlusion_accuracy
            best_occlusion_threshold = threshold
        
        if crop_accuracy > best_crop_accuracy:
            best_crop_accuracy = crop_accuracy
            best_crop_threshold = threshold
    
    # Print results
    print(f"\n🏆 OPTIMAL THRESHOLDS FOUND:")
    print(f"  Overall Best:    {best_overall_threshold:.2f} (accuracy: {best_overall_accuracy:.3f})")
    print(f"  Damage Best:     {best_damage_threshold:.2f} (accuracy: {best_damage_accuracy:.3f})")
    print(f"  Occlusion Best:  {best_occlusion_threshold:.2f} (accuracy: {best_occlusion_accuracy:.3f})")
    print(f"  Crop Best:       {best_crop_threshold:.2f} (accuracy: {best_crop_accuracy:.3f})")
    
    # Show top 5 thresholds for overall accuracy
    threshold_results.sort(key=lambda x: x['overall_accuracy'], reverse=True)
    print(f"\n📊 TOP 5 THRESHOLDS (Overall Accuracy):")
    for i, result in enumerate(threshold_results[:5]):
        print(f"  {i+1}. Threshold {result['threshold']:.2f}: {result['overall_accuracy']:.3f} accuracy")
    
    # Test the best overall threshold
    print(f"\n🎯 ENSEMBLE PERFORMANCE AT OPTIMAL SINGLE THRESHOLD ({best_overall_threshold:.2f}):")
    test_ensemble_threshold(all_results, best_overall_threshold)
    
    # Test per-class thresholds
    print(f"\n🎯 ENSEMBLE PERFORMANCE WITH PER-CLASS THRESHOLDS:")
    print(f"    Damage: {best_damage_threshold:.2f}, Occlusion: {best_occlusion_threshold:.2f}, Crop: {best_crop_threshold:.2f}")
    per_class_accuracy = test_ensemble_per_class_thresholds(all_results, 
                                                          best_damage_threshold, 
                                                          best_occlusion_threshold, 
                                                          best_crop_threshold)
    
    return {
        'best_overall_threshold': best_overall_threshold,
        'best_overall_accuracy': best_overall_accuracy,
        'best_damage_threshold': best_damage_threshold,
        'best_occlusion_threshold': best_occlusion_threshold,
        'best_crop_threshold': best_crop_threshold,
        'per_class_accuracy': per_class_accuracy,
        'threshold_results': threshold_results
    }

def test_ensemble_threshold(all_results, threshold):
    """Test ensemble performance at specific threshold."""
    
    correct_damage = 0
    correct_occlusion = 0
    correct_crop = 0
    
    ensemble_damage_confidences = []
    ensemble_occlusion_confidences = []
    ensemble_crop_confidences = []
    
    for result in all_results:
        model_b_result = result['model_results']['Model_B']
        model_h_result = result['model_results']['Model_H']
        ground_truth = result['ground_truth']
        
        # Calculate ensemble confidence (average of both models)
        ensemble_damage = (model_b_result['damage_confidence'] + model_h_result['damage_confidence']) / 2
        ensemble_occlusion = (model_b_result['occlusion_confidence'] + model_h_result['occlusion_confidence']) / 2
        ensemble_crop = (model_b_result['crop_confidence'] + model_h_result['crop_confidence']) / 2
        
        ensemble_damage_confidences.append(ensemble_damage)
        ensemble_occlusion_confidences.append(ensemble_occlusion)
        ensemble_crop_confidences.append(ensemble_crop)
        
        # Make binary predictions using threshold
        pred_damage = ensemble_damage > threshold
        pred_occlusion = ensemble_occlusion > threshold
        pred_crop = ensemble_crop > threshold
        
        # Check against ground truth
        if pred_damage == ground_truth['damage']:
            correct_damage += 1
        if pred_occlusion == ground_truth['occlusion']:
            correct_occlusion += 1
        if pred_crop == ground_truth['crop']:
            correct_crop += 1
    
    total = len(all_results)
    
    print(f"  Accuracy:")
    print(f"    Damage:    {correct_damage}/{total} ({correct_damage/total:.3f})")
    print(f"    Occlusion: {correct_occlusion}/{total} ({correct_occlusion/total:.3f})")
    print(f"    Crop:      {correct_crop}/{total} ({correct_crop/total:.3f})")
    print(f"    Overall:   {(correct_damage + correct_occlusion + correct_crop)/(3*total):.3f}")
    
    print(f"  Average Ensemble Confidence:")
    print(f"    Damage:    {np.mean(ensemble_damage_confidences):.3f} (±{np.std(ensemble_damage_confidences):.3f})")
    print(f"    Occlusion: {np.mean(ensemble_occlusion_confidences):.3f} (±{np.std(ensemble_occlusion_confidences):.3f})")
    print(f"    Crop:      {np.mean(ensemble_crop_confidences):.3f} (±{np.std(ensemble_crop_confidences):.3f})")

def test_ensemble_per_class_thresholds(all_results, damage_threshold, occlusion_threshold, crop_threshold):
    """Test ensemble performance with different thresholds per class."""
    
    correct_damage = 0
    correct_occlusion = 0
    correct_crop = 0
    
    ensemble_damage_confidences = []
    ensemble_occlusion_confidences = []
    ensemble_crop_confidences = []
    
    for result in all_results:
        model_b_result = result['model_results']['Model_B']
        model_h_result = result['model_results']['Model_H']
        ground_truth = result['ground_truth']
        
        # Calculate ensemble confidence (average of both models)
        ensemble_damage = (model_b_result['damage_confidence'] + model_h_result['damage_confidence']) / 2
        ensemble_occlusion = (model_b_result['occlusion_confidence'] + model_h_result['occlusion_confidence']) / 2
        ensemble_crop = (model_b_result['crop_confidence'] + model_h_result['crop_confidence']) / 2
        
        ensemble_damage_confidences.append(ensemble_damage)
        ensemble_occlusion_confidences.append(ensemble_occlusion)
        ensemble_crop_confidences.append(ensemble_crop)
        
        # Make binary predictions using PER-CLASS thresholds
        pred_damage = ensemble_damage > damage_threshold
        pred_occlusion = ensemble_occlusion > occlusion_threshold
        pred_crop = ensemble_crop > crop_threshold
        
        # Check against ground truth
        if pred_damage == ground_truth['damage']:
            correct_damage += 1
        if pred_occlusion == ground_truth['occlusion']:
            correct_occlusion += 1
        if pred_crop == ground_truth['crop']:
            correct_crop += 1
    
    total = len(all_results)
    damage_accuracy = correct_damage / total
    occlusion_accuracy = correct_occlusion / total
    crop_accuracy = correct_crop / total
    overall_accuracy = (correct_damage + correct_occlusion + correct_crop) / (total * 3)
    
    print(f"  Accuracy:")
    print(f"    Damage:    {correct_damage}/{total} ({damage_accuracy:.3f}) [threshold: {damage_threshold:.2f}]")
    print(f"    Occlusion: {correct_occlusion}/{total} ({occlusion_accuracy:.3f}) [threshold: {occlusion_threshold:.2f}]")
    print(f"    Crop:      {correct_crop}/{total} ({crop_accuracy:.3f}) [threshold: {crop_threshold:.2f}]")
    print(f"    Overall:   {(correct_damage + correct_occlusion + correct_crop)}/{total * 3} ({overall_accuracy:.3f})")
    
    print(f"  Average Ensemble Confidence:")
    print(f"    Damage:    {np.mean(ensemble_damage_confidences):.3f} (±{np.std(ensemble_damage_confidences):.3f})")
    print(f"    Occlusion: {np.mean(ensemble_occlusion_confidences):.3f} (±{np.std(ensemble_occlusion_confidences):.3f})")
    print(f"    Crop:      {np.mean(ensemble_crop_confidences):.3f} (±{np.std(ensemble_crop_confidences):.3f})")
    
    return {
        'damage_accuracy': damage_accuracy,
        'occlusion_accuracy': occlusion_accuracy,
        'crop_accuracy': crop_accuracy,
        'overall_accuracy': overall_accuracy,
        'damage_threshold': damage_threshold,
        'occlusion_threshold': occlusion_threshold,
        'crop_threshold': crop_threshold
    }

def print_summary_statistics(all_results, models):
    """Print summary statistics for all test results."""
    if not all_results:
        return
    
    print("\n📊 INDIVIDUAL MODEL STATISTICS:")
    print("=" * 80)
    
    for model_name in models.keys():
        print(f"\n{model_name}:")
        
        # Collect all confidence scores and predictions
        damage_confidences = []
        occlusion_confidences = []
        crop_confidences = []
        
        correct_damage = 0
        correct_occlusion = 0
        correct_crop = 0
        
        for result in all_results:
            model_result = result['model_results'][model_name]
            ground_truth = result['ground_truth']
            
            damage_confidences.append(model_result['damage_confidence'])
            occlusion_confidences.append(model_result['occlusion_confidence'])
            crop_confidences.append(model_result['crop_confidence'])
            
            if model_result['predictions']['damage'] == ground_truth['damage']:
                correct_damage += 1
            if model_result['predictions']['occlusion'] == ground_truth['occlusion']:
                correct_occlusion += 1
            if model_result['predictions']['crop'] == ground_truth['crop']:
                correct_crop += 1
        
        total = len(all_results)
        
        print(f"  Accuracy:")
        print(f"    Damage:    {correct_damage}/{total} ({correct_damage/total:.3f})")
        print(f"    Occlusion: {correct_occlusion}/{total} ({correct_occlusion/total:.3f})")
        print(f"    Crop:      {correct_crop}/{total} ({correct_crop/total:.3f})")
        print(f"    Overall:   {(correct_damage + correct_occlusion + correct_crop)/(3*total):.3f}")
        
        print(f"  Average Confidence:")
        print(f"    Damage:    {np.mean(damage_confidences):.3f} (±{np.std(damage_confidences):.3f})")
        print(f"    Occlusion: {np.mean(occlusion_confidences):.3f} (±{np.std(occlusion_confidences):.3f})")
        print(f"    Crop:      {np.mean(crop_confidences):.3f} (±{np.std(crop_confidences):.3f})")
    
    # Find optimal ensemble threshold
    if len(models) >= 2:
        optimal_results = find_optimal_ensemble_threshold(all_results, models)
        return optimal_results
    
    return None

def main():
    """Main function to test model confidence."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model confidence on road distress images')
    parser.add_argument('--mode', choices=['single', 'sample', 'all'], default='single',
                        help='Test mode: single image, sample of images, or all images')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to test in sample mode')
    
    args = parser.parse_args()
    
    # Configuration paths (adjusted for new location)
    base_config = "../2025-07-05_hybrid_training/config/base_config.yaml"
    model_b_config = "model_b_config.yaml"  # Local copy
    model_h_config = "../2025-07-05_hybrid_training/config/model_h_config.yaml"
    
    # Initialize tester
    tester = ModelConfidenceTester(base_config)
    
    # Load models
    print("Loading models...")
    models = {}
    
    try:
        model_b = tester.load_model('model_b', model_b_config)
        models['Model_B'] = {
            'model': model_b,
            'variant': 'model_b'
        }
        print("✓ Model B loaded successfully (no CLAHE)")
    except Exception as e:
        print(f"✗ Failed to load Model B: {e}")
    
    try:
        model_h = tester.load_model('model_h', model_h_config)
        models['Model_H'] = {
            'model': model_h,
            'variant': 'model_h'
        }
        print("✓ Model H loaded successfully (uses CLAHE)")
    except Exception as e:
        print(f"✗ Failed to load Model H: {e}")
    
    if not models:
        print("❌ No models loaded successfully. Cannot proceed.")
        return
    
    if args.mode == 'single':
        # Test on a single image first
        test_labels_path = Path("../../test_labels.csv")
        if test_labels_path.exists():
            df = pd.read_csv(test_labels_path)
            first_image = df.iloc[0]['image_name']
            
            # Construct full path
            test_image_path = Path("../../coryell") / first_image
            
            if test_image_path.exists():
                print(f"\n🔍 Testing single image: {test_image_path}")
                
                # Get ground truth
                ground_truth = tester.get_test_image_info(str(test_image_path))
                if ground_truth:
                    print(f"Ground Truth - Damage: {ground_truth['damage']}, Occlusion: {ground_truth['occlusion']}, Crop: {ground_truth['crop']}")
                
                # Test with models
                results = tester.test_single_image(str(test_image_path), models)
                
                # Display results
                print("\n📊 CONFIDENCE SCORES:")
                print("-" * 80)
                
                for model_name, model_results in results.items():
                    print(f"\n{model_name}:")
                    print(f"  Damage:    {model_results['damage_confidence']:.3f} ({'✓' if model_results['predictions']['damage'] else '✗'})")
                    print(f"  Occlusion: {model_results['occlusion_confidence']:.3f} ({'✓' if model_results['predictions']['occlusion'] else '✗'})")
                    print(f"  Crop:      {model_results['crop_confidence']:.3f} ({'✓' if model_results['predictions']['crop'] else '✗'})")
                
                # Save results to JSON
                output_data = {
                    'test_image': str(test_image_path),
                    'ground_truth': ground_truth,
                    'model_results': results,
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                with open('single_image_confidence_test.json', 'w') as f:
                    json.dump(output_data, f, indent=2)
                
                print(f"\n💾 Results saved to: single_image_confidence_test.json")
                
            else:
                print(f"❌ Test image not found: {test_image_path}")
        else:
            print(f"❌ Test labels file not found: {test_labels_path}")
    
    elif args.mode == 'sample':
        test_all_images(tester, models, args.num_images)
    
    elif args.mode == 'all':
        test_all_images(tester, models)

if __name__ == "__main__":
    main()