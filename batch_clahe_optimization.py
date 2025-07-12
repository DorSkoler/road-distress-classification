#!/usr/bin/env python3
"""
Batch CLAHE Parameter Optimization for Road Distress Dataset

This script processes all images in the dataset, finds optimal CLAHE parameters
for each image, and saves the results to a JSON file for use during training.

Usage:
    python batch_clahe_optimization.py --dataset-dir coryell --output-json clahe_params.json
"""

import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, List
from optimize_clahe_for_road_distress import CLAHEOptimizer
np.random.seed(42)  # For reproducible results with slight randomization


class BatchCLAHEOptimizer:
    """Optimizes CLAHE parameters for all images in a dataset"""
    
    def __init__(self, dataset_dir: str, output_json: str, fast_mode: bool = False):
        self.dataset_dir = Path(dataset_dir)
        self.output_json = output_json
        self.results = {}
        self.fast_mode = fast_mode
        
        if fast_mode:
            # Reduced parameter space for faster processing
            self.clip_limits = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]  # 6 instead of 10
            self.tile_grid_sizes = [(4, 4), (8, 8), (12, 12), (16, 16)]  # 4 instead of 8
            # Total combinations: 24 instead of 80 (70% reduction)
        else:
            # Full parameter space for maximum quality
            self.clip_limits = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
            self.tile_grid_sizes = [(3, 3), (4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (16, 16), (20, 20)]
        
    def find_all_images(self) -> List[Path]:
        """Find all image files in the dataset directory"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}
        image_paths = []
        
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    full_path = Path(root) / file
                    image_paths.append(full_path)
        
        return sorted(image_paths)
    
    def optimize_single_image(self, image_path: Path) -> Dict:
        """Optimize CLAHE parameters for a single image"""
        try:
            # Use a simplified optimizer for batch processing
            optimizer = SimpleCLAHEOptimizer(str(image_path), self.clip_limits, self.tile_grid_sizes, self.fast_mode)
            best_config = optimizer.optimize()
            
            # Get relative path for JSON key
            relative_path = str(image_path.relative_to(self.dataset_dir))
            
            return {
                'clip_limit': best_config['clip_limit'],
                'tile_grid_size': best_config['tile_grid_size'],
                'composite_score': best_config['composite_score'],
                'metrics': {
                    'edge_quality': best_config['metrics'].get('edge_quality', 0.0),
                    'contrast_enhancement': best_config['metrics'].get('contrast_enhancement', 0.0),
                    'texture_preservation': best_config['metrics'].get('texture_preservation', 0.0),
                    'noise_penalty': best_config['metrics'].get('noise_penalty', 0.0),
                    'histogram_spread': best_config['metrics'].get('histogram_spread', 0.0),
                    'dynamic_range': best_config['metrics'].get('dynamic_range', 0.0)
                }
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            # Return default parameters on error
            return {
                'clip_limit': 3.0,
                'tile_grid_size': [8, 8],
                'composite_score': 0.0,
                'metrics': {
                    'edge_quality': 0.0,
                    'contrast_enhancement': 1.0,
                    'texture_preservation': 0.0,
                    'noise_penalty': 0.0,
                    'histogram_spread': 0.0,
                    'dynamic_range': 0.0
                }
            }
    
    def process_all_images(self):
        """Process all images and save results to JSON"""
        image_paths = self.find_all_images()
        print(f"Found {len(image_paths)} images to process")
        
        if not image_paths:
            print("No images found in dataset directory!")
            return
        
        # Process images with progress bar
        for image_path in tqdm(image_paths, desc="Optimizing CLAHE parameters"):
            relative_path = str(image_path.relative_to(self.dataset_dir))
            result = self.optimize_single_image(image_path)
            self.results[relative_path] = result
        
        # Save results to JSON
        self.save_to_json()
        print(f"Optimization complete! Results saved to {self.output_json}")
    
    def save_to_json(self):
        """Save optimization results to JSON file"""
        if not self.results:
            print("No results to save!")
            return
        
        # Create output directory if needed
        output_path = Path(self.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON file
        with open(self.output_json, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Saved {len(self.results)} optimization results to {self.output_json}")


class SimpleCLAHEOptimizer:
    """Simplified CLAHE optimizer for batch processing"""
    
    def __init__(self, image_path: str, clip_limits: List[float], tile_grid_sizes: List[Tuple[int, int]], fast_mode: bool = False):
        self.image_path = image_path
        self.clip_limits = clip_limits
        self.tile_grid_sizes = tile_grid_sizes
        self.fast_mode = fast_mode
        
        # Load and prepare image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.lab_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2LAB)
        self.l_channel = self.lab_image[:, :, 0]
    
    def apply_clahe(self, clip_limit: float, tile_grid_size: Tuple[int, int]) -> np.ndarray:
        """Apply CLAHE with given parameters"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_l = clahe.apply(self.l_channel)
        
        enhanced_lab = self.lab_image.copy()
        enhanced_lab[:, :, 0] = enhanced_l
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr
    
    def calculate_fast_quality_score(self, original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
        """Faster quality assessment with reduced computational cost"""
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        metrics = {}
        
        # 1. Edge Enhancement Quality (simplified)
        orig_edges = cv2.Canny(orig_gray, 50, 150)
        enh_edges = cv2.Canny(enh_gray, 50, 150)
        
        orig_edge_density = np.sum(orig_edges > 0) / orig_edges.size
        enh_edge_density = np.sum(enh_edges > 0) / enh_edges.size
        
        edge_enhancement = enh_edge_density / (orig_edge_density + 1e-6)
        metrics['edge_quality'] = min(edge_enhancement, 2.0) / 2.0
        
        # 2. Contrast Enhancement (simplified - using std instead of filtering)
        orig_contrast = np.std(orig_gray)
        enh_contrast = np.std(enh_gray)
        
        contrast_improvement = enh_contrast / (orig_contrast + 1e-6)
        metrics['contrast_enhancement'] = min(contrast_improvement, 3.0) / 3.0
        
        # 3. Texture Preservation (simplified)
        orig_texture = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
        enh_texture = cv2.Laplacian(enh_gray, cv2.CV_64F).var()
        
        texture_ratio = enh_texture / (orig_texture + 1e-6)
        metrics['texture_preservation'] = 1.0 - abs(np.log(texture_ratio + 1e-6)) / 2.0
        
        # 4. Histogram Spread (simplified)
        orig_spread = np.std(orig_gray)
        enh_spread = np.std(enh_gray)
        
        spread_improvement = enh_spread / (orig_spread + 1e-6)
        metrics['histogram_spread'] = min(spread_improvement, 2.0) / 2.0
        
        # 5. Noise Penalty (simplified - using gradient instead of bilateral filter)
        orig_grad = np.mean(np.abs(cv2.Sobel(orig_gray, cv2.CV_64F, 1, 1, ksize=3)))
        enh_grad = np.mean(np.abs(cv2.Sobel(enh_gray, cv2.CV_64F, 1, 1, ksize=3)))
        
        noise_ratio = enh_grad / (orig_grad + 1e-6)
        metrics['noise_penalty'] = 1.0 / (1.0 + max(0, noise_ratio - 1.5))  # Penalize excessive gradients
        
        # 6. Dynamic Range (simplified)
        orig_range = np.percentile(orig_gray, 95) - np.percentile(orig_gray, 5)
        enh_range = np.percentile(enh_gray, 95) - np.percentile(enh_gray, 5)
        
        range_improvement = enh_range / (orig_range + 1e-6)
        metrics['dynamic_range'] = min(range_improvement, 2.0) / 2.0
        
        return metrics
    
    def calculate_comprehensive_quality_score(self, original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
        """Comprehensive quality assessment comparing original vs enhanced (slower but more accurate)"""
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        metrics = {}
        
        # 1. Edge Enhancement Quality
        orig_edges = cv2.Canny(orig_gray, 50, 150)
        enh_edges = cv2.Canny(enh_gray, 50, 150)
        
        orig_edge_density = np.sum(orig_edges > 0) / orig_edges.size
        enh_edge_density = np.sum(enh_edges > 0) / enh_edges.size
        
        edge_enhancement = enh_edge_density / (orig_edge_density + 1e-6)
        metrics['edge_quality'] = min(edge_enhancement, 2.0) / 2.0
        
        # 2. Local Contrast Improvement
        kernel = np.ones((9, 9), np.float32) / 81
        
        orig_mean = cv2.filter2D(orig_gray.astype(np.float32), -1, kernel)
        orig_var = cv2.filter2D((orig_gray.astype(np.float32))**2, -1, kernel) - orig_mean**2
        orig_contrast = np.mean(np.sqrt(orig_var))
        
        enh_mean = cv2.filter2D(enh_gray.astype(np.float32), -1, kernel)
        enh_var = cv2.filter2D((enh_gray.astype(np.float32))**2, -1, kernel) - enh_mean**2
        enh_contrast = np.mean(np.sqrt(enh_var))
        
        contrast_improvement = enh_contrast / (orig_contrast + 1e-6)
        metrics['contrast_enhancement'] = min(contrast_improvement, 3.0) / 3.0
        
        # 3. Texture Preservation vs Enhancement
        orig_texture = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
        enh_texture = cv2.Laplacian(enh_gray, cv2.CV_64F).var()
        
        texture_ratio = enh_texture / (orig_texture + 1e-6)
        metrics['texture_preservation'] = 1.0 - abs(np.log(texture_ratio + 1e-6)) / 2.0
        
        # 4. Histogram Spread Improvement
        orig_hist = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
        enh_hist = cv2.calcHist([enh_gray], [0], None, [256], [0, 256])
        
        orig_spread = np.std(np.where(orig_hist.flatten() > 0)[0])
        enh_spread = np.std(np.where(enh_hist.flatten() > 0)[0])
        
        spread_improvement = enh_spread / (orig_spread + 1e-6)
        metrics['histogram_spread'] = min(spread_improvement, 2.0) / 2.0
        
        # 5. Noise/Artifact Detection (penalize over-enhancement)
        # Use bilateral filter to detect artifacts (expensive but accurate)
        orig_filtered = cv2.bilateralFilter(orig_gray, 9, 75, 75)
        enh_filtered = cv2.bilateralFilter(enh_gray, 9, 75, 75)
        
        orig_noise = np.mean(np.abs(orig_gray.astype(np.float32) - orig_filtered.astype(np.float32)))
        enh_noise = np.mean(np.abs(enh_gray.astype(np.float32) - enh_filtered.astype(np.float32)))
        
        noise_ratio = enh_noise / (orig_noise + 1e-6)
        metrics['noise_penalty'] = 1.0 / (1.0 + noise_ratio)
        
        # 6. Dynamic Range Utilization
        orig_range = np.percentile(orig_gray, 95) - np.percentile(orig_gray, 5)
        enh_range = np.percentile(enh_gray, 95) - np.percentile(enh_gray, 5)
        
        range_improvement = enh_range / (orig_range + 1e-6)
        metrics['dynamic_range'] = min(range_improvement, 2.0) / 2.0
        
        return metrics
    
    def calculate_adaptive_composite_score(self, metrics: Dict[str, float], image_characteristics: Dict[str, float]) -> float:
        """Calculate adaptive score based on image characteristics"""
        
        # Base weights
        weights = {
            'edge_quality': 0.25,
            'contrast_enhancement': 0.25,
            'texture_preservation': 0.20,
            'histogram_spread': 0.15,
            'noise_penalty': 0.10,
            'dynamic_range': 0.05
        }
        
        # Adapt weights based on image characteristics
        if image_characteristics['is_low_contrast']:
            weights['contrast_enhancement'] += 0.1
            weights['dynamic_range'] += 0.05
            
        if image_characteristics['is_low_texture']:
            weights['texture_preservation'] += 0.1
            weights['edge_quality'] += 0.05
            
        if image_characteristics['is_dark']:
            weights['contrast_enhancement'] += 0.1
            weights['histogram_spread'] += 0.05
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate weighted score
        score = sum(weights[key] * metrics[key] for key in weights.keys())
        
        # Add small random factor to break ties
        score += np.random.normal(0, 0.001)
        
        return score
    
    def analyze_image_characteristics(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze image characteristics to adapt scoring"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        characteristics = {}
        
        # Check if image is low contrast
        contrast = np.std(gray)
        characteristics['is_low_contrast'] = contrast < 30
        
        # Check if image is low texture
        texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        characteristics['is_low_texture'] = texture_var < 500
        
        # Check if image is dark
        mean_brightness = np.mean(gray)
        characteristics['is_dark'] = mean_brightness < 100
        
        # Check if image has poor dynamic range
        dynamic_range = np.percentile(gray, 95) - np.percentile(gray, 5)
        characteristics['poor_dynamic_range'] = dynamic_range < 100
        
        return characteristics
    
    def optimize(self) -> Dict:
        """Find optimal CLAHE parameters using comprehensive evaluation"""
        # Analyze image characteristics first
        image_characteristics = self.analyze_image_characteristics(self.original_image)
        
        best_score = -float('inf')
        best_config = None
        
        for clip_limit in self.clip_limits:
            for tile_grid_size in self.tile_grid_sizes:
                # Apply CLAHE
                enhanced_image = self.apply_clahe(clip_limit, tile_grid_size)
                
                # Choose quality evaluation method based on mode
                if self.fast_mode:
                    metrics = self.calculate_fast_quality_score(self.original_image, enhanced_image)
                else:
                    metrics = self.calculate_comprehensive_quality_score(self.original_image, enhanced_image)
                
                # Calculate adaptive score based on image characteristics
                score = self.calculate_adaptive_composite_score(metrics, image_characteristics)
                
                if score > best_score:
                    best_score = score
                    best_config = {
                        'clip_limit': clip_limit,
                        'tile_grid_size': tile_grid_size,
                        'composite_score': score,
                        'metrics': metrics
                    }
        
        # Use default if no valid configuration found
        if best_config is None:
            best_config = {
                'clip_limit': 3.0,
                'tile_grid_size': (8, 8),
                'composite_score': 0.0,
                'metrics': {
                    'edge_quality': 0.0,
                    'contrast_enhancement': 1.0,
                    'texture_preservation': 0.0,
                    'noise_penalty': 1.0,
                    'histogram_spread': 0.0,
                    'dynamic_range': 0.0
                }
            }
        
        return best_config


def main():
    parser = argparse.ArgumentParser(description='Batch CLAHE parameter optimization for road distress dataset')
    parser.add_argument('--dataset-dir', required=True, help='Root directory of the dataset')
    parser.add_argument('--output-json', default='clahe_parameters.json', help='Output JSON file for parameters')
    parser.add_argument('--max-images', type=int, help='Maximum number of images to process (for testing)')
    parser.add_argument('--fast-mode', action='store_true', help='Run in fast mode (3x faster, 24 vs 80 parameter combinations)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        return
    
    # Print mode information
    if args.fast_mode:
        print("Running in FAST mode:")
        print("  - 24 parameter combinations (vs 80 in normal mode)")
        print("  - Simplified quality metrics")
        print("  - Expected speed: ~1 second per image")
    else:
        print("Running in COMPREHENSIVE mode:")
        print("  - 80 parameter combinations")
        print("  - Full quality metrics (including bilateral filtering)")
        print("  - Expected speed: ~3-4 seconds per image")
    
    # Initialize and run batch optimizer
    optimizer = BatchCLAHEOptimizer(args.dataset_dir, args.output_json, args.fast_mode)
    
    if args.max_images:
        print(f"Processing maximum {args.max_images} images for testing")
        # Modify find_all_images to limit results for testing
        original_find = optimizer.find_all_images
        def limited_find():
            all_images = original_find()
            return all_images[:args.max_images]
        optimizer.find_all_images = limited_find
    
    optimizer.process_all_images()
    
    # Print some statistics
    if optimizer.results:
        clip_limits = [r['clip_limit'] for r in optimizer.results.values()]
        grid_sizes = [(r['tile_grid_size'][0], r['tile_grid_size'][1]) for r in optimizer.results.values()]
        
        print(f"\nOptimization Statistics:")
        print(f"Total images processed: {len(optimizer.results)}")
        print(f"Average clip limit: {np.mean(clip_limits):.2f}")
        print(f"Most common clip limit: {max(set(clip_limits), key=clip_limits.count)}")
        print(f"Most common grid size: {max(set(grid_sizes), key=grid_sizes.count)}")
        print(f"Mode used: {'Fast' if args.fast_mode else 'Comprehensive'}")
        
        # Show parameter diversity
        unique_clips = len(set(clip_limits))
        unique_grids = len(set(grid_sizes))
        print(f"Parameter diversity: {unique_clips} unique clip limits, {unique_grids} unique grid sizes")


if __name__ == "__main__":
    main() 