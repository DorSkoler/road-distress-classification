#!/usr/bin/env python3
"""
CLAHE Parameter Optimization for Road Distress Analysis

This script tests multiple CLAHE parameter combinations and selects the best
configuration for road distress detection tasks (damage, occlusion, crop analysis).

Usage:
    python optimize_clahe_for_road_distress.py --image path/to/image.jpg
    python optimize_clahe_for_road_distress.py --image path/to/image.jpg --save-results
"""

import cv2
import numpy as np
import argparse
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import json
from pathlib import Path

# Set random seed for reproducible results with slight randomization
np.random.seed(42)


class CLAHEOptimizer:
    """Optimizes CLAHE parameters for road distress analysis"""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to LAB for better CLAHE results
        self.lab_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2LAB)
        self.l_channel = self.lab_image[:, :, 0]
        
        # Parameter ranges to test - more diverse for better discrimination
        self.clip_limits = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
        self.tile_grid_sizes = [(3, 3), (4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (16, 16), (20, 20)]
        
        self.results = []
    
    def apply_clahe(self, clip_limit: float, tile_grid_size: Tuple[int, int]) -> np.ndarray:
        """Apply CLAHE with given parameters and return enhanced image"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_l = clahe.apply(self.l_channel)
        
        # Reconstruct the image
        enhanced_lab = self.lab_image.copy()
        enhanced_lab[:, :, 0] = enhanced_l
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr
    
    def calculate_edge_quality(self, image: np.ndarray) -> float:
        """Calculate edge detection quality - important for crack detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Canny edge detection with adaptive thresholds
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # If no edges found, return 0
        if edge_density == 0:
            return 0.0
        
        # Calculate edge coherence using connected components
        num_labels, labels = cv2.connectedComponents(edges)
        edge_coherence = 1.0 / max(num_labels, 1)  # Prefer fewer, more coherent edges
        
        # Combine density and coherence with proper bounds
        quality = edge_density * edge_coherence
        return quality if np.isfinite(quality) else 0.0
    
    def calculate_contrast_enhancement(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate contrast improvement - important for shadow/occlusion detection"""
        def local_std(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kernel = np.ones((9, 9), np.float32) / 81
            mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
            return np.sqrt(sqr_mean - mean**2)
        
        original_std = np.mean(local_std(original))
        enhanced_std = np.mean(local_std(enhanced))
        
        contrast_ratio = enhanced_std / (original_std + 1e-6)
        return contrast_ratio if np.isfinite(contrast_ratio) else 1.0
    
    def calculate_texture_preservation(self, image: np.ndarray) -> float:
        """Calculate texture quality - important for road surface analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Laplacian variance as texture measure
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        
        # Normalize by image intensity to be scale-invariant
        mean_intensity = np.mean(gray)
        normalized_texture = laplacian_var / (mean_intensity + 1e-6)
        
        return normalized_texture if np.isfinite(normalized_texture) else 0.0
    
    def calculate_noise_artifacts(self, image: np.ndarray) -> float:
        """Calculate noise/artifacts level - lower is better"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use bilateral filter to separate signal from noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        noise = np.abs(gray.astype(np.float32) - filtered.astype(np.float32))
        
        noise_level = np.mean(noise)
        return noise_level if np.isfinite(noise_level) else 0.0
    
    def calculate_overall_quality(self, image: np.ndarray) -> float:
        """Calculate overall image quality using BRISQUE-like metrics"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Calculate local variance with better numerical stability
        kernel = np.ones((3, 3), np.float32) / 9
        mean = cv2.filter2D(gray, -1, kernel)
        variance = cv2.filter2D(gray**2, -1, kernel) - mean**2
        
        # Ensure variance is non-negative and add stability term
        variance = np.maximum(variance, 0) + 1e-4
        
        # Calculate MSCN (Mean Subtracted Contrast Normalized) coefficients
        mscn = (gray - mean) / np.sqrt(variance)
        
        # Handle potential NaN/inf values
        mscn = np.nan_to_num(mscn, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Return negative of variance of MSCN (higher variance = lower quality)
        quality_score = -np.var(mscn)
        return quality_score if np.isfinite(quality_score) else 0.0
    
    def evaluate_clahe_config(self, clip_limit: float, tile_grid_size: Tuple[int, int]) -> Dict:
        """Evaluate a single CLAHE configuration"""
        enhanced_image = self.apply_clahe(clip_limit, tile_grid_size)
        
        # Calculate all metrics
        edge_quality = self.calculate_edge_quality(enhanced_image)
        contrast_enhancement = self.calculate_contrast_enhancement(self.original_image, enhanced_image)
        texture_preservation = self.calculate_texture_preservation(enhanced_image)
        noise_artifacts = self.calculate_noise_artifacts(enhanced_image)
        overall_quality = self.calculate_overall_quality(enhanced_image)
        
        return {
            'clip_limit': clip_limit,
            'tile_grid_size': tile_grid_size,
            'enhanced_image': enhanced_image,
            'metrics': {
                'edge_quality': edge_quality,
                'contrast_enhancement': contrast_enhancement,
                'texture_preservation': texture_preservation,
                'noise_artifacts': noise_artifacts,
                'overall_quality': overall_quality
            }
        }
    
    def calculate_composite_score(self, metrics: Dict) -> float:
        """Calculate weighted composite score for road distress analysis"""
        # Analyze image characteristics for adaptive weighting
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Image characteristics
        mean_brightness = np.mean(gray)
        contrast_std = np.std(gray)
        texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        dynamic_range = np.percentile(gray, 95) - np.percentile(gray, 5)
        
        # Adaptive weights based on image characteristics
        weights = {
            'edge_quality': 0.35,          # High weight - crucial for crack detection
            'contrast_enhancement': 0.25,   # Important for shadow/occlusion handling
            'texture_preservation': 0.20,   # Important for road surface analysis
            'noise_artifacts': -0.10,       # Negative weight - penalize noise
            'overall_quality': 0.30         # Overall image quality
        }
        
        # Adapt weights based on image characteristics
        if mean_brightness < 100:  # Dark image
            weights['contrast_enhancement'] += 0.05
            weights['overall_quality'] += 0.05
            
        if contrast_std < 30:  # Low contrast image
            weights['contrast_enhancement'] += 0.1
            weights['edge_quality'] += 0.05
            
        if texture_var < 500:  # Low texture image
            weights['texture_preservation'] += 0.1
            weights['edge_quality'] += 0.05
            
        if dynamic_range < 100:  # Poor dynamic range
            weights['contrast_enhancement'] += 0.05
            weights['overall_quality'] += 0.05
        
        # Normalize weights to sum to 1.0
        total_positive_weight = sum(max(0, w) for w in weights.values())
        for key in weights:
            if weights[key] > 0:
                weights[key] = weights[key] / total_positive_weight
        
        # Handle NaN/inf values in metrics
        clean_metrics = {}
        for key, value in metrics.items():
            if np.isfinite(value):
                clean_metrics[key] = value
            else:
                clean_metrics[key] = 0.0
        
        # Normalize noise artifacts (lower is better, so invert)
        normalized_metrics = clean_metrics.copy()
        normalized_metrics['noise_artifacts'] = 1.0 / (1.0 + clean_metrics['noise_artifacts'])
        
        score = sum(weights[key] * normalized_metrics[key] for key in weights.keys())
        
        # Add small random factor to break ties between similar scores
        score += np.random.normal(0, 0.001)
        
        return score if np.isfinite(score) else 0.0
    
    def optimize(self) -> Dict:
        """Find optimal CLAHE parameters"""
        print(f"Optimizing CLAHE parameters for: {self.image_path}")
        print(f"Testing {len(self.clip_limits)} clip limits × {len(self.tile_grid_sizes)} tile sizes = {len(self.clip_limits) * len(self.tile_grid_sizes)} combinations")
        
        best_score = -float('inf')
        best_config = None
        
        for i, clip_limit in enumerate(self.clip_limits):
            for j, tile_grid_size in enumerate(self.tile_grid_sizes):
                config = self.evaluate_clahe_config(clip_limit, tile_grid_size)
                score = self.calculate_composite_score(config['metrics'])
                config['composite_score'] = score
                
                self.results.append(config)
                
                if score > best_score:
                    best_score = score
                    best_config = config
                
                print(f"Progress: {i * len(self.tile_grid_sizes) + j + 1:2d}/{len(self.clip_limits) * len(self.tile_grid_sizes)} | "
                      f"Clip: {clip_limit:3.1f}, Grid: {tile_grid_size}, Score: {score:.4f}")
        
        # Sort results by score
        self.results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print(f"\nOptimization complete!")
        
        if best_config is None:
            print("Warning: All configurations resulted in invalid scores. Using default configuration.")
            # Use a reasonable default configuration
            best_config = {
                'clip_limit': 3.0,
                'tile_grid_size': (8, 8),
                'composite_score': 0.0,
                'enhanced_image': self.apply_clahe(3.0, (8, 8)),
                'metrics': {
                    'edge_quality': 0.0,
                    'contrast_enhancement': 1.0,
                    'texture_preservation': 0.0,
                    'noise_artifacts': 0.0,
                    'overall_quality': 0.0
                }
            }
        
        print(f"Best configuration:")
        print(f"  Clip Limit: {best_config['clip_limit']}")
        print(f"  Tile Grid Size: {best_config['tile_grid_size']}")
        print(f"  Composite Score: {best_config['composite_score']:.4f}")
        
        return best_config
    
    def save_results(self, output_dir: str):
        """Save optimization results and visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save numerical results
        results_data = []
        for result in self.results:
            # Convert numpy types to native Python types for JSON serialization
            data = {
                'clip_limit': float(result['clip_limit']),
                'tile_grid_size': result['tile_grid_size'],
                'composite_score': float(result['composite_score']),
                'metrics': {k: float(v) for k, v in result['metrics'].items()}
            }
            results_data.append(data)
        
        with open(output_path / 'clahe_optimization_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Create comparison visualization
        self.create_comparison_plot(output_path)
        
        # Save top configurations
        self.save_top_configs(output_path)
        
        print(f"Results saved to: {output_path}")
    
    def create_comparison_plot(self, output_path: Path):
        """Create visualization comparing top CLAHE configurations"""
        top_configs = self.results[:6]  # Top 6 configurations
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, config in enumerate(top_configs):
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(config['enhanced_image'], cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img_rgb)
            axes[i].set_title(f"Rank {i+1}\n"
                             f"Clip: {config['clip_limit']}, Grid: {config['tile_grid_size']}\n"
                             f"Score: {config['composite_score']:.4f}", fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / 'clahe_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create metrics comparison
        self.create_metrics_plot(output_path)
    
    def create_metrics_plot(self, output_path: Path):
        """Create detailed metrics comparison plot"""
        top_configs = self.results[:10]  # Top 10 for metrics analysis
        
        metrics_names = ['edge_quality', 'contrast_enhancement', 'texture_preservation', 'overall_quality']
        config_labels = [f"C{c['clip_limit']}_G{c['tile_grid_size'][0]}" for c in top_configs]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_names):
            values = [c['metrics'][metric] for c in top_configs]
            axes[i].bar(range(len(values)), values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xticks(range(len(values)))
            axes[i].set_xticklabels(config_labels, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_top_configs(self, output_path: Path, num_configs: int = 5):
        """Save images with top CLAHE configurations"""
        for i, config in enumerate(self.results[:num_configs]):
            filename = f"rank_{i+1}_clip_{config['clip_limit']}_grid_{config['tile_grid_size'][0]}x{config['tile_grid_size'][1]}.png"
            cv2.imwrite(str(output_path / filename), config['enhanced_image'])
    
    def apply_best_clahe(self, image_path: Optional[str] = None) -> np.ndarray:
        """Apply the best CLAHE configuration to an image"""
        if not self.results:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        if image_path:
            img = cv2.imread(image_path)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
        else:
            img = self.original_image
            lab = self.lab_image
            l_channel = self.l_channel
        
        best_config = self.results[0]
        clahe = cv2.createCLAHE(
            clipLimit=best_config['clip_limit'],
            tileGridSize=best_config['tile_grid_size']
        )
        
        enhanced_l = clahe.apply(l_channel)
        enhanced_lab = lab.copy()
        enhanced_lab[:, :, 0] = enhanced_l
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr


def main():
    parser = argparse.ArgumentParser(description='Optimize CLAHE parameters for road distress analysis')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--save-results', action='store_true', help='Save detailed results and visualizations')
    parser.add_argument('--output-dir', default='clahe_optimization_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Initialize optimizer and run optimization
    optimizer = CLAHEOptimizer(args.image)
    best_config = optimizer.optimize()
    
    # Display results
    print(f"\nDetailed metrics for best configuration:")
    for metric, value in best_config['metrics'].items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    if args.save_results:
        optimizer.save_results(args.output_dir)
        
        # Also save the original and best enhanced image side by side
        original = cv2.imread(args.image)
        enhanced = optimizer.apply_best_clahe()
        
        # Create side-by-side comparison
        comparison = np.hstack([original, enhanced])
        cv2.imwrite(os.path.join(args.output_dir, 'before_after_comparison.png'), comparison)
        
        print(f"\nBest enhanced image saved as: {os.path.join(args.output_dir, 'before_after_comparison.png')}")
    
    print(f"\nRecommended CLAHE parameters for road distress analysis:")
    print(f"  cv2.createCLAHE(clipLimit={best_config['clip_limit']}, tileGridSize={best_config['tile_grid_size']})")


if __name__ == "__main__":
    main() 