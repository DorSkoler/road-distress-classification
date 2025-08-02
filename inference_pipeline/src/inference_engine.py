#!/usr/bin/env python3
"""
Inference Engine for Road Distress Classification
Date: 2025-08-01

This module provides the core inference engine that combines model loading,
image preprocessing, and confidence extraction for road distress detection.
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import logging

from .model_loader import ModelLoader
from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Main inference engine for road distress classification.
    
    Combines model loading, image preprocessing, and confidence extraction
    to provide a unified interface for inference on arbitrary images.
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 device: Optional[str] = None,
                 target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize the inference engine.
        
        Args:
            checkpoint_path: Path to the best_model.pth checkpoint
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            target_size: Target image size for model input
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_size = target_size
        
        # Initialize components
        self.model_loader = ModelLoader(checkpoint_path, device=self.device)
        self.image_processor = ImageProcessor(target_size=target_size)
        
        # Load model
        self.model = self.model_loader.load_model()
        self.model_info = self.model_loader.get_model_info()
        
        # Class information
        self.class_names = ['damage', 'occlusion', 'crop']
        self.class_colors = {
            'damage': (255, 0, 0),      # Red
            'occlusion': (0, 255, 0),   # Green  
            'crop': (0, 0, 255)         # Blue
        }
        
        logger.info(f"InferenceEngine initialized:")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Target size: {target_size}")
        logger.info(f"  - Classes: {self.class_names}")
    
    def predict_single(self, image: Union[str, np.ndarray]) -> Dict[str, Union[np.ndarray, float, Dict]]:
        """
        Perform inference on a single image.
        
        Args:
            image: Input image (file path or numpy array)
            
        Returns:
            Dictionary containing:
            - probabilities: Class probabilities [3]
            - predictions: Binary predictions [3] 
            - confidence: Confidence scores [3]
            - class_results: Per-class detailed results
            - overall_confidence: Overall prediction confidence
        """
        # Get original image size for later use
        original_size = self.image_processor.get_original_size_info(image)
        
        # Preprocess image
        model_input, resized_image = self.image_processor.preprocess_for_visualization(image)
        model_input = model_input.to(self.device)
        
        # Run inference
        with torch.no_grad():
            results = self.model_loader.predict_with_confidence(model_input)
        
        # Extract results
        probabilities = results['probabilities'].cpu().numpy().squeeze()
        predictions = results['predictions'].cpu().numpy().squeeze()
        confidence_scores = results['confidence'].cpu().numpy().squeeze()
        
        # Calculate overall confidence (mean of individual class confidences)
        overall_confidence = float(np.mean(confidence_scores))
        
        # Create per-class results
        class_results = {}
        for i, class_name in enumerate(self.class_names):
            class_results[class_name] = {
                'probability': float(probabilities[i]),
                'prediction': bool(predictions[i]),
                'confidence': float(confidence_scores[i]),
                'color': self.class_colors[class_name]
            }
        
        return {
            'probabilities': probabilities,
            'predictions': predictions.astype(bool),
            'confidence': confidence_scores,
            'class_results': class_results,
            'overall_confidence': overall_confidence,
            'original_size': original_size,
            'processed_size': self.target_size,
            'resized_image': resized_image
        }
    
    def predict_batch(self, images: List[Union[str, np.ndarray]]) -> List[Dict]:
        """
        Perform inference on a batch of images.
        
        Args:
            images: List of input images (file paths or numpy arrays)
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")
            try:
                result = self.predict_single(image)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image {i}: {e}")
                results.append(None)
        
        return results
    
    def get_damage_confidence_map(self, image: Union[str, np.ndarray], method: str = 'sliding_window', target_class: str = 'damage') -> Tuple[np.ndarray, Dict]:
        """
        Generate a confidence map focused on damage detection.
        
        Args:
            image: Input image
            method: Method for generating spatial confidence 
                   ('sliding_window', 'patch_based', 'gradient', 'regional', 'edge_based', 'uniform')
            
        Returns:
            Tuple of (confidence_map, prediction_results)
        """
        # Get overall prediction results first
        results = self.predict_single(image)
        
        if method == 'sliding_window':
            # True pixel-by-pixel analysis using sliding windows
            confidence_map = self._create_sliding_window_heatmap(image)
            
        elif method == 'patch_based':
            # Patch-based analysis for faster processing
            confidence_map = self._create_patch_based_heatmap(image)
            
        elif method == 'simple_grid':
            # Simple grid-based analysis (most reliable)
            confidence_map = self._create_simple_grid_heatmap(image)
            
        elif method == 'gradcam':
            # Grad-CAM visualization showing where model looks for target class
            confidence_map = self._create_gradcam_heatmap(image, target_class=target_class)
            
        elif method == 'gradcam_all':
            # Multi-class Grad-CAM showing all classes
            confidence_map = self._create_multi_class_gradcam_heatmap(image)
            
        elif method == 'uniform':
            # Simple uniform confidence map
            damage_prob = results['class_results']['damage']['probability']
            confidence_map = np.full(self.target_size, damage_prob, dtype=np.float32)
            
        elif method == 'gradient':
            # Create gradient-based confidence map (center-focused)
            damage_prob = results['class_results']['damage']['probability']
            confidence_map = self._create_gradient_confidence_map(damage_prob)
            
        elif method == 'regional':
            # Regional analysis based on texture (artificial pattern)
            damage_prob = results['class_results']['damage']['probability']
            confidence_map = self._create_regional_confidence_map(image, damage_prob)
            
        elif method == 'edge_based':
            # Edge-based analysis (artificial pattern)
            damage_prob = results['class_results']['damage']['probability']
            confidence_map = self._create_edge_based_confidence_map(image, damage_prob)
            
        else:
            damage_prob = results['class_results']['damage']['probability']
            confidence_map = np.full(self.target_size, damage_prob, dtype=np.float32)
        
        return confidence_map, results
    
    def _create_simple_grid_heatmap(self, image: Union[str, np.ndarray], 
                                  grid_size: int = 8) -> np.ndarray:
        """
        Create a simple grid-based heatmap that definitely works.
        
        Args:
            image: Input image
            grid_size: Number of grid cells per dimension
            
        Returns:
            Grid-based confidence map
        """
        # Get image as numpy array
        if isinstance(image, str):
            img_pil = self.image_processor.load_image(image)
            img_array = np.array(img_pil)
        else:
            img_array = image
        
        # Resize to target size
        img_resized = cv2.resize(img_array, (self.target_size[1], self.target_size[0]))
        h, w = self.target_size
        
        # Initialize confidence map
        confidence_map = np.zeros((h, w), dtype=np.float32)
        
        # Calculate cell size
        cell_h = h // grid_size
        cell_w = w // grid_size
        
        logger.info(f"Creating {grid_size}x{grid_size} grid heatmap, cell size: {cell_h}x{cell_w}")
        
        # Process each grid cell
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate cell boundaries
                y1 = i * cell_h
                y2 = min((i + 1) * cell_h, h)
                x1 = j * cell_w
                x2 = min((j + 1) * cell_w, w)
                
                # Extract cell
                cell = img_resized[y1:y2, x1:x2]
                
                try:
                    # Get prediction for this cell
                    cell_results = self.predict_single(cell)
                    damage_prob = cell_results['class_results']['damage']['probability']
                    
                    # Fill cell area with this probability
                    confidence_map[y1:y2, x1:x2] = damage_prob
                    
                    logger.info(f"Grid cell ({i},{j}): damage_prob = {damage_prob:.3f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process grid cell ({i},{j}): {e}")
                    # Use a default value
                    confidence_map[y1:y2, x1:x2] = 0.5
        
        return confidence_map
    
    def _create_gradient_confidence_map(self, damage_prob: float) -> np.ndarray:
        """Create a gradient-based confidence map with center focus."""
        h, w = self.target_size
        y, x = np.ogrid[:h, :w]
        
        # Create distance from center
        center_y, center_x = h // 2, w // 2
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Normalize distance (0 at center, 1 at corners)
        normalized_dist = dist_from_center / max_dist
        
        # Create confidence map (higher in center, varies with damage probability)
        if damage_prob > 0.5:
            # For high damage probability, create center-focused pattern
            confidence_map = damage_prob * (1.2 - 0.4 * normalized_dist)
        else:
            # For low damage probability, create more uniform pattern
            confidence_map = damage_prob * (1.1 - 0.2 * normalized_dist)
        
        # Ensure values are in valid range
        confidence_map = np.clip(confidence_map, 0, 1)
        
        return confidence_map.astype(np.float32)
    
    def _create_regional_confidence_map(self, image: Union[str, np.ndarray], damage_prob: float, target_class: str = 'damage') -> np.ndarray:
        """Create regional confidence map based on image texture analysis."""
        # Get image as numpy array
        if isinstance(image, str):
            img_pil = self.image_processor.load_image(image)
            img_array = np.array(img_pil)
        else:
            img_array = image
        
        # Resize to target size
        img_resized = cv2.resize(img_array, (self.target_size[1], self.target_size[0]))
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        
        # Calculate local variance (texture indicator)
        kernel = np.ones((9, 9), np.float32) / 81
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        
        # Normalize variance
        variance_norm = (local_variance - local_variance.min()) / (local_variance.max() - local_variance.min() + 1e-8)
        
        # Create confidence map based on texture
        if damage_prob > 0.5:
            # High damage: focus on high-texture areas (potential damage)
            confidence_map = damage_prob * (0.3 + 0.7 * variance_norm)
        else:
            # Low damage: more uniform with slight texture influence
            confidence_map = damage_prob * (0.7 + 0.3 * variance_norm)
        
        return confidence_map.astype(np.float32)
    
    def _create_edge_based_confidence_map(self, image: Union[str, np.ndarray], damage_prob: float) -> np.ndarray:
        """Create edge-based confidence map focusing on road edges and features."""
        # Get image as numpy array
        if isinstance(image, str):
            img_pil = self.image_processor.load_image(image)
            img_array = np.array(img_pil)
        else:
            img_array = image
        
        # Resize to target size
        img_resized = cv2.resize(img_array, (self.target_size[1], self.target_size[0]))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to create regions
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Normalize edge map
        edge_norm = dilated_edges.astype(np.float32) / 255.0
        
        # Create confidence map
        if damage_prob > 0.5:
            # High damage: emphasize edge areas
            confidence_map = damage_prob * (0.4 + 0.6 * edge_norm)
        else:
            # Low damage: de-emphasize edge areas
            confidence_map = damage_prob * (0.8 - 0.3 * edge_norm)
        
        return confidence_map.astype(np.float32)
    
    def _create_sliding_window_heatmap(self, image: Union[str, np.ndarray], 
                                     window_size: int = 128, 
                                     stride: int = 32) -> np.ndarray:
        """
        Create true pixel-by-pixel heatmap using sliding window approach.
        
        Args:
            image: Input image
            window_size: Size of sliding window patches
            stride: Step size for sliding window
            
        Returns:
            Pixel-level confidence map
        """
        # Get image as numpy array
        if isinstance(image, str):
            img_pil = self.image_processor.load_image(image)
            img_array = np.array(img_pil)
        else:
            img_array = image
        
        # Resize to target size
        img_resized = cv2.resize(img_array, (self.target_size[1], self.target_size[0]))
        h, w = self.target_size
        
        # Initialize confidence map and count map for averaging
        confidence_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        logger.info(f"Starting sliding window analysis: {h}x{w} image, window={window_size}, stride={stride}")
        
        total_patches = ((h - window_size) // stride + 1) * ((w - window_size) // stride + 1)
        processed_patches = 0
        
        # Sliding window over the image
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                # Extract patch
                patch = img_resized[y:y+window_size, x:x+window_size]
                
                # Skip if patch is too small
                if patch.shape[0] < window_size or patch.shape[1] < window_size:
                    continue
                
                try:
                    # Get prediction for this patch using the inference engine
                    patch_results = self.predict_single(patch)
                    
                    # Extract damage probability
                    damage_prob = patch_results['class_results']['damage']['probability']
                    
                    # Add to confidence map
                    confidence_map[y:y+window_size, x:x+window_size] += damage_prob
                    count_map[y:y+window_size, x:x+window_size] += 1
                    
                    processed_patches += 1
                    if processed_patches % 10 == 0:
                        logger.info(f"Processed {processed_patches}/{total_patches} patches, current damage prob: {damage_prob:.3f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process patch at ({x}, {y}): {e}")
                    continue
        
        # Average overlapping predictions
        confidence_map = np.divide(confidence_map, count_map, 
                                 out=np.zeros_like(confidence_map), 
                                 where=count_map!=0)
        
        return confidence_map
    
    def _create_patch_based_heatmap(self, image: Union[str, np.ndarray], 
                                  patch_size: int = 64) -> np.ndarray:
        """
        Create patch-based heatmap for faster processing.
        
        Args:
            image: Input image
            patch_size: Size of non-overlapping patches
            
        Returns:
            Patch-level confidence map
        """
        # Get image as numpy array
        if isinstance(image, str):
            img_pil = self.image_processor.load_image(image)
            img_array = np.array(img_pil)
        else:
            img_array = image
        
        # Resize to target size
        img_resized = cv2.resize(img_array, (self.target_size[1], self.target_size[0]))
        h, w = self.target_size
        
        # Initialize confidence map
        confidence_map = np.zeros((h, w), dtype=np.float32)
        
        # Process non-overlapping patches
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                # Calculate patch boundaries
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                
                # Extract patch
                patch = img_resized[y:y_end, x:x_end]
                
                # Skip if patch is too small
                if patch.shape[0] < patch_size//2 or patch.shape[1] < patch_size//2:
                    continue
                
                try:
                    # Resize patch to model input size if needed
                    if patch.shape[:2] != (patch_size, patch_size):
                        patch = cv2.resize(patch, (patch_size, patch_size))
                    
                    # Get prediction for this patch using the inference engine
                    patch_results = self.predict_single(patch)
                    
                    # Extract damage probability
                    damage_prob = patch_results['class_results']['damage']['probability']
                    
                    # Fill patch area with this probability
                    confidence_map[y:y_end, x:x_end] = damage_prob
                    
                except Exception as e:
                    logger.warning(f"Failed to process patch at ({x}, {y}): {e}")
                    # Use overall image prediction as fallback
                    overall_results = self.predict_single(img_array)
                    damage_prob = overall_results['class_results']['damage']['probability']
                    confidence_map[y:y_end, x:x_end] = damage_prob
        
        return confidence_map
    
    def _create_gradcam_heatmap(self, image: Union[str, np.ndarray], target_class: str = 'damage') -> np.ndarray:
        """
        Create Grad-CAM heatmap showing where the model looks for the target class.
        
        Args:
            image: Input image
            target_class: Target class ('damage', 'occlusion', 'crop')
            
        Returns:
            Grad-CAM heatmap
        """
        # Get image as tensor
        if isinstance(image, str):
            img_pil = self.image_processor.load_image(image)
            img_array = np.array(img_pil)
        else:
            img_array = image
        
        # Preprocess for model
        input_tensor = self.image_processor.preprocess_for_model(img_array).to(self.device)
        input_tensor.requires_grad_(True)
        
        # Get class index
        class_idx = {'damage': 0, 'occlusion': 1, 'crop': 2}[target_class]
        
        # Hook to capture gradients and feature maps
        gradients = []
        feature_maps = []
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        def forward_hook(module, input, output):
            feature_maps.append(output)
        
        # Register hooks on the last convolutional layer (EfficientNet backbone)
        # Find the last conv layer in EfficientNet
        target_layer = None
        for name, module in self.model.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            logger.error("Could not find convolutional layer for Grad-CAM")
            return np.full(self.target_size, 0.5, dtype=np.float32)
        
        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        try:
            # Forward pass
            self.model.eval()
            output = self.model(input_tensor)
            
            # Get the score for target class
            class_score = output[0, class_idx]
            
            # Backward pass
            self.model.zero_grad()
            class_score.backward(retain_graph=True)
            
            # Get gradients and feature maps
            if not gradients or not feature_maps:
                logger.warning("No gradients or feature maps captured")
                return np.full(self.target_size, 0.5, dtype=np.float32)
            
            grads = gradients[0][0]  # [C, H, W]
            fmaps = feature_maps[0][0]  # [C, H, W]
            
            # Calculate alpha weights (Grad-CAM step 2)
            alphas = torch.mean(grads, dim=(1, 2))  # [C]
            
            # Calculate weighted combination (Grad-CAM step 3)
            gradcam = torch.zeros(fmaps.shape[1:], device=self.device)  # [H, W]
            for i, alpha in enumerate(alphas):
                gradcam += alpha * fmaps[i]
            
            # Apply ReLU
            gradcam = torch.relu(gradcam)
            
            # Normalize
            if gradcam.max() > 0:
                gradcam = gradcam / gradcam.max()
            
            # Convert to numpy and resize to target size
            gradcam_np = gradcam.detach().cpu().numpy()
            gradcam_resized = cv2.resize(gradcam_np, self.target_size[::-1])  # cv2 uses (width, height)
            
            logger.info(f"Grad-CAM for {target_class}: min={gradcam_resized.min():.3f}, max={gradcam_resized.max():.3f}")
            
            return gradcam_resized.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Grad-CAM failed: {e}")
            return np.full(self.target_size, 0.5, dtype=np.float32)
        
        finally:
            # Remove hooks
            forward_handle.remove()
            backward_handle.remove()
    
    def _create_multi_class_gradcam_heatmap(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Create multi-class Grad-CAM heatmap combining all classes.
        
        Args:
            image: Input image
            
        Returns:
            Combined Grad-CAM heatmap
        """
        # Get Grad-CAM for each class
        damage_cam = self._create_gradcam_heatmap(image, 'damage')
        occlusion_cam = self._create_gradcam_heatmap(image, 'occlusion')
        crop_cam = self._create_gradcam_heatmap(image, 'crop')
        
        # Get class probabilities for weighting
        results = self.predict_single(image)
        damage_prob = results['class_results']['damage']['probability']
        occlusion_prob = results['class_results']['occlusion']['probability']
        crop_prob = results['class_results']['crop']['probability']
        
        # Weight each Grad-CAM by its class probability
        combined_cam = (damage_prob * damage_cam + 
                       occlusion_prob * occlusion_cam + 
                       crop_prob * crop_cam)
        
        # Normalize
        if combined_cam.max() > 0:
            combined_cam = combined_cam / combined_cam.max()
        
        logger.info(f"Multi-class Grad-CAM: damage_weight={damage_prob:.3f}, "
                   f"occlusion_weight={occlusion_prob:.3f}, crop_weight={crop_prob:.3f}")
        
        return combined_cam.astype(np.float32)
    
    def analyze_image_regions(self, image: Union[str, np.ndarray], 
                            grid_size: Tuple[int, int] = (4, 4)) -> Dict:
        """
        Analyze image by dividing it into regions and getting predictions for each.
        
        Args:
            image: Input image
            grid_size: Size of the grid to divide image into (rows, cols)
            
        Returns:
            Dictionary with regional analysis results
        """
        # Load and preprocess image
        if isinstance(image, str):
            pil_image = self.image_processor.load_image(image)
        else:
            pil_image = self.image_processor.load_image(image) if isinstance(image, str) else None
        
        if pil_image is None:
            raise ValueError("Could not load image for regional analysis")
        
        # Resize to target size for consistent analysis
        resized_image = pil_image.resize(self.target_size)
        img_array = np.array(resized_image)
        
        rows, cols = grid_size
        h, w = self.target_size
        
        region_height = h // rows
        region_width = w // cols
        
        regional_results = []
        
        for i in range(rows):
            for j in range(cols):
                # Extract region
                y1 = i * region_height
                y2 = min((i + 1) * region_height, h)
                x1 = j * region_width  
                x2 = min((j + 1) * region_width, w)
                
                region = img_array[y1:y2, x1:x2]
                
                # Get prediction for region
                try:
                    region_result = self.predict_single(region)
                    region_result['grid_position'] = (i, j)
                    region_result['bbox'] = (x1, y1, x2, y2)
                    regional_results.append(region_result)
                except Exception as e:
                    logger.warning(f"Failed to analyze region ({i}, {j}): {e}")
        
        # Analyze overall results
        damage_regions = [r for r in regional_results if r['class_results']['damage']['prediction']]
        occlusion_regions = [r for r in regional_results if r['class_results']['occlusion']['prediction']]
        crop_regions = [r for r in regional_results if r['class_results']['crop']['prediction']]
        
        return {
            'grid_size': grid_size,
            'total_regions': len(regional_results),
            'regional_results': regional_results,
            'damage_regions': len(damage_regions),
            'occlusion_regions': len(occlusion_regions),
            'crop_regions': len(crop_regions),
            'damage_percentage': len(damage_regions) / len(regional_results) * 100,
            'occlusion_percentage': len(occlusion_regions) / len(regional_results) * 100,
            'crop_percentage': len(crop_regions) / len(regional_results) * 100
        }
    
    def get_summary_statistics(self, results: Dict) -> Dict[str, float]:
        """
        Get summary statistics from prediction results.
        
        Args:
            results: Prediction results dictionary
            
        Returns:
            Summary statistics
        """
        stats = {
            'overall_confidence': results['overall_confidence'],
            'max_probability': float(np.max(results['probabilities'])),
            'min_probability': float(np.min(results['probabilities'])),
            'mean_probability': float(np.mean(results['probabilities'])),
            'num_positive_predictions': int(np.sum(results['predictions'])),
            'damage_score': results['class_results']['damage']['probability'],
            'occlusion_score': results['class_results']['occlusion']['probability'],
            'crop_score': results['class_results']['crop']['probability']
        }
        
        return stats


def create_inference_engine(experiments_path: str = "../experiments/2025-07-05_hybrid_training") -> InferenceEngine:
    """
    Convenience function to create an inference engine with the best Model B.
    
    Args:
        experiments_path: Path to the experiments directory
        
    Returns:
        Configured InferenceEngine instance
    """
    checkpoint_path = Path(experiments_path) / "results" / "model_b" / "checkpoints" / "best_model.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    return InferenceEngine(str(checkpoint_path))