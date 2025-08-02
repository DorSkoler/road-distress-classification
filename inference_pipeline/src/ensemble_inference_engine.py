"""
Ensemble Inference Engine for Multi-Model Road Distress Classification
Combines predictions from Model B and Model H for improved accuracy.
"""

import torch
import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Union, Optional
from PIL import Image

from .multi_model_loader import MultiModelLoader
from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)

class EnsembleInferenceEngine:
    """Ensemble inference engine combining multiple models."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ensemble inference engine.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.multi_loader = MultiModelLoader(config_path)
        self.image_processor = ImageProcessor()
        self.models = {}
        self.class_names = []
        self.thresholds = {}
        
        # Load models and configuration
        self._load_models()
        self._load_thresholds()
        
    def _load_models(self):
        """Load all available models."""
        try:
            self.models = self.multi_loader.load_models()
            self.class_names = self.multi_loader.get_class_names()
            logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _load_thresholds(self):
        """Load per-class thresholds from config."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.thresholds = config.get('inference', {}).get('thresholds', {
                'damage': 0.5,
                'occlusion': 0.5,
                'crop': 0.5
            })
            logger.info(f"Loaded thresholds: {self.thresholds}")
        except Exception as e:
            logger.warning(f"Failed to load thresholds, using defaults: {e}")
            self.thresholds = {'damage': 0.5, 'occlusion': 0.5, 'crop': 0.5}
    
    def update_thresholds(self, thresholds: Dict[str, float]):
        """Update per-class thresholds."""
        self.thresholds.update(thresholds)
        logger.info(f"Updated thresholds: {self.thresholds}")
    
    def predict_single_model(self, image: Union[str, np.ndarray], model_name: str) -> Dict:
        """
        Run inference on a single model.
        
        Args:
            image: Input image (path or numpy array)
            model_name: Name of the model to use
            
        Returns:
            Dictionary containing predictions and metadata
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Loaded models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        model_info = self.multi_loader.get_model_info(model_name)
        
        # Process image for both model and visualization
        processed_image, resized_image = self.image_processor.preprocess_for_visualization(image)
        
        # Get original image
        if isinstance(image, str):
            original_image = self.image_processor.load_image(image)
        elif isinstance(image, np.ndarray):
            original_image = Image.fromarray(image.astype(np.uint8))
        else:
            original_image = image
        
        # Run inference
        model.eval()
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            # Forward pass (ensure input is on same device as model)
            processed_image = processed_image.to(next(model.parameters()).device)
            logits = model(processed_image)
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time) / 1000.0
            else:
                inference_time = 0.0
        
        # Apply thresholds
        predictions = [prob >= self.thresholds[class_name] 
                      for prob, class_name in zip(probabilities, self.class_names)]
        
        # Calculate confidence scores
        confidences = [prob if pred else 1 - prob 
                      for prob, pred in zip(probabilities, predictions)]
        
        # Create class results
        class_results = {}
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # Red, Green, Blue
        
        for i, class_name in enumerate(self.class_names):
            class_results[class_name] = {
                'probability': float(probabilities[i]),
                'prediction': bool(predictions[i]),
                'confidence': float(confidences[i]),
                'threshold': float(self.thresholds[class_name]),
                'color': colors[i]
            }
        
        return {
            'model_name': model_name,
            'model_info': model_info,
            'probabilities': probabilities.tolist(),
            'predictions': predictions,
            'class_results': class_results,
            'overall_confidence': float(np.mean(confidences)),
            'inference_time': inference_time,
            'original_image': original_image,
            'resized_image': resized_image
        }
    
    def predict_ensemble(self, image: Union[str, np.ndarray], 
                        model_names: Optional[List[str]] = None,
                        weights: Optional[List[float]] = None) -> Dict:
        """
        Run ensemble inference combining multiple models.
        
        Args:
            image: Input image (path or numpy array)
            model_names: List of models to use. If None, uses all loaded models.
            weights: Weights for ensemble averaging. If None, uses equal weights.
            
        Returns:
            Dictionary containing ensemble predictions and individual model results
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        if weights is None:
            weights = [1.0 / len(model_names)] * len(model_names)
        
        if len(weights) != len(model_names):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(model_names)})")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Get predictions from each model
        individual_results = {}
        all_probabilities = []
        
        for model_name in model_names:
            result = self.predict_single_model(image, model_name)
            individual_results[model_name] = result
            all_probabilities.append(result['probabilities'])
        
        # Ensemble averaging
        ensemble_probabilities = np.average(all_probabilities, axis=0, weights=weights)
        
        # Apply thresholds to ensemble
        ensemble_predictions = [prob >= self.thresholds[class_name] 
                              for prob, class_name in zip(ensemble_probabilities, self.class_names)]
        
        # Calculate ensemble confidence scores
        ensemble_confidences = [prob if pred else 1 - prob 
                              for prob, pred in zip(ensemble_probabilities, ensemble_predictions)]
        
        # Create ensemble class results
        ensemble_class_results = {}
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # Red, Green, Blue
        
        for i, class_name in enumerate(self.class_names):
            ensemble_class_results[class_name] = {
                'probability': float(ensemble_probabilities[i]),
                'prediction': bool(ensemble_predictions[i]),
                'confidence': float(ensemble_confidences[i]),
                'threshold': float(self.thresholds[class_name]),
                'color': colors[i]
            }
        
        # Get image info from first model result
        first_result = list(individual_results.values())[0]
        
        return {
            'ensemble_results': {
                'probabilities': ensemble_probabilities.tolist(),
                'predictions': ensemble_predictions,
                'class_results': ensemble_class_results,
                'overall_confidence': float(np.mean(ensemble_confidences)),
                'weights_used': weights.tolist(),
                'models_used': model_names
            },
            'individual_results': individual_results,
            'original_image': first_result['original_image'],
            'resized_image': first_result['resized_image']
        }
    
    def get_damage_confidence_map(self, image: Union[str, np.ndarray], 
                                 method: str = 'gradcam', 
                                 target_class: str = 'damage',
                                 model_name: str = 'model_b') -> Tuple[np.ndarray, Dict]:
        """
        Generate confidence map using Grad-CAM for a specific model or combined models.
        
        Args:
            image: Input image
            method: Visualization method ('gradcam' or 'gradcam_all')
            target_class: Target class for single-class Grad-CAM
            model_name: Which model to use for Grad-CAM ('model_b', 'model_h', or 'combined')
            
        Returns:
            Tuple of (confidence_map, metadata)
        """
        if model_name == 'combined':
            return self._get_combined_gradcam(image, method, target_class)
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Loaded models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Get image as numpy array
        if isinstance(image, str):
            img_pil = self.image_processor.load_image(image)
            img_array = np.array(img_pil)
        else:
            img_array = image
        
        # Resize to target size
        target_size = (256, 256)  # height, width
        img_resized = cv2.resize(img_array, (target_size[1], target_size[0]))
        
        if method == 'gradcam':
            confidence_map = self._create_gradcam_heatmap(img_resized, target_class, model)
        elif method == 'gradcam_all':
            confidence_map = self._create_multi_class_gradcam_heatmap(img_resized, model)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        metadata = {
            'method': method,
            'target_class': target_class,
            'model_name': model_name,
            'shape': confidence_map.shape
        }
        
        return confidence_map, metadata
    
    def _create_gradcam_heatmap(self, image: np.ndarray, target_class: str, model: torch.nn.Module) -> np.ndarray:
        """Create Grad-CAM heatmap for a specific class."""
        # Convert to tensor and preprocess
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.unsqueeze(0).to(next(model.parameters()).device)
        
        # Get target class index
        class_idx = self.class_names.index(target_class)
        
        # Variables to store gradients and feature maps
        gradients = None
        feature_maps = None
        
        def save_gradients(grad):
            nonlocal gradients
            gradients = grad
        
        def save_feature_maps(module, input, output):
            nonlocal feature_maps
            feature_maps = output
        
        # Register hooks on the last convolutional layer (EfficientNet backbone)
        target_layer = None
        for name, module in model.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            raise RuntimeError("Could not find convolutional layer for Grad-CAM")
        
        # Register hooks
        hook_grad = target_layer.register_backward_hook(lambda module, grad_in, grad_out: save_gradients(grad_out[0]))
        hook_fmap = target_layer.register_forward_hook(save_feature_maps)
        
        model.eval()
        try:
            # Forward pass
            output = model(img_tensor)
            
            # Get the score for target class
            target_score = output[0, class_idx]
            
            # Backward pass
            model.zero_grad()
            target_score.backward()
            
            # Generate Grad-CAM
            if gradients is not None and feature_maps is not None:
                # Global average pooling of gradients
                alphas = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
                
                # Weighted combination of feature maps
                gradcam = torch.sum(alphas * feature_maps, dim=1).squeeze(0)  # [H, W]
                
                # Apply ReLU
                gradcam = torch.relu(gradcam)
                
                # Normalize to [0, 1]
                if gradcam.max() > 0:
                    gradcam = gradcam / gradcam.max()
                
                # Resize to input image size
                gradcam_resized = torch.nn.functional.interpolate(
                    gradcam.unsqueeze(0).unsqueeze(0),
                    size=(image.shape[0], image.shape[1]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().detach().cpu().numpy()
                
                return gradcam_resized
            else:
                # Fallback: return uniform map
                return np.ones((image.shape[0], image.shape[1]), dtype=np.float32) * 0.5
                
        finally:
            # Remove hooks
            hook_grad.remove()
            hook_fmap.remove()
    
    def _create_multi_class_gradcam_heatmap(self, image: np.ndarray, model: torch.nn.Module) -> np.ndarray:
        """Create multi-class Grad-CAM heatmap."""
        # Get individual class heatmaps
        class_heatmaps = []
        class_probs = []
        
        # Get model prediction for weighting
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.unsqueeze(0).to(next(model.parameters()).device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.sigmoid(output).cpu().numpy().flatten()
        
        # Generate heatmap for each class
        for i, class_name in enumerate(self.class_names):
            heatmap = self._create_gradcam_heatmap(image, class_name, model)
            class_heatmaps.append(heatmap)
            class_probs.append(probs[i])
        
        # Combine heatmaps weighted by probabilities
        combined_heatmap = np.zeros_like(class_heatmaps[0])
        total_weight = sum(class_probs)
        
        if total_weight > 0:
            for heatmap, prob in zip(class_heatmaps, class_probs):
                combined_heatmap += heatmap * (prob / total_weight)
        else:
            # Fallback: equal weighting
            for heatmap in class_heatmaps:
                combined_heatmap += heatmap / len(class_heatmaps)
        
        return combined_heatmap
    
    def get_model_names(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a specific model."""
        return self.multi_loader.get_model_info(model_name)
    
    def _get_combined_gradcam(self, image: Union[str, np.ndarray], 
                             method: str = 'gradcam', 
                             target_class: str = 'damage') -> Tuple[np.ndarray, Dict]:
        """
        Generate combined Grad-CAM from both models.
        
        Args:
            image: Input image
            method: Visualization method
            target_class: Target class for Grad-CAM
            
        Returns:
            Tuple of (combined_confidence_map, metadata)
        """
        # Get Grad-CAM from both models
        gradcam_b, meta_b = self.get_damage_confidence_map(image, method, target_class, 'model_b')
        gradcam_h, meta_h = self.get_damage_confidence_map(image, method, target_class, 'model_h')
        
        # Combine using weighted average (same weights as ensemble predictions)
        weights = self.multi_loader.config.get('ensemble', {}).get('default_weights', [0.5, 0.5])
        weight_b, weight_h = weights[0], weights[1]
        
        combined_gradcam = (weight_b * gradcam_b + weight_h * gradcam_h)
        
        # Normalize to [0, 1] range
        combined_gradcam = (combined_gradcam - combined_gradcam.min()) / (combined_gradcam.max() - combined_gradcam.min() + 1e-8)
        
        metadata = {
            'method': method,
            'target_class': target_class,
            'model_name': 'combined',
            'shape': combined_gradcam.shape,
            'weights': {'model_b': weight_b, 'model_h': weight_h},
            'individual_metadata': {'model_b': meta_b, 'model_h': meta_h}
        }
        
        return combined_gradcam, metadata
    
    def get_all_model_info(self) -> Dict[str, Dict]:
        """Get information about all loaded models."""
        return self.multi_loader.get_all_model_info()