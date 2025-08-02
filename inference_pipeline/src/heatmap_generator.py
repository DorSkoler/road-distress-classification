#!/usr/bin/env python3
"""
Heatmap Generator for Road Distress Visualization
Date: 2025-08-01

This module generates confidence heatmaps and visualizations for road distress
classification results, with a focus on damage confidence visualization.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Tuple, Union, Optional, List
import logging

logger = logging.getLogger(__name__)


class HeatmapGenerator:
    """
    Generates confidence heatmaps and visualizations for road distress results.
    
    Focuses on creating intuitive visualizations that highlight areas of
    predicted road damage with confidence-based coloring.
    """
    
    def __init__(self, 
                 colormap: str = 'hot',
                 alpha: float = 0.6,
                 font_size: int = 16):
        """
        Initialize the heatmap generator.
        
        Args:
            colormap: Matplotlib colormap name for heatmaps
            alpha: Transparency for overlay heatmaps (0.0 - 1.0)
            font_size: Font size for text annotations
        """
        self.colormap = colormap
        self.alpha = alpha
        self.font_size = font_size
        
        # Create custom colormaps
        self.damage_cmap = LinearSegmentedColormap.from_list(
            'damage', ['blue', 'green', 'yellow', 'red'], N=256
        )
        
        self.confidence_cmap = LinearSegmentedColormap.from_list(
            'confidence', ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red'], N=256
        )
        
        logger.info(f"HeatmapGenerator initialized:")
        logger.info(f"  - Colormap: {colormap}")
        logger.info(f"  - Alpha: {alpha}")
        logger.info(f"  - Font size: {font_size}")
    
    def create_damage_confidence_heatmap(self, 
                                       image: Union[np.ndarray, Image.Image],
                                       confidence_map: np.ndarray,
                                       prediction_results: Dict,
                                       title: Optional[str] = None,
                                       add_text_overlay: bool = False) -> np.ndarray:
        """
        Create a damage confidence heatmap overlay on the original image.
        
        Args:
            image: Original image (numpy array or PIL Image)
            confidence_map: Confidence map array [H, W]
            prediction_results: Prediction results dictionary
            title: Optional title for the visualization
            
        Returns:
            Combined heatmap visualization as numpy array
        """
        # Convert image to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Ensure image is RGB
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_rgb = img_array
        else:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Resize confidence map to match image size if needed
        if confidence_map.shape != img_rgb.shape[:2]:
            confidence_map = cv2.resize(confidence_map, (img_rgb.shape[1], img_rgb.shape[0]))
        
        # Normalize confidence map to [0, 1]
        conf_normalized = (confidence_map - confidence_map.min()) / (confidence_map.max() - confidence_map.min() + 1e-8)
        
        # Create heatmap using custom damage colormap
        heatmap = self.damage_cmap(conf_normalized)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Blend original image with heatmap
        blended = cv2.addWeighted(img_rgb, 1 - self.alpha, heatmap, self.alpha, 0)
        
        # Add text annotations only if requested
        if add_text_overlay:
            annotated = self._add_prediction_annotations(blended, prediction_results, title)
            return annotated
        
        return blended
    
    def create_clean_heatmap(self, 
                           image: Union[np.ndarray, Image.Image],
                           confidence_map: np.ndarray,
                           scale_factor: float = 1.0) -> np.ndarray:
        """
        Create a clean damage confidence heatmap without any text overlays.
        
        Args:
            image: Original image
            confidence_map: Confidence map array [H, W]
            scale_factor: Scale factor for output image
            
        Returns:
            Clean heatmap visualization
        """
        # Convert image to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Ensure RGB format
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_rgb = img_array
        else:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Resize confidence map to match image size if needed
        if confidence_map.shape != img_rgb.shape[:2]:
            confidence_map = cv2.resize(confidence_map, (img_rgb.shape[1], img_rgb.shape[0]))
        
        # Normalize confidence map to [0, 1]
        conf_normalized = (confidence_map - confidence_map.min()) / (confidence_map.max() - confidence_map.min() + 1e-8)
        
        # Create heatmap using custom damage colormap
        heatmap = self.damage_cmap(conf_normalized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Blend original image with heatmap
        blended = cv2.addWeighted(img_rgb, 1 - self.alpha, heatmap, self.alpha, 0)
        
        # Apply scaling if requested
        if scale_factor != 1.0:
            new_height = int(blended.shape[0] * scale_factor)
            new_width = int(blended.shape[1] * scale_factor)
            blended = cv2.resize(blended, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        return blended
    
    def create_pure_confidence_map(self, confidence_map: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
        """
        Create a pure confidence map visualization without original image.
        
        Args:
            confidence_map: Confidence map array [H, W]
            scale_factor: Scale factor for output image
            
        Returns:
            Pure confidence heatmap
        """
        # Normalize confidence map
        conf_normalized = (confidence_map - confidence_map.min()) / (confidence_map.max() - confidence_map.min() + 1e-8)
        
        # Apply colormap
        heatmap = self.damage_cmap(conf_normalized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Apply scaling if requested
        if scale_factor != 1.0:
            new_height = int(heatmap.shape[0] * scale_factor)
            new_width = int(heatmap.shape[1] * scale_factor)
            heatmap = cv2.resize(heatmap, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        return heatmap
    
    def create_regional_heatmap(self, 
                              image: Union[np.ndarray, Image.Image],
                              regional_results: Dict,
                              focus_class: str = 'damage') -> np.ndarray:
        """
        Create a heatmap based on regional analysis results.
        
        Args:
            image: Original image
            regional_results: Results from regional analysis
            focus_class: Class to focus on ('damage', 'occlusion', 'crop')
            
        Returns:
            Regional heatmap visualization
        """
        # Convert image to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Ensure RGB format
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_rgb = img_array
        else:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Create confidence overlay
        overlay = img_rgb.copy()
        
        # Process each region
        for region_result in regional_results['regional_results']:
            bbox = region_result['bbox']
            x1, y1, x2, y2 = bbox
            
            # Get confidence for focus class
            confidence = region_result['class_results'][focus_class]['probability']
            prediction = region_result['class_results'][focus_class]['prediction']
            
            # Create color based on confidence
            color_intensity = int(confidence * 255)
            if prediction:
                # Strong color for positive predictions
                color = (255, 255 - color_intensity, 255 - color_intensity)  # Red gradient
            else:
                # Blue gradient for negative predictions
                color = (255 - color_intensity, 255 - color_intensity, 255)
            
            # Draw rectangle with transparency
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # Add confidence text
            conf_text = f"{confidence:.2f}"
            cv2.putText(overlay, conf_text, (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Blend with original
        result = cv2.addWeighted(img_rgb, 1 - self.alpha, overlay, self.alpha, 0)
        
        # Add summary information
        summary_text = [
            f"Grid: {regional_results['grid_size'][0]}x{regional_results['grid_size'][1]}",
            f"{focus_class.capitalize()}: {regional_results[f'{focus_class}_percentage']:.1f}%",
            f"Regions: {regional_results[f'{focus_class}_regions']}/{regional_results['total_regions']}"
        ]
        
        result = self._add_text_overlay(result, summary_text, position='top_left')
        
        return result
    
    def create_multi_class_visualization(self, 
                                       image: Union[np.ndarray, Image.Image],
                                       prediction_results: Dict) -> np.ndarray:
        """
        Create a visualization showing all class predictions.
        
        Args:
            image: Original image
            prediction_results: Prediction results dictionary
            
        Returns:
            Multi-class visualization
        """
        # Convert image to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Ensure RGB format
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_rgb = img_array
        else:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Create visualization with colored borders for each predicted class
        result = img_rgb.copy()
        border_thickness = 10
        
        class_colors = {
            'damage': (255, 0, 0),      # Red
            'occlusion': (0, 255, 0),   # Green
            'crop': (0, 0, 255)         # Blue
        }
        
        # Add colored borders for positive predictions
        height, width = result.shape[:2]
        border_offset = 0
        
        for class_name, color in class_colors.items():
            if prediction_results['class_results'][class_name]['prediction']:
                # Draw colored border
                cv2.rectangle(result, 
                            (border_offset, border_offset), 
                            (width - border_offset - 1, height - border_offset - 1),
                            color, border_thickness)
                border_offset += border_thickness
        
        # Add prediction information
        info_text = []
        for class_name in ['damage', 'occlusion', 'crop']:
            class_result = prediction_results['class_results'][class_name]
            status = "✓" if class_result['prediction'] else "✗"
            info_text.append(f"{status} {class_name.capitalize()}: {class_result['probability']:.3f}")
        
        info_text.append(f"Overall Confidence: {prediction_results['overall_confidence']:.3f}")
        
        result = self._add_text_overlay(result, info_text, position='bottom_left')
        
        return result
    
    def create_comparison_grid(self, 
                             original_image: Union[np.ndarray, Image.Image],
                             prediction_results: Dict,
                             confidence_map: np.ndarray) -> np.ndarray:
        """
        Create a comparison grid showing original, heatmap, and combined views.
        
        Args:
            original_image: Original input image
            prediction_results: Prediction results
            confidence_map: Confidence map
            
        Returns:
            Grid comparison visualization
        """
        # Convert image to numpy array if needed
        if isinstance(original_image, Image.Image):
            img_array = np.array(original_image)
        else:
            img_array = original_image.copy()
        
        # Ensure RGB format
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_rgb = img_array
        else:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Create different visualizations
        original = img_rgb.copy()
        heatmap_only = self._create_pure_heatmap(confidence_map)
        combined = self.create_damage_confidence_heatmap(img_rgb, confidence_map, prediction_results)
        multi_class = self.create_multi_class_visualization(img_rgb, prediction_results)
        
        # Resize all to same size
        target_size = (256, 256)
        original = cv2.resize(original, target_size)
        heatmap_only = cv2.resize(heatmap_only, target_size)
        combined = cv2.resize(combined, target_size)
        multi_class = cv2.resize(multi_class, target_size)
        
        # Add titles
        original = self._add_title(original, "Original")
        heatmap_only = self._add_title(heatmap_only, "Damage Heatmap")
        combined = self._add_title(combined, "Combined View")
        multi_class = self._add_title(multi_class, "Multi-Class")
        
        # Create 2x2 grid
        top_row = np.hstack([original, heatmap_only])
        bottom_row = np.hstack([combined, multi_class])
        grid = np.vstack([top_row, bottom_row])
        
        return grid
    
    def _create_pure_heatmap(self, confidence_map: np.ndarray) -> np.ndarray:
        """Create a pure heatmap without image overlay."""
        # Normalize confidence map
        conf_normalized = (confidence_map - confidence_map.min()) / (confidence_map.max() - confidence_map.min() + 1e-8)
        
        # Apply colormap
        heatmap = self.damage_cmap(conf_normalized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        
        return heatmap
    
    def _add_prediction_annotations(self, 
                                  image: np.ndarray, 
                                  prediction_results: Dict,
                                  title: Optional[str] = None) -> np.ndarray:
        """Add prediction annotations to image."""
        result = image.copy()
        
        # Prepare text information
        text_lines = []
        if title:
            text_lines.append(title)
        
        # Add damage information (primary focus)
        damage_result = prediction_results['class_results']['damage']
        damage_status = "DAMAGE DETECTED" if damage_result['prediction'] else "NO DAMAGE"
        text_lines.append(f"{damage_status} ({damage_result['probability']:.3f})")
        
        # Add other classes if predicted
        for class_name in ['occlusion', 'crop']:
            class_result = prediction_results['class_results'][class_name]
            if class_result['prediction']:
                text_lines.append(f"{class_name.upper()}: {class_result['probability']:.3f}")
        
        text_lines.append(f"Confidence: {prediction_results['overall_confidence']:.3f}")
        
        return self._add_text_overlay(result, text_lines, position='top_right')
    
    def _add_text_overlay(self, 
                         image: np.ndarray, 
                         text_lines: List[str],
                         position: str = 'top_left') -> np.ndarray:
        """Add text overlay to image."""
        result = image.copy()
        height, width = result.shape[:2]
        
        # Define positions
        if position == 'top_left':
            start_x, start_y = 10, 30
        elif position == 'top_right':
            start_x, start_y = width - 200, 30
        elif position == 'bottom_left':
            start_x, start_y = 10, height - len(text_lines) * 25 - 10
        else:  # bottom_right
            start_x, start_y = width - 200, height - len(text_lines) * 25 - 10
        
        # Add background rectangle
        bg_height = len(text_lines) * 25 + 10
        bg_width = 190
        cv2.rectangle(result, 
                     (start_x - 5, start_y - 20), 
                     (start_x + bg_width, start_y + bg_height - 20),
                     (0, 0, 0), -1)
        cv2.rectangle(result, 
                     (start_x - 5, start_y - 20), 
                     (start_x + bg_width, start_y + bg_height - 20),
                     (255, 255, 255), 2)
        
        # Add text
        for i, line in enumerate(text_lines):
            y_pos = start_y + i * 25
            cv2.putText(result, line, (start_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result
    
    def _add_title(self, image: np.ndarray, title: str) -> np.ndarray:
        """Add title to image."""
        result = image.copy()
        height, width = result.shape[:2]
        
        # Add title background
        cv2.rectangle(result, (0, 0), (width, 30), (0, 0, 0), -1)
        
        # Add title text
        cv2.putText(result, title, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result
    
    def save_visualization(self, 
                          visualization: np.ndarray, 
                          save_path: str,
                          dpi: int = 150) -> None:
        """
        Save visualization to file.
        
        Args:
            visualization: Visualization array
            save_path: Path to save the image
            dpi: DPI for saved image
        """
        # Convert BGR to RGB if needed
        if len(visualization.shape) == 3:
            # Assume it's already RGB from our processing
            save_array = visualization
        else:
            save_array = visualization
        
        # Save using PIL for better quality control
        pil_image = Image.fromarray(save_array.astype(np.uint8))
        pil_image.save(save_path, dpi=(dpi, dpi), quality=95)
        
        logger.info(f"Visualization saved to: {save_path}")