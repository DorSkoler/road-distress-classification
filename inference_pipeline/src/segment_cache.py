"""
Segment Cache for storing pre-generated segment visualizations.
This prevents UI duplication by pre-computing all segment data.
"""

import os
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class SegmentCache:
    """Cache for pre-generated segment visualizations to prevent UI duplication."""
    
    def __init__(self):
        """Initialize segment cache with temporary directory."""
        self.cache_dir = None
        self.cached_segments = {}
        self.is_initialized = False
        
    def initialize_cache(self, road_data: Dict) -> bool:
        """
        Initialize cache directory and prepare for segment caching.
        
        Args:
            road_data: Processed road data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create temporary directory for this session
            self.cache_dir = Path(tempfile.mkdtemp(prefix="road_segments_"))
            self.cached_segments = {}
            self.is_initialized = True
            
            logger.info(f"Initialized segment cache at {self.cache_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize segment cache: {e}")
            self.is_initialized = False
            return False
    
    def cache_segment_visualizations(self, road_data: Dict, uploaded_files) -> bool:
        """
        Pre-generate and cache all segment visualizations.
        
        Args:
            road_data: Processed road data dictionary
            uploaded_files: Original uploaded files for image access
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized:
            logger.error("Cache not initialized")
            return False
            
        try:
            total_segments = len(road_data['images'])
            logger.info(f"Pre-generating {total_segments} segment visualizations...")
            
            for i, image_info in enumerate(road_data['images']):
                logger.debug(f"Processing segment {i+1}/{total_segments}: {image_info['filename']}")
                
                # Find uploaded file
                uploaded_file = None
                for file in uploaded_files:
                    if file.name == image_info['filename']:
                        uploaded_file = file
                        break
                
                if not uploaded_file:
                    logger.warning(f"Could not find uploaded file: {image_info['filename']}")
                    continue
                
                # Generate segment visualization data
                start_time = time.time()
                segment_data = self._generate_segment_data(road_data, i, uploaded_file)
                generation_time = time.time() - start_time
                
                # Cache the data
                self.cached_segments[i] = segment_data
                logger.debug(f"Segment {i} cached in {generation_time:.2f}s")
            
            logger.info(f"Successfully cached {len(self.cached_segments)} segment visualizations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache segment visualizations: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _generate_segment_data(self, road_data: Dict, segment_idx: int, uploaded_file) -> Dict:
        """Generate all visualization data for a segment."""
        from PIL import Image
        import matplotlib.pyplot as plt
        import numpy as np
        
        try:
            image_info = road_data['images'][segment_idx]
            
            # Load original image
            original_image = Image.open(uploaded_file)
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            
            segment_data = {
                'image_info': image_info,
                'original_image': original_image,
                'gradcam_path': None,
                'mask_path': None,
                'predictions': {
                    'damage': road_data['predictions']['damage'][segment_idx],
                    'occlusion': road_data['predictions']['occlusion'][segment_idx],
                    'crop': road_data['predictions']['crop'][segment_idx]
                }
            }
            
            # Generate Grad-CAM visualization if available
            gradcam_info = None
            for gc_data in road_data['gradcam_data']:
                if gc_data['sequence'] == image_info['sequence']:
                    gradcam_info = gc_data
                    break
            
            if gradcam_info and gradcam_info['gradcam'] is not None:
                gradcam_path = self._save_gradcam_visualization(
                    original_image, gradcam_info['gradcam'], segment_idx
                )
                segment_data['gradcam_path'] = gradcam_path
            
            # Generate mask visualization if available
            if 'mask_data' in road_data and segment_idx < len(road_data['mask_data']):
                mask_info = road_data['mask_data'][segment_idx]
                mask_data = mask_info.get('mask')
                mask_overlay = mask_info.get('mask_overlay')
                mask_stats = mask_info.get('mask_stats', {})
                
                if mask_overlay is not None:
                    # Save mask overlay directly
                    mask_path = self.cache_dir / f"mask_{segment_idx}.png"
                    Image.fromarray(mask_overlay).save(mask_path)
                    segment_data['mask_path'] = mask_path
                elif mask_data is not None:
                    # Generate mask visualization
                    mask_path = self._save_mask_visualization(
                        original_image, mask_data, segment_idx
                    )
                    segment_data['mask_path'] = mask_path
                
                segment_data['mask_stats'] = mask_stats
            
            return segment_data
            
        except Exception as e:
            logger.error(f"Error generating segment {segment_idx} data: {e}")
            return {'error': str(e)}
    
    def _save_gradcam_visualization(self, original_image: Image.Image, gradcam_data: np.ndarray, 
                                   segment_idx: int) -> Path:
        """Save Grad-CAM visualization to cache."""
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(np.array(original_image))
            ax.imshow(gradcam_data, alpha=0.6, cmap='jet')
            ax.set_title('Damage Attention Map', fontsize=10)
            ax.axis('off')
            
            gradcam_path = self.cache_dir / f"gradcam_{segment_idx}.png"
            fig.savefig(gradcam_path, bbox_inches='tight', dpi=100)
            plt.close(fig)
            
            return gradcam_path
            
        except Exception as e:
            logger.error(f"Error saving Grad-CAM for segment {segment_idx}: {e}")
            return None
    
    def _save_mask_visualization(self, original_image: Image.Image, mask_data: np.ndarray, 
                                segment_idx: int) -> Path:
        """Save mask visualization to cache."""
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(np.array(original_image))
            ax.imshow(mask_data, alpha=0.4, cmap='Greens')
            ax.set_title('Road Segmentation', fontsize=10)
            ax.axis('off')
            
            mask_path = self.cache_dir / f"mask_viz_{segment_idx}.png"
            fig.savefig(mask_path, bbox_inches='tight', dpi=100)
            plt.close(fig)
            
            return mask_path
            
        except Exception as e:
            logger.error(f"Error saving mask visualization for segment {segment_idx}: {e}")
            return None
    
    def get_segment_data(self, segment_idx: int) -> Optional[Dict]:
        """Get cached segment data by index."""
        return self.cached_segments.get(segment_idx)
    
    def get_available_segments(self) -> List[int]:
        """Get list of available cached segment indices."""
        return list(self.cached_segments.keys())
    
    def cleanup(self):
        """Clean up temporary cache directory."""
        if self.cache_dir and self.cache_dir.exists():
            try:
                shutil.rmtree(self.cache_dir)
                logger.info("Cleaned up segment cache")
            except Exception as e:
                logger.error(f"Error cleaning up cache: {e}")
        
        self.cached_segments = {}
        self.is_initialized = False
