"""
Simple segment display system that avoids complex caching and matplotlib issues.
Uses direct image processing and streamlit components for better performance.
"""

import numpy as np
from PIL import Image
import cv2
import logging
from typing import Dict, Optional

# Set up detailed logging for debugging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SimpleSegmentDisplay:
    """Simple, fast segment display without complex caching or matplotlib."""
    
    def __init__(self):
        """Initialize simple segment display."""
        self.processed_segments = {}
        
    def display_segment_details(self, road_data: Dict, segment_idx: int, uploaded_files):
        """
        Display segment details using simple, direct approach with fragment optimization.
        
        Args:
            road_data: Processed road data dictionary
            segment_idx: Index of the segment to display
            uploaded_files: Original uploaded files for image access
        """
        import streamlit as st
        
        logger.info(f"Displaying segment {segment_idx} using simple approach")
        
        # Check if this segment was already processed to avoid unnecessary work
        segment_key = f"segment_{segment_idx}_{road_data['images'][segment_idx]['sequence']}"
        if segment_key in self.processed_segments:
            logger.debug(f"Using cached processing for {segment_key}")
        else:
            logger.debug(f"Processing new segment {segment_key}")
            self.processed_segments[segment_key] = True
        
        if segment_idx >= len(road_data['images']):
            st.error("Invalid segment index")
            return
        
        # Get segment data
        image_info = road_data['images'][segment_idx]
        
        # Find uploaded file
        uploaded_file = None
        for file in uploaded_files:
            if file.name == image_info['filename']:
                uploaded_file = file
                break
        
        if not uploaded_file:
            st.error(f"Could not find uploaded file: {image_info['filename']}")
            return
        
        # Create a unique key for this segment to prevent UI conflicts
        segment_key = f"segment_{segment_idx}_{image_info['sequence']}"
        
        # Direct display without nested containers to avoid rerun triggers
        st.subheader(f"🔍 Segment {image_info['sequence']:03d} Details")
        
        # Load and display images
        self._display_images(road_data, segment_idx, uploaded_file, segment_key)
        
        # Display predictions
        self._display_predictions(road_data, segment_idx, segment_key)
        
        # Display coordinates
        self._display_coordinates(image_info, segment_key)
    
    def _display_images(self, road_data: Dict, segment_idx: int, uploaded_file, segment_key: str):
        """Display the three main images for a segment with caching optimization."""
        import streamlit as st
        from PIL import Image
        from io import BytesIO
        
        # Cache image loading to avoid reloading on UI updates
        @st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
        def load_segment_image(file_name: str, file_content: bytes):
            """Load and preprocess segment image with caching."""
            image = Image.open(BytesIO(file_content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        
        # Load original image with caching
        original_image = load_segment_image(uploaded_file.name, uploaded_file.getvalue())
        
        # Create columns
        col1, col2, col3 = st.columns(3)
        
        # Original image
        with col1:
            st.markdown("**Original Image**")
            st.image(original_image, 
                    caption=f"Segment {road_data['images'][segment_idx]['sequence']:03d}", 
                    use_container_width=True)
        
        # Grad-CAM
        with col2:
            st.markdown("**Grad-CAM Visualization**")
            gradcam_image = self._get_gradcam_image(road_data, segment_idx, original_image)
            if gradcam_image is not None:
                st.image(gradcam_image, 
                        caption="Damage Attention Map",
                        use_container_width=True)
            else:
                st.info("Grad-CAM data not available for this segment")
        
        # Road mask
        with col3:
            st.markdown("**Road Mask**")
            mask_image, mask_stats = self._get_mask_image(road_data, segment_idx, original_image)
            if mask_image is not None:
                st.image(mask_image, 
                        caption="Road Segmentation",
                        use_container_width=True)
                
                # Display coverage stats
                if mask_stats:
                    coverage = mask_stats.get('road_coverage', 0) * 100
                    st.caption(f"🛣️ Road Coverage: {coverage:.1f}%")
            else:
                st.info("Road mask not available for this segment")
    
    def _get_gradcam_image(self, road_data: Dict, segment_idx: int, original_image: Image.Image) -> Optional[np.ndarray]:
        """Get Grad-CAM visualization as image array."""
        try:
            image_info = road_data['images'][segment_idx]
            
            # Find corresponding grad-CAM data
            for gc_data in road_data['gradcam_data']:
                if gc_data['sequence'] == image_info['sequence']:
                    gradcam_data = gc_data['gradcam']
                    if gradcam_data is not None:
                        # Create simple overlay
                        return self._create_gradcam_overlay(original_image, gradcam_data)
                    break
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating Grad-CAM image: {e}")
            return None
    
    def _get_mask_image(self, road_data: Dict, segment_idx: int, original_image: Image.Image) -> tuple:
        """Get mask visualization as image array and stats."""
        try:
            logger.debug(f"🔍 Checking mask data for segment {segment_idx}")
            
            if 'mask_data' not in road_data:
                logger.warning(f"❌ No mask_data key in road_data. Available keys: {list(road_data.keys())}")
                return None, {}
            
            if segment_idx >= len(road_data['mask_data']):
                logger.warning(f"❌ Segment index {segment_idx} out of range for mask_data (length: {len(road_data['mask_data'])})")
                return None, {}
            
            mask_info = road_data['mask_data'][segment_idx]
            logger.debug(f"📊 Mask info keys: {list(mask_info.keys())}")
            
            mask_overlay = mask_info.get('mask_overlay')
            mask_data = mask_info.get('mask')
            mask_stats = mask_info.get('mask_stats', {})
            
            logger.debug(f"🎭 mask_overlay: {'✅ Available' if mask_overlay is not None else '❌ None'}")
            logger.debug(f"🎭 mask_data: {'✅ Available' if mask_data is not None else '❌ None'}")
            logger.debug(f"📈 mask_stats: {mask_stats}")
            
            if mask_overlay is not None:
                logger.info("✅ Using pre-generated mask overlay")
                return mask_overlay, mask_stats
            elif mask_data is not None:
                logger.info("🔨 Creating mask overlay from raw mask data")
                # Create mask overlay using same approach as preprocessing annotations
                overlay_image = self._create_mask_overlay(original_image, mask_data)
                return overlay_image, mask_stats
            else:
                logger.warning("❌ No mask overlay or mask data available")
            
            return None, {}
            
        except Exception as e:
            logger.error(f"❌ Error creating mask image: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None, {}
    
    def _create_gradcam_overlay(self, original_image: Image.Image, gradcam_data: np.ndarray) -> np.ndarray:
        """Create Grad-CAM overlay directly without matplotlib, ensuring full image coverage."""
        try:
            # Convert PIL to numpy
            img_array = np.array(original_image)
            
            logger.debug(f"Original image shape: {img_array.shape}")
            logger.debug(f"Grad-CAM data shape: {gradcam_data.shape}")
            
            # Ensure Grad-CAM matches image dimensions exactly
            if gradcam_data.shape != img_array.shape[:2]:
                logger.info(f"Resizing Grad-CAM from {gradcam_data.shape} to {img_array.shape[:2]}")
                gradcam_resized = cv2.resize(gradcam_data, (img_array.shape[1], img_array.shape[0]), 
                                           interpolation=cv2.INTER_LINEAR)
            else:
                gradcam_resized = gradcam_data
            
            # Normalize Grad-CAM to 0-255 range
            if gradcam_resized.max() > gradcam_resized.min():
                gradcam_norm = ((gradcam_resized - gradcam_resized.min()) / 
                              (gradcam_resized.max() - gradcam_resized.min()) * 255).astype(np.uint8)
            else:
                # Handle case where all values are the same
                gradcam_norm = np.zeros_like(gradcam_resized, dtype=np.uint8)
            
            # Apply colormap (jet colormap for attention visualization)
            gradcam_colored = cv2.applyColorMap(gradcam_norm, cv2.COLORMAP_JET)
            gradcam_colored = cv2.cvtColor(gradcam_colored, cv2.COLOR_BGR2RGB)
            
            logger.debug(f"Final Grad-CAM overlay shape: {gradcam_colored.shape}")
            
            # Blend with original image (60% original, 40% heatmap)
            overlay = cv2.addWeighted(img_array, 0.6, gradcam_colored, 0.4, 0)
            
            return overlay
            
        except Exception as e:
            logger.error(f"Error creating Grad-CAM overlay: {e}")
            return np.array(original_image)
    
    def _create_mask_overlay(self, original_image: Image.Image, mask_data: np.ndarray) -> np.ndarray:
        """Create mask overlay using the same approach as preprocessing annotations."""
        try:
            # Convert PIL to numpy
            img_array = np.array(original_image)
            
            logger.debug(f"Original image shape: {img_array.shape}")
            logger.debug(f"Mask data shape: {mask_data.shape}")
            logger.debug(f"Mask data range: {mask_data.min()} to {mask_data.max()}")
            
            # Resize mask to match image if needed
            if mask_data.shape != img_array.shape[:2]:
                logger.info(f"Resizing mask from {mask_data.shape} to {img_array.shape[:2]}")
                mask_resized = cv2.resize(mask_data, (img_array.shape[1], img_array.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask_data
            
            # Create colored mask (green for roads, same as preprocessing annotations)
            if len(mask_resized.shape) == 2:
                colored_mask = np.zeros_like(img_array)
                # Handle mask that's already in 0-255 range (from test_inference.py format)
                if mask_resized.max() > 1:
                    logger.debug("Using 0-255 range mask")
                    colored_mask[:, :, 1] = mask_resized  # Green channel, already 0-255
                else:
                    logger.debug("Converting 0-1 range mask to 0-255")
                    colored_mask[:, :, 1] = mask_resized * 255  # Convert 0-1 to 0-255
            else:
                colored_mask = mask_resized.copy()
            
            logger.debug(f"Colored mask shape: {colored_mask.shape}, range: {colored_mask.min()} to {colored_mask.max()}")
            
            # Apply overlay with opacity using cv2.addWeighted (same as experiments)
            opacity = 0.5  # Semi-transparent overlay like in annotations
            overlay = cv2.addWeighted(img_array, 1.0 - opacity, colored_mask, opacity, 0)
            
            logger.info(f"✅ Created mask overlay successfully with opacity {opacity}")
            
            return overlay
            
        except Exception as e:
            logger.error(f"❌ Error creating mask overlay: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return np.array(original_image)
    
    def _display_predictions(self, road_data: Dict, segment_idx: int, segment_key: str):
        """Display prediction metrics."""
        import streamlit as st
        
        st.markdown("**Predictions**")
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        # Get predictions for this segment
        damage_pred = road_data['predictions']['damage'][segment_idx]
        occlusion_pred = road_data['predictions']['occlusion'][segment_idx]
        crop_pred = road_data['predictions']['crop'][segment_idx]
        
        with pred_col1:
            color = "🔴" if damage_pred['prediction'] else "🟢"
            st.metric(
                f"{color} Damage",
                f"{damage_pred['probability']:.3f}",
                delta="DETECTED" if damage_pred['prediction'] else "None"
            )
        
        with pred_col2:
            color = "🟡" if occlusion_pred['prediction'] else "🟢"
            st.metric(
                f"{color} Occlusion", 
                f"{occlusion_pred['probability']:.3f}",
                delta="DETECTED" if occlusion_pred['prediction'] else "None"
            )
        
        with pred_col3:
            color = "🟠" if crop_pred['prediction'] else "🟢"
            st.metric(
                f"{color} Crop Issues",
                f"{crop_pred['probability']:.3f}",
                delta="DETECTED" if crop_pred['prediction'] else "None"
            )
    
    def _display_coordinates(self, image_info: Dict, segment_key: str):
        """Display coordinate information."""
        import streamlit as st
        
        st.markdown("**Location Information**")
        coord_col1, coord_col2 = st.columns(2)
        
        with coord_col1:
            st.info(f"📍 Latitude: {image_info['latitude']:.6f}")
        
        with coord_col2:
            st.info(f"📍 Longitude: {image_info['longitude']:.6f}")
