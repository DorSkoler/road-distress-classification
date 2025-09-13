#!/usr/bin/env python3
"""
Road Folder Processing Module
Handles coordinate-based road reconstruction and analysis.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class RoadImageData:
    """Data structure for individual road image with coordinates."""
    
    def __init__(self, filepath: str, sequence: int, longitude: float, latitude: float):
        self.filepath = filepath
        self.sequence = sequence  
        self.longitude = longitude
        self.latitude = latitude
        self.predictions = None
        self.gradcam = None
        self.mask = None
        self.mask_overlay = None
        self.mask_stats = {}

class RoadProcessor:
    """Processes entire road folders with coordinate-based reconstruction."""
    
    def __init__(self, ensemble_engine, heatmap_generator, mask_generator=None):
        """
        Initialize road processor.
        
        Args:
            ensemble_engine: EnsembleInferenceEngine instance
            heatmap_generator: HeatmapGenerator instance
            mask_generator: RoadMaskGenerator instance (optional)
        """
        self.ensemble_engine = ensemble_engine
        self.heatmap_generator = heatmap_generator
        self.mask_generator = mask_generator
        self.road_data = []
        
        logger.info(f"RoadProcessor initialized with mask generation: {mask_generator is not None}")
        
    def parse_filename(self, filename: str) -> Optional[Tuple[int, float, float]]:
        """
        Parse filename format: 'XXX_longitude_latitude.png'
        
        Args:
            filename: Image filename to parse
            
        Returns:
            (sequence_number, longitude, latitude) or None if invalid
        """
        # Pattern to match: digits_float_float.extension
        pattern = r'^(\d+)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)\.(?:png|jpg|jpeg)$'
        match = re.match(pattern, filename, re.IGNORECASE)
        
        if match:
            sequence = int(match.group(1))
            latitude = float(match.group(2))   # Second number is latitude
            longitude = float(match.group(3))  # Third number is longitude
            
            # Log parsed coordinates for debugging
            logger.info(f"Parsed {filename}: seq={sequence}, lon={longitude}, lat={latitude}")
            
            # Validate coordinates
            if not self._is_valid_coordinate(latitude, longitude):
                logger.error(f"Invalid coordinates in {filename}: lat={latitude}, lon={longitude}")
                return None
            
            return (sequence, longitude, latitude)
        
        logger.warning(f"Invalid filename format: {filename}")
        return None
    
    def load_road_folder(self, uploaded_files) -> List[RoadImageData]:
        """
        Load and validate road images from uploaded files.
        
        Args:
            uploaded_files: List of uploaded Streamlit file objects
            
        Returns:
            List of RoadImageData objects sorted by sequence
        """
        road_images = []
        
        for uploaded_file in uploaded_files:
            try:
                # Parse filename
                parsed = self.parse_filename(uploaded_file.name)
                if not parsed:
                    logger.warning(f"Skipping invalid filename: {uploaded_file.name}")
                    continue
                
                sequence, longitude, latitude = parsed
                
                # Create RoadImageData object
                road_img = RoadImageData(
                    filepath=uploaded_file.name,
                    sequence=sequence,
                    longitude=longitude, 
                    latitude=latitude
                )
                road_images.append(road_img)
                
            except Exception as e:
                logger.error(f"Error processing {uploaded_file.name}: {e}")
                continue
        
        # Sort by sequence number
        road_images.sort(key=lambda x: x.sequence)
        
        logger.info(f"Loaded {len(road_images)} valid road images")
        return road_images
    
    def process_road_images(self, uploaded_files, progress_callback=None) -> Dict:
        """
        Process entire road folder and return comprehensive results.
        
        Args:
            uploaded_files: List of uploaded Streamlit file objects
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing processed road data
        """
        
        # Load and validate images
        road_images = self.load_road_folder(uploaded_files)
        
        if not road_images:
            raise ValueError("No valid road images found in upload")
        
        # Initialize results structure
        results = {
            'images': [],
            'coordinates': [],
            'predictions': {
                'damage': [],
                'occlusion': [], 
                'crop': []
            },
            'gradcam_data': [],
            'mask_data': [],
            'metadata': {
                'total_images': len(road_images),
                'sequence_range': (road_images[0].sequence, road_images[-1].sequence),
                'coordinate_bounds': self._calculate_bounds(road_images)
            }
        }
        
        logger.info(f"Processing {len(road_images)} road images")
        
        # Process each image
        for i, road_img in enumerate(road_images):
            if progress_callback:
                progress_callback((i + 1) / len(road_images))
            
            try:
                # Find uploaded file for this road image
                uploaded_file = next(f for f in uploaded_files if f.name == road_img.filepath)
                
                # Load image
                image = Image.open(uploaded_file)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Generate road mask if mask generator is available
                if self.mask_generator and self.mask_generator.is_available():
                    try:
                        # Generate binary mask
                        mask = self.mask_generator.generate_mask(image)
                        road_img.mask = mask
                        
                        # Generate mask overlay for visualization
                        if mask is not None:
                            mask_overlay = self.mask_generator.create_mask_overlay(np.array(image), mask)
                            road_img.mask_overlay = mask_overlay
                            
                            # Calculate mask statistics
                            road_img.mask_stats = self.mask_generator.get_mask_statistics(mask)
                        
                        logger.debug(f"Generated mask for {road_img.filepath}: {road_img.mask_stats}")
                        
                    except Exception as mask_error:
                        logger.warning(f"Mask generation failed for {road_img.filepath}: {mask_error}")
                        road_img.mask = None
                        road_img.mask_overlay = None
                        road_img.mask_stats = {}
                
                # Run ensemble prediction
                ensemble_results = self.ensemble_engine.predict_ensemble(np.array(image))
                road_img.predictions = ensemble_results['ensemble_results']
                
                # Generate Grad-CAM for damage visualization
                try:
                    gradcam_map, _ = self.ensemble_engine.get_damage_confidence_map(
                        np.array(image), method='gradcam', target_class='damage', model_name='combined'
                    )
                    road_img.gradcam = gradcam_map
                except Exception as gradcam_error:
                    logger.warning(f"Grad-CAM generation failed for {road_img.filepath}: {gradcam_error}")
                    # Create fallback confidence map
                    road_img.gradcam = np.full((256, 256), 
                                              road_img.predictions['class_results']['damage']['probability'], 
                                              dtype=np.float32)
                
                # Store image metadata
                results['images'].append({
                    'sequence': road_img.sequence,
                    'filename': road_img.filepath,
                    'longitude': road_img.longitude,
                    'latitude': road_img.latitude
                })
                
                # Store coordinates for mapping (validate first)
                if self._is_valid_coordinate(road_img.latitude, road_img.longitude):
                    results['coordinates'].append([road_img.latitude, road_img.longitude])
                else:
                    logger.warning(f"Invalid coordinates for {road_img.filepath}: lat={road_img.latitude}, lon={road_img.longitude}")
                    continue
                
                # Store predictions by class
                for class_name in ['damage', 'occlusion', 'crop']:
                    class_result = road_img.predictions['class_results'][class_name]
                    results['predictions'][class_name].append({
                        'sequence': road_img.sequence,
                        'probability': float(class_result['probability']),
                        'prediction': bool(class_result['prediction']),
                        'confidence': float(class_result['confidence'])
                    })
                
                # Store Grad-CAM data (only if coordinates are valid)
                if self._is_valid_coordinate(road_img.latitude, road_img.longitude):
                    results['gradcam_data'].append({
                        'sequence': road_img.sequence,
                        'gradcam': road_img.gradcam,
                        'coordinates': [road_img.latitude, road_img.longitude]
                    })
                    
                    # Store mask data if available
                    results['mask_data'].append({
                        'sequence': road_img.sequence,
                        'mask': road_img.mask,
                        'mask_overlay': road_img.mask_overlay,
                        'mask_stats': road_img.mask_stats
                    })
                
                damage_prob = class_result['probability']
                logger.info(f"Processed image {road_img.sequence:03d}: damage={damage_prob:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to process image {road_img.filepath}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(results['images'])} road segments")
        logger.info(f"Valid coordinates: {len(results['coordinates'])}")
        logger.info(f"Valid Grad-CAM data: {len(results['gradcam_data'])}")
        
        if len(results['coordinates']) == 0:
            logger.error("No valid coordinates found! Check filename format and coordinate ranges.")
        
        return results
    
    def calculate_road_score(self, results: Dict) -> Dict:
        """
        Calculate overall road health score (0-100) with detailed breakdown.
        
        Args:
            results: Processed road data dictionary
            
        Returns:
            Dictionary containing scoring results and analysis
        """
        
        if not results['predictions']['damage']:
            return {
                'overall_score': 0, 
                'breakdown': {}, 
                'segments': [],
                'health_category': 'Unknown'
            }
        
        # Scoring weights and penalties
        DAMAGE_PENALTIES = {'high': 50, 'medium': 30, 'low': 15}  # Based on confidence
        OCCLUSION_PENALTIES = {'high': 20, 'medium': 12, 'low': 5}
        CROP_PENALTIES = {'high': 15, 'medium': 8, 'low': 3}
        
        segment_scores = []
        
        # Calculate score for each segment
        for i in range(len(results['predictions']['damage'])):
            segment_score = 100  # Start with perfect score
            
            # Apply damage penalties
            damage_prob = results['predictions']['damage'][i]['probability']
            damage_pred = results['predictions']['damage'][i]['prediction']
            
            if damage_pred:
                if damage_prob > 0.8:
                    penalty = DAMAGE_PENALTIES['high']
                elif damage_prob > 0.5:
                    penalty = DAMAGE_PENALTIES['medium'] 
                else:
                    penalty = DAMAGE_PENALTIES['low']
                segment_score -= penalty
            
            # Apply occlusion penalties
            occlusion_prob = results['predictions']['occlusion'][i]['probability']
            occlusion_pred = results['predictions']['occlusion'][i]['prediction']
            
            if occlusion_pred:
                if occlusion_prob > 0.8:
                    penalty = OCCLUSION_PENALTIES['high']
                elif occlusion_prob > 0.5:
                    penalty = OCCLUSION_PENALTIES['medium']
                else:
                    penalty = OCCLUSION_PENALTIES['low']
                segment_score -= penalty
            
            # Apply crop penalties
            crop_prob = results['predictions']['crop'][i]['probability']
            crop_pred = results['predictions']['crop'][i]['prediction']
            
            if crop_pred:
                if crop_prob > 0.8:
                    penalty = CROP_PENALTIES['high']
                elif crop_prob > 0.5:
                    penalty = CROP_PENALTIES['medium']
                else:
                    penalty = CROP_PENALTIES['low']
                segment_score -= penalty
            
            # Ensure score doesn't go below 0
            segment_score = max(0, segment_score)
            
            segment_scores.append({
                'sequence': i,
                'score': float(segment_score),
                'damage_prob': float(damage_prob),
                'occlusion_prob': float(occlusion_prob),
                'crop_prob': float(crop_prob)
            })
        
        # Calculate overall score (weighted average)
        overall_score = float(np.mean([s['score'] for s in segment_scores]))
        
        # Count issues by type
        damage_count = sum(1 for p in results['predictions']['damage'] if p['prediction'])
        occlusion_count = sum(1 for p in results['predictions']['occlusion'] if p['prediction'])
        crop_count = sum(1 for p in results['predictions']['crop'] if p['prediction'])
        
        # Calculate average probabilities
        avg_damage_prob = float(np.mean([p['probability'] for p in results['predictions']['damage']]))
        avg_occlusion_prob = float(np.mean([p['probability'] for p in results['predictions']['occlusion']]))
        avg_crop_prob = float(np.mean([p['probability'] for p in results['predictions']['crop']]))
        
        scoring_result = {
            'overall_score': overall_score,
            'breakdown': {
                'total_segments': len(segment_scores),
                'damage_segments': damage_count,
                'occlusion_segments': occlusion_count,
                'crop_segments': crop_count,
                'average_damage_prob': avg_damage_prob,
                'average_occlusion_prob': avg_occlusion_prob,
                'average_crop_prob': avg_crop_prob,
                'damage_percentage': (damage_count / len(segment_scores)) * 100,
                'occlusion_percentage': (occlusion_count / len(segment_scores)) * 100,
                'crop_percentage': (crop_count / len(segment_scores)) * 100
            },
            'segments': segment_scores,
            'health_category': self._get_health_category(overall_score)
        }
        
        logger.info(f"Road scoring complete: {overall_score:.1f}/100 ({scoring_result['health_category']})")
        return scoring_result
    
    def _calculate_bounds(self, road_images: List[RoadImageData]) -> Dict:
        """Calculate coordinate bounds for the road."""
        lats = [img.latitude for img in road_images]
        lons = [img.longitude for img in road_images]
        
        return {
            'lat_min': min(lats), 
            'lat_max': max(lats),
            'lon_min': min(lons), 
            'lon_max': max(lons),
            'center_lat': (min(lats) + max(lats)) / 2,
            'center_lon': (min(lons) + max(lons)) / 2
        }
    
    def _get_health_category(self, score: float) -> str:
        """Convert numeric score to health category."""
        if score >= 90: 
            return "Excellent"
        elif score >= 75: 
            return "Good" 
        elif score >= 60: 
            return "Fair"
        elif score >= 40: 
            return "Poor"
        else: 
            return "Critical"
    
    def _is_valid_coordinate(self, latitude: float, longitude: float) -> bool:
        """Validate GPS coordinates are within valid ranges."""
        try:
            # Check for valid latitude (-90 to 90)
            if not (-90 <= latitude <= 90):
                return False
            
            # Check for valid longitude (-180 to 180)
            if not (-180 <= longitude <= 180):
                return False
            
            # Check for NaN or infinity
            if not (np.isfinite(latitude) and np.isfinite(longitude)):
                return False
                
            return True
        except (TypeError, ValueError):
            return False
    
    def validate_road_sequence(self, road_images: List[RoadImageData]) -> Dict:
        """
        Validate road sequence for gaps and inconsistencies.
        
        Args:
            road_images: List of road image data
            
        Returns:
            Validation results dictionary
        """
        if not road_images:
            return {'valid': False, 'issues': ['No images provided']}
        
        issues = []
        sequences = [img.sequence for img in road_images]
        
        # Check for duplicate sequences
        if len(sequences) != len(set(sequences)):
            issues.append("Duplicate sequence numbers found")
        
        # Check for large gaps in sequence
        if len(sequences) > 1:
            sorted_sequences = sorted(sequences)
            gaps = []
            for i in range(1, len(sorted_sequences)):
                gap = sorted_sequences[i] - sorted_sequences[i-1]
                if gap > 1:
                    gaps.append(f"Gap of {gap-1} between {sorted_sequences[i-1]} and {sorted_sequences[i]}")
            
            if gaps:
                issues.append(f"Sequence gaps found: {', '.join(gaps)}")
        
        # Check coordinate consistency (basic validation)
        lats = [img.latitude for img in road_images]
        lons = [img.longitude for img in road_images]
        
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        
        # Very basic sanity check for coordinate ranges
        if lat_range > 1.0 or lon_range > 1.0:
            issues.append("Unusually large coordinate range - may span multiple roads")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'stats': {
                'sequence_count': len(sequences),
                'sequence_range': (min(sequences), max(sequences)),
                'coordinate_range': {
                    'lat_range': lat_range,
                    'lon_range': lon_range
                }
            }
        }
