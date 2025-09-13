# Detailed Implementation Guide: Road Folder Analysis

## 📁 Current Data Analysis

Based on examination of `road-distress-classification/data/coryell/Co Rd 342/img/`, I found:
- **300+ images** in sequence format `{num}_{longitude}_{latitude}.png`
- **Coordinates range**: Lat ~31.296-31.334, Lon ~-97.543 to -97.570
- **Sequential numbering**: 000-299+ creating a continuous road path
- **File example**: `000_31.296905_-97.543646.png`

## 🏗️ Files to Create

### 1. Core Processing Module
**📁 `src/road_processor.py`**
```python
#!/usr/bin/env python3
"""
Road Folder Processing Module
Handles coordinate-based road reconstruction and analysis.
"""

import re
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
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
        
class RoadProcessor:
    """Processes entire road folders with coordinate-based reconstruction."""
    
    def __init__(self, ensemble_engine, heatmap_generator):
        self.ensemble_engine = ensemble_engine
        self.heatmap_generator = heatmap_generator
        self.road_data = []
        
    def parse_filename(self, filename: str) -> Optional[Tuple[int, float, float]]:
        """
        Parse filename format: 'XXX_longitude_latitude.png'
        Returns: (sequence_number, longitude, latitude) or None if invalid
        """
        pattern = r'^(\d+)_(-?\d+\.?\d*)_(-?\d+\.?\d*)\.(?:png|jpg|jpeg)$'
        match = re.match(pattern, filename, re.IGNORECASE)
        
        if match:
            sequence = int(match.group(1))
            longitude = float(match.group(2))  
            latitude = float(match.group(3))
            return (sequence, longitude, latitude)
        return None
    
    def load_road_folder(self, uploaded_files) -> List[RoadImageData]:
        """Load and validate road images from uploaded files."""
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
        """Process entire road folder and return comprehensive results."""
        
        # Load and validate images
        road_images = self.load_road_folder(uploaded_files)
        
        if not road_images:
            raise ValueError("No valid road images found in upload")
        
        # Process each image
        results = {
            'images': [],
            'coordinates': [],
            'predictions': {
                'damage': [],
                'occlusion': [], 
                'crop': []
            },
            'gradcam_data': [],
            'metadata': {
                'total_images': len(road_images),
                'sequence_range': (road_images[0].sequence, road_images[-1].sequence),
                'coordinate_bounds': self._calculate_bounds(road_images)
            }
        }
        
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
                
                # Run ensemble prediction
                ensemble_results = self.ensemble_engine.predict_ensemble(np.array(image))
                road_img.predictions = ensemble_results['ensemble_results']
                
                # Generate Grad-CAM  
                gradcam_map, _ = self.ensemble_engine.get_damage_confidence_map(
                    np.array(image), method='gradcam', target_class='damage', model_name='combined'
                )
                road_img.gradcam = gradcam_map
                
                # Store results
                results['images'].append({
                    'sequence': road_img.sequence,
                    'filename': road_img.filepath,
                    'longitude': road_img.longitude,
                    'latitude': road_img.latitude
                })
                
                results['coordinates'].append([road_img.latitude, road_img.longitude])
                
                # Store predictions
                for class_name in ['damage', 'occlusion', 'crop']:
                    class_result = road_img.predictions['class_results'][class_name]
                    results['predictions'][class_name].append({
                        'sequence': road_img.sequence,
                        'probability': class_result['probability'],
                        'prediction': class_result['prediction'],
                        'confidence': class_result['confidence']
                    })
                
                # Store Grad-CAM data
                results['gradcam_data'].append({
                    'sequence': road_img.sequence,
                    'gradcam': gradcam_map,
                    'coordinates': [road_img.latitude, road_img.longitude]
                })
                
                logger.info(f"Processed image {road_img.sequence}: damage={class_result['probability']:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to process image {road_img.filepath}: {e}")
                continue
        
        return results
    
    def calculate_road_score(self, results: Dict) -> Dict:
        """Calculate overall road health score (0-100) with detailed breakdown."""
        
        if not results['predictions']['damage']:
            return {'overall_score': 0, 'breakdown': {}, 'segments': []}
        
        # Scoring weights and penalties
        DAMAGE_PENALTIES = {'high': 50, 'medium': 30, 'low': 15}  # Based on confidence
        OCCLUSION_PENALTIES = {'high': 20, 'medium': 12, 'low': 5}
        CROP_PENALTIES = {'high': 15, 'medium': 8, 'low': 3}
        
        segment_scores = []
        total_penalties = 0
        
        for i in range(len(results['predictions']['damage'])):
            segment_score = 100  # Start with perfect score
            
            # Damage penalties
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
                total_penalties += penalty
            
            # Occlusion penalties
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
                total_penalties += penalty
            
            # Crop penalties
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
                total_penalties += penalty
            
            # Ensure score doesn't go below 0
            segment_score = max(0, segment_score)
            segment_scores.append({
                'sequence': i,
                'score': segment_score,
                'damage_prob': damage_prob,
                'occlusion_prob': occlusion_prob,
                'crop_prob': crop_prob
            })
        
        # Calculate overall score (weighted average)
        overall_score = np.mean([s['score'] for s in segment_scores])
        
        # Count issues
        damage_count = sum(1 for p in results['predictions']['damage'] if p['prediction'])
        occlusion_count = sum(1 for p in results['predictions']['occlusion'] if p['prediction'])
        crop_count = sum(1 for p in results['predictions']['crop'] if p['prediction'])
        
        return {
            'overall_score': float(overall_score),
            'breakdown': {
                'total_segments': len(segment_scores),
                'damage_segments': damage_count,
                'occlusion_segments': occlusion_count,
                'crop_segments': crop_count,
                'average_damage_prob': float(np.mean([p['probability'] for p in results['predictions']['damage']])),
                'total_penalties': total_penalties
            },
            'segments': segment_scores,
            'health_category': self._get_health_category(overall_score)
        }
    
    def _calculate_bounds(self, road_images: List[RoadImageData]) -> Dict:
        """Calculate coordinate bounds for the road."""
        lats = [img.latitude for img in road_images]
        lons = [img.longitude for img in road_images]
        
        return {
            'lat_min': min(lats), 'lat_max': max(lats),
            'lon_min': min(lons), 'lon_max': max(lons),
            'center_lat': (min(lats) + max(lats)) / 2,
            'center_lon': (min(lons) + max(lons)) / 2
        }
    
    def _get_health_category(self, score: float) -> str:
        """Convert numeric score to health category."""
        if score >= 90: return "Excellent"
        elif score >= 75: return "Good" 
        elif score >= 60: return "Fair"
        elif score >= 40: return "Poor"
        else: return "Critical"
```

### 2. Road Visualization Module
**📁 `src/road_visualizer.py`**
```python
#!/usr/bin/env python3
"""
Road Visualization Module
Creates interactive maps with Grad-CAM overlays for road analysis.
"""

import folium
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple
import base64
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class RoadVisualizer:
    """Creates interactive road visualizations with health indicators."""
    
    def __init__(self):
        self.color_map = {
            'Excellent': '#00ff00',  # Green
            'Good': '#80ff00',       # Light green  
            'Fair': '#ffff00',       # Yellow
            'Poor': '#ff8000',       # Orange
            'Critical': '#ff0000'    # Red
        }
    
    def create_interactive_road_map(self, road_data: Dict, scoring_data: Dict) -> go.Figure:
        """Create interactive Plotly map with road health visualization."""
        
        # Extract coordinates and scores
        coordinates = road_data['coordinates']
        lats = [coord[0] for coord in coordinates]
        lons = [coord[1] for coord in coordinates]
        
        # Create segment scores for color mapping
        segment_scores = [s['score'] for s in scoring_data['segments']]
        damage_probs = [s['damage_prob'] for s in scoring_data['segments']]
        
        # Create color scale based on scores
        colors = []
        for score in segment_scores:
            if score >= 90: colors.append('#00ff00')      # Excellent
            elif score >= 75: colors.append('#80ff00')    # Good
            elif score >= 60: colors.append('#ffff00')    # Fair  
            elif score >= 40: colors.append('#ff8000')    # Poor
            else: colors.append('#ff0000')                # Critical
        
        # Create the main map
        fig = go.Figure()
        
        # Add road path as a line
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='lines+markers',
            line=dict(width=6, color='blue'),
            marker=dict(size=8, color=colors, opacity=0.8),
            text=[f"Segment {i}<br>Score: {score:.1f}<br>Damage: {prob:.3f}" 
                  for i, (score, prob) in enumerate(zip(segment_scores, damage_probs))],
            hovertemplate="<b>%{text}</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>",
            name="Road Segments"
        ))
        
        # Calculate center point
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Configure map layout
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=14
            ),
            height=600,
            margin=dict(l=0, r=0, t=30, b=0),
            title=f"Road Health Map - Overall Score: {scoring_data['overall_score']:.1f}/100"
        )
        
        return fig
    
    def create_score_breakdown_chart(self, scoring_data: Dict) -> go.Figure:
        """Create breakdown chart of road health components."""
        
        breakdown = scoring_data['breakdown']
        
        # Create subplot with multiple charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Health Distribution',
                'Issue Counts', 
                'Score Along Road',
                'Problem Severity'
            ),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Health category pie chart
        health_counts = {}
        for segment in scoring_data['segments']:
            score = segment['score']
            category = self._score_to_category(score)
            health_counts[category] = health_counts.get(category, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(health_counts.keys()),
                values=list(health_counts.values()),
                marker_colors=[self.color_map[cat] for cat in health_counts.keys()]
            ),
            row=1, col=1
        )
        
        # 2. Issue counts bar chart
        issue_types = ['Damage', 'Occlusion', 'Crop']
        issue_counts = [
            breakdown['damage_segments'],
            breakdown['occlusion_segments'], 
            breakdown['crop_segments']
        ]
        
        fig.add_trace(
            go.Bar(
                x=issue_types,
                y=issue_counts,
                marker_color=['#ff4444', '#ffaa44', '#4444ff']
            ),
            row=1, col=2
        )
        
        # 3. Score progression along road
        sequences = [s['sequence'] for s in scoring_data['segments']]
        scores = [s['score'] for s in scoring_data['segments']]
        
        fig.add_trace(
            go.Scatter(
                x=sequences,
                y=scores,
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        # 4. Problem severity (average probabilities)
        avg_damage = breakdown['average_damage_prob']
        avg_occlusion = np.mean([s['occlusion_prob'] for s in scoring_data['segments']])
        avg_crop = np.mean([s['crop_prob'] for s in scoring_data['segments']])
        
        fig.add_trace(
            go.Bar(
                x=['Damage', 'Occlusion', 'Crop'],
                y=[avg_damage, avg_occlusion, avg_crop],
                marker_color=['#ff4444', '#ffaa44', '#4444ff']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Road Health Analysis Breakdown"
        )
        
        return fig
    
    def create_gradcam_overlay_map(self, road_data: Dict, gradcam_data: List) -> go.Figure:
        """Create map with Grad-CAM overlays for damage visualization."""
        
        # This would require more complex implementation to overlay actual Grad-CAM images
        # For now, create a heatmap based on damage probabilities
        
        coordinates = road_data['coordinates']
        lats = [coord[0] for coord in coordinates]
        lons = [coord[1] for coord in coordinates]
        
        # Extract damage probabilities for heatmap intensity
        damage_probs = []
        for pred in road_data['predictions']['damage']:
            damage_probs.append(pred['probability'])
        
        fig = go.Figure()
        
        # Add heatmap layer
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=dict(
                size=15,
                color=damage_probs,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Damage Probability"),
                opacity=0.7
            ),
            text=[f"Segment {i}<br>Damage: {prob:.3f}" 
                  for i, prob in enumerate(damage_probs)],
            hovertemplate="<b>%{text}</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>",
            name="Damage Heatmap"
        ))
        
        # Add road path
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='lines',
            line=dict(width=3, color='blue'),
            opacity=0.5,
            name="Road Path"
        ))
        
        # Calculate center
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=14
            ),
            height=600,
            margin=dict(l=0, r=0, t=30, b=0),
            title="Road Damage Heatmap (Grad-CAM Based)"
        )
        
        return fig
    
    def _score_to_category(self, score: float) -> str:
        """Convert numeric score to health category."""
        if score >= 90: return "Excellent"
        elif score >= 75: return "Good"
        elif score >= 60: return "Fair" 
        elif score >= 40: return "Poor"
        else: return "Critical"
```

## 🔧 Files to Modify

### 1. Main UI Application
**📁 `app.py` - Key Modifications**

Add after imports:
```python
from src.road_processor import RoadProcessor
from src.road_visualizer import RoadVisualizer
```

Modify processing mode selection:
```python
# Replace existing processing_mode radio button
processing_mode = st.sidebar.radio(
    "Processing Mode",
    ["Single Image", "Batch Processing", "Road Folder Analysis"],  # NEW OPTION
    help="Choose between analyzing one image, multiple images, or entire road folder"
)
```

Add new main processing section:
```python
elif processing_mode == "Road Folder Analysis":  # NEW SECTION
    st.header("📁 Road Folder Analysis")
    
    uploaded_files = st.file_uploader(
        "Choose road images...",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload all images from a road folder with format: XXX_longitude_latitude.ext"
    )
    
    if uploaded_files:
        # Validate filename format
        valid_count = 0
        for file in uploaded_files:
            if re.match(r'^\d+_-?\d+\.?\d*_-?\d+\.?\d*\.(png|jpg|jpeg)$', file.name, re.IGNORECASE):
                valid_count += 1
        
        st.info(f"Found {valid_count}/{len(uploaded_files)} valid road images")
        
        if valid_count < 2:
            st.error("Need at least 2 valid road images with correct filename format")
        else:
            if st.button("🚀 Analyze Road", type="primary"):
                process_road_folder(uploaded_files, engine, heatmap_gen, thresholds)
```

Add new processing function:
```python
def process_road_folder(uploaded_files, engine, heatmap_gen, thresholds):
    """Process entire road folder and display comprehensive results."""
    
    # Initialize processors
    road_processor = RoadProcessor(engine, heatmap_gen)
    road_visualizer = RoadVisualizer()
    
    # Update engine thresholds
    engine.update_thresholds(thresholds)
    
    # Process road with progress tracking
    with st.spinner("Processing road images..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress):
            progress_bar.progress(progress)
            status_text.text(f"Processing: {progress:.1%} complete")
        
        try:
            # Process all road images
            road_data = road_processor.process_road_images(uploaded_files, update_progress)
            
            # Calculate road score
            scoring_data = road_processor.calculate_road_score(road_data)
            
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            display_road_results(road_data, scoring_data, road_visualizer)
            
        except Exception as e:
            st.error(f"Road processing failed: {e}")
            return

def display_road_results(road_data, scoring_data, visualizer):
    """Display comprehensive road analysis results."""
    
    st.subheader("🎯 Road Health Assessment")
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = scoring_data['overall_score']
        category = scoring_data['health_category']
        color = "normal" if score >= 70 else "inverse" if score >= 40 else "off"
        
        st.metric(
            "Road Health Score", 
            f"{score:.1f}/100",
            delta=f"{category}",
            help="Overall road condition score"
        )
    
    with col2:
        st.metric(
            "Total Segments",
            scoring_data['breakdown']['total_segments'],
            help="Number of road image segments analyzed"
        )
    
    with col3:
        damage_pct = (scoring_data['breakdown']['damage_segments'] / 
                     scoring_data['breakdown']['total_segments'] * 100)
        st.metric(
            "Damage Detected",
            f"{damage_pct:.1f}%",
            delta=f"{scoring_data['breakdown']['damage_segments']} segments"
        )
    
    with col4:
        avg_damage = scoring_data['breakdown']['average_damage_prob']
        st.metric(
            "Avg Damage Confidence",
            f"{avg_damage:.3f}",
            help="Average damage detection confidence"
        )
    
    # Interactive road map
    st.subheader("🗺️ Interactive Road Map")
    road_map = visualizer.create_interactive_road_map(road_data, scoring_data)
    st.plotly_chart(road_map, use_container_width=True)
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["📊 Detailed Analysis", "🔥 Damage Heatmap", "📋 Segment Details"])
    
    with tab1:
        breakdown_chart = visualizer.create_score_breakdown_chart(scoring_data)
        st.plotly_chart(breakdown_chart, use_container_width=True)
    
    with tab2:
        gradcam_map = visualizer.create_gradcam_overlay_map(road_data, road_data['gradcam_data'])
        st.plotly_chart(gradcam_map, use_container_width=True)
    
    with tab3:
        # Segment details table
        segment_data = []
        for i, segment in enumerate(scoring_data['segments'][:20]):  # Show first 20
            img_data = road_data['images'][i]
            segment_data.append({
                'Sequence': segment['sequence'],
                'Filename': img_data['filename'],
                'Score': f"{segment['score']:.1f}",
                'Damage': f"{segment['damage_prob']:.3f}",
                'Occlusion': f"{segment['occlusion_prob']:.3f}",
                'Crop': f"{segment['crop_prob']:.3f}",
                'Health': visualizer._score_to_category(segment['score'])
            })
        
        st.dataframe(segment_data, use_container_width=True)
        
        if len(scoring_data['segments']) > 20:
            st.info(f"Showing first 20 of {len(scoring_data['segments'])} segments")
    
    # Download section
    st.subheader("💾 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # Road report JSON
        road_report = {
            'road_data': road_data,
            'scoring_data': scoring_data,
            'metadata': {
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_images': len(uploaded_files)
            }
        }
        
        json_str = json.dumps(road_report, indent=2, default=str)
        st.download_button(
            label="📄 Download Road Report",
            data=json_str,
            file_name=f"road_analysis_{int(time.time())}.json",
            mime="application/json"
        )
    
    with col2:
        st.info("📊 Additional exports (maps, charts) coming soon!")
```

### 2. Module Initialization
**📁 `src/__init__.py`** - Add imports:
```python
from .road_processor import RoadProcessor, RoadImageData
from .road_visualizer import RoadVisualizer
```

Update `__all__` list:
```python
__all__ = [
    'ModelLoader',
    'MultiModelLoader', 
    'HybridRoadDistressModel',
    'load_best_model_b',
    'ImageProcessor',
    'InferenceEngine',
    'EnsembleInferenceEngine',
    'create_inference_engine',
    'HeatmapGenerator',
    'RoadProcessor',           # NEW
    'RoadImageData',           # NEW
    'RoadVisualizer'           # NEW
]
```

### 3. Configuration Updates
**📁 `config.yaml`** - Add new section:
```yaml
# Road Processing Configuration
road_processing:
  scoring:
    damage_penalties:
      high_confidence: 50    # >0.8 confidence
      medium_confidence: 30  # 0.5-0.8 confidence
      low_confidence: 15     # <0.5 confidence
    
    occlusion_penalties:
      high: 20     # High obstruction  
      medium: 12   # Medium obstruction
      low: 5       # Low obstruction
    
    crop_penalties:
      high: 15     # Poor image quality
      medium: 8    # Some quality issues
      low: 3       # Minor issues
  
  visualization:
    map_zoom_level: 14
    health_colors:
      excellent: "#00ff00"    # 90-100
      good: "#80ff00"         # 75-89
      fair: "#ffff00"         # 60-74
      poor: "#ff8000"         # 40-59
      critical: "#ff0000"     # 0-39
      
  export:
    include_gradcam: true
    max_segments_display: 50
```

### 4. Requirements Update
**📁 `requirements.txt`** - Add new dependencies:
```txt
# Existing dependencies...
torch>=2.0.0
torchvision>=0.15.0
# ... (keep all existing)

# New dependencies for road analysis
folium>=0.14.0
geopandas>=0.14.0
haversine>=2.7.0
branca>=0.6.0
```

## 🧪 Testing Strategy

### 1. Create Test Data
**📁 `test_road_folder/`** - Sample structure:
```
test_road_folder/
├── 000_31.296905_-97.543646.png
├── 001_31.296954_-97.543848.png  
├── 002_31.296988_-97.544054.png
└── (copy 10-20 images from actual data)
```

### 2. Unit Tests
**📁 `tests/test_road_processor.py`**
```python
def test_filename_parsing():
    processor = RoadProcessor(None, None)
    
    # Valid filenames
    assert processor.parse_filename("000_31.296905_-97.543646.png") == (0, 31.296905, -97.543646)
    assert processor.parse_filename("123_45.678_-123.456.jpg") == (123, 45.678, -123.456)
    
    # Invalid filenames  
    assert processor.parse_filename("invalid.png") is None
    assert processor.parse_filename("000_invalid_coords.png") is None

def test_road_scoring():
    # Test scoring algorithm with known data
    pass
```

### 3. Integration Test
**📁 `test_road_integration.py`**
```python
def test_full_road_pipeline():
    """Test complete road processing pipeline."""
    
    # Load test images
    # Run processing
    # Validate results
    # Check score calculation
    # Verify visualization creation
    pass
```

## 🎯 Implementation Priority

### Phase 1 (Week 1): Core Infrastructure
1. ✅ Create `RoadProcessor` class with filename parsing
2. ✅ Implement basic coordinate validation
3. ✅ Add road processing to main UI
4. ✅ Test with sample data

### Phase 2 (Week 2): Scoring System  
1. ✅ Implement road health scoring algorithm
2. ✅ Add penalty system for damage/occlusion/crop
3. ✅ Create segment-level analysis
4. ✅ Test scoring accuracy

### Phase 3 (Week 3): Visualization
1. ✅ Create interactive Plotly maps
2. ✅ Add Grad-CAM heatmap overlays
3. ✅ Implement color-coded health indicators
4. ✅ Add zoom/pan functionality

### Phase 4 (Week 4): Polish & Testing
1. ✅ Comprehensive error handling
2. ✅ Performance optimization for large datasets  
3. ✅ Export functionality
4. ✅ Documentation and user guide

This guide provides a concrete implementation path with specific code examples and clear priorities. Each component can be developed and tested incrementally.
