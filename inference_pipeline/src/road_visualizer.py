#!/usr/bin/env python3
"""
Road Visualization Module
Creates interactive maps with health indicators for road analysis.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class RoadVisualizer:
    """Creates interactive road visualizations with health indicators."""
    
    def __init__(self):
        """Initialize the road visualizer with color schemes."""
        self.health_colors = {
            'Excellent': '#00ff00',  # Bright green
            'Good': '#80ff00',       # Light green  
            'Fair': '#ffff00',       # Yellow
            'Poor': '#ff8000',       # Orange
            'Critical': '#ff0000'    # Red
        }
        
        # Color scale for continuous mapping
        self.score_colorscale = [
            [0.0, '#ff0000'],    # 0-20: Red (Critical)
            [0.2, '#ff4000'],    
            [0.4, '#ff8000'],    # 20-40: Orange (Poor)
            [0.6, '#ffff00'],    # 40-60: Yellow (Fair)
            [0.75, '#80ff00'],   # 60-75: Light green (Good)  
            [1.0, '#00ff00']     # 75-100: Bright green (Excellent)
        ]
        
        logger.info("RoadVisualizer initialized")
    
    def create_interactive_road_map(self, road_data: Dict, scoring_data: Dict) -> go.Figure:
        """
        Create interactive Plotly map with road health visualization.
        
        Args:
            road_data: Processed road data dictionary
            scoring_data: Road scoring results
            
        Returns:
            Plotly Figure with interactive road map
        """
        
        # Extract coordinates and scores
        coordinates = road_data['coordinates']
        if not coordinates or len(coordinates) == 0:
            # Return empty map with error message
            fig = go.Figure()
            fig.update_layout(
                title="No valid coordinates found",
                annotations=[
                    dict(
                        text="No valid GPS coordinates found in road data.<br>Check filename format: XXX_latitude_longitude.ext",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, xanchor='center', yanchor='middle'
                    )
                ]
            )
            return fig
        
        # Validate coordinates
        valid_coordinates = []
        valid_indices = []
        for i, coord in enumerate(coordinates):
            lat, lon = coord
            if self._is_valid_coordinate(lat, lon):
                valid_coordinates.append(coord)
                valid_indices.append(i)
            else:
                logger.warning(f"Skipping invalid coordinate: lat={lat}, lon={lon}")
        
        if not valid_coordinates:
            fig = go.Figure()
            fig.update_layout(
                title="No valid coordinates found",
                annotations=[
                    dict(
                        text="All GPS coordinates are invalid.<br>Check coordinate ranges: lat(-90,90), lon(-180,180)",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, xanchor='center', yanchor='middle'
                    )
                ]
            )
            return fig
        
        coordinates = valid_coordinates
        
        lats = [coord[0] for coord in coordinates]
        lons = [coord[1] for coord in coordinates]
        
        # Create segment scores for color mapping (only for valid indices)
        segment_scores = [scoring_data['segments'][i]['score'] for i in valid_indices if i < len(scoring_data['segments'])]
        damage_probs = [scoring_data['segments'][i]['damage_prob'] for i in valid_indices if i < len(scoring_data['segments'])]
        
        # Ensure we have matching data
        if len(segment_scores) != len(coordinates) or len(damage_probs) != len(coordinates):
            logger.error(f"Data length mismatch: coords={len(coordinates)}, scores={len(segment_scores)}, damage={len(damage_probs)}")
            fig = go.Figure()
            fig.update_layout(title="Data processing error - coordinate/score mismatch")
            return fig
        
        # Create hover text with detailed information
        hover_texts = []
        for j, (score, damage_prob) in enumerate(zip(segment_scores, damage_probs)):
            original_index = valid_indices[j]
            segment = scoring_data['segments'][original_index]
            image_info = road_data['images'][original_index]
            
            hover_text = f"""
            <b>Segment {segment['sequence']}</b><br>
            Score: {score:.1f}/100<br>
            Health: {self._score_to_category(score)}<br>
            <br>
            Damage: {damage_prob:.3f}<br>
            Occlusion: {segment['occlusion_prob']:.3f}<br>
            Crop: {segment['crop_prob']:.3f}<br>
            <br>
            File: {image_info['filename']}<br>
            Coords: {image_info['latitude']:.6f}, {image_info['longitude']:.6f}
            """.strip()
            hover_texts.append(hover_text)
        
        # Create the main map figure
        fig = go.Figure()
        
        # Add road segments as markers with color coding (back to mapbox for proper map display)
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers+lines',
            marker=dict(
                size=12,
                color=segment_scores,
                colorscale=self.score_colorscale,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Health Score"),
                    len=0.7,
                    thickness=15
                ),
                cmin=0,
                cmax=100
            ),
            line=dict(width=4, color='rgba(100,100,100,0.8)'),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            name="Road Health"
        ))
        
        # Calculate center point for map
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Configure mapbox layout for proper map display
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=15
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            title=f"Road Health Map - Overall Score: {scoring_data['overall_score']:.1f}/100"
        )
        
        return fig
    
    def create_score_breakdown_chart(self, scoring_data: Dict) -> go.Figure:
        """
        Create comprehensive breakdown charts of road health components.
        
        Args:
            scoring_data: Road scoring results
            
        Returns:
            Plotly Figure with multiple subplots
        """
        
        breakdown = scoring_data['breakdown']
        
        # Create subplot with 2x2 layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Health Distribution',
                'Issue Detection Counts', 
                'Score Progression Along Road',
                'Average Problem Probabilities'
            ),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Health category distribution (pie chart)
        health_counts = {}
        for segment in scoring_data['segments']:
            score = segment['score']
            category = self._score_to_category(score)
            health_counts[category] = health_counts.get(category, 0) + 1
        
        if health_counts:
            fig.add_trace(
                go.Pie(
                    labels=list(health_counts.keys()),
                    values=list(health_counts.values()),
                    marker_colors=[self.health_colors[cat] for cat in health_counts.keys()],
                    textinfo='label+percent'
                ),
                row=1, col=1
            )
        
        # 2. Issue detection counts (bar chart)
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
                marker_color=['#ff4444', '#ffaa44', '#4444ff'],
                text=issue_counts,
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. Score progression along road (line chart)
        sequences = [s['sequence'] for s in scoring_data['segments']]
        scores = [s['score'] for s in scoring_data['segments']]
        
        fig.add_trace(
            go.Scatter(
                x=sequences,
                y=scores,
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=6),
                name='Health Score'
            ),
            row=2, col=1
        )
        
        # 4. Average problem probabilities (bar chart)
        avg_probs = [
            breakdown['average_damage_prob'],
            breakdown['average_occlusion_prob'],
            breakdown['average_crop_prob']
        ]
        
        fig.add_trace(
            go.Bar(
                x=['Damage', 'Occlusion', 'Crop'],
                y=avg_probs,
                marker_color=['#ff4444', '#ffaa44', '#4444ff'],
                text=[f"{p:.3f}" for p in avg_probs],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text=f"Road Health Analysis - {breakdown['total_segments']} Segments"
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Segment Number", row=2, col=1)
        fig.update_yaxes(title_text="Health Score", row=2, col=1)
        fig.update_yaxes(title_text="Detection Count", row=1, col=2)
        fig.update_yaxes(title_text="Average Probability", row=2, col=2)
        
        return fig
    
    def create_damage_heatmap(self, road_data: Dict) -> go.Figure:
        """
        Create heatmap visualization focused on damage probability.
        
        Args:
            road_data: Processed road data dictionary
            
        Returns:
            Plotly Figure with damage heatmap
        """
        
        coordinates = road_data['coordinates']
        if not coordinates or len(coordinates) == 0:
            fig = go.Figure()
            fig.update_layout(title="No road data available")
            return fig
        
        # Validate coordinates
        valid_coordinates = []
        valid_indices = []
        for i, coord in enumerate(coordinates):
            lat, lon = coord
            if self._is_valid_coordinate(lat, lon):
                valid_coordinates.append(coord)
                valid_indices.append(i)
        
        if not valid_coordinates:
            fig = go.Figure()
            fig.update_layout(title="No valid coordinates for heatmap")
            return fig
        
        lats = [coord[0] for coord in valid_coordinates]
        lons = [coord[1] for coord in valid_coordinates]
        
        # Extract damage probabilities for heatmap intensity (only valid indices)
        damage_probs = [road_data['predictions']['damage'][i]['probability'] for i in valid_indices if i < len(road_data['predictions']['damage'])]
        
        # Create hover text
        hover_texts = []
        for j, prob in enumerate(damage_probs):
            original_index = valid_indices[j]
            image_info = road_data['images'][original_index]
            hover_text = f"""
            <b>Segment {original_index}</b><br>
            Damage Probability: {prob:.3f}<br>
            Prediction: {'DAMAGE' if prob > 0.5 else 'NO DAMAGE'}<br>
            File: {image_info['filename']}
            """.strip()
            hover_texts.append(hover_text)
        
        fig = go.Figure()
        
        # Add damage probability heatmap
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=dict(
                size=15,
                color=damage_probs,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(
                    title=dict(text="Damage Probability"),
                    len=0.7,
                    thickness=15
                ),
                cmin=0,
                cmax=1,
                opacity=0.8
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            name="Damage Heatmap"
        ))
        
        # Add road path as reference line
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='lines',
            line=dict(width=3, color='blue'),
            opacity=0.5,
            name="Road Path",
            showlegend=False
        ))
        
        # Calculate center
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=15
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            title="Road Damage Probability Heatmap"
        )
        
        return fig
    
    def create_segment_details_table(self, road_data: Dict, scoring_data: Dict, max_rows: int = 20) -> List[Dict]:
        """
        Create formatted data for segment details table.
        
        Args:
            road_data: Processed road data dictionary
            scoring_data: Road scoring results
            max_rows: Maximum number of rows to return
            
        Returns:
            List of dictionaries for table display
        """
        
        table_data = []
        segments_to_show = min(max_rows, len(scoring_data['segments']))
        
        for i in range(segments_to_show):
            segment = scoring_data['segments'][i]
            image_info = road_data['images'][i]
            
            # Health status with color indicator
            health = self._score_to_category(segment['score'])
            
            table_data.append({
                'Sequence': f"{segment['sequence']:03d}",
                'Filename': image_info['filename'],
                'Score': f"{segment['score']:.1f}/100",
                'Health': health,
                'Damage': f"{segment['damage_prob']:.3f}",
                'Occlusion': f"{segment['occlusion_prob']:.3f}",
                'Crop': f"{segment['crop_prob']:.3f}",
                'Latitude': f"{image_info['latitude']:.6f}",
                'Longitude': f"{image_info['longitude']:.6f}"
            })
        
        return table_data
    
    def _score_to_category(self, score: float) -> str:
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
    
    def create_summary_metrics(self, scoring_data: Dict) -> Dict:
        """
        Create summary metrics for dashboard display.
        
        Args:
            scoring_data: Road scoring results
            
        Returns:
            Dictionary with formatted metrics
        """
        
        breakdown = scoring_data['breakdown']
        
        # Calculate percentages
        total_segments = breakdown['total_segments']
        damage_pct = (breakdown['damage_segments'] / total_segments * 100) if total_segments > 0 else 0
        
        # Determine road status
        overall_score = scoring_data['overall_score']
        if overall_score >= 80:
            status = "🟢 Good Condition"
            status_color = "normal"
        elif overall_score >= 60:
            status = "🟡 Fair Condition" 
            status_color = "normal"
        elif overall_score >= 40:
            status = "🟠 Poor Condition"
            status_color = "inverse"
        else:
            status = "🔴 Critical Condition"
            status_color = "off"
        
        return {
            'overall_score': {
                'value': f"{overall_score:.1f}/100",
                'delta': status,
                'color': status_color
            },
            'total_segments': {
                'value': total_segments,
                'label': "Road Segments"
            },
            'damage_detected': {
                'value': f"{damage_pct:.1f}%",
                'delta': f"{breakdown['damage_segments']} segments",
                'label': "Damage Detected"
            },
            'avg_damage_confidence': {
                'value': f"{breakdown['average_damage_prob']:.3f}",
                'label': "Avg Damage Confidence"
            }
        }
    
    def display_segment_details(self, road_data: Dict, segment_idx: int, uploaded_files, segment_cache=None):
        """
        Display detailed view of a specific road segment using simple, fast approach.
        
        Args:
            road_data: Processed road data dictionary
            segment_idx: Index of the segment to display
            uploaded_files: Original uploaded files for image access
            segment_cache: SegmentCache instance for pre-generated visualizations (optional)
        """
        import streamlit as st
        import logging
        import time
        
        logger = logging.getLogger(__name__)
        start_time = time.time()
        
        logger.info(f"Displaying segment {segment_idx}")
        
        # Use simple, direct approach (faster and more reliable)
        from .simple_segment_display import SimpleSegmentDisplay
        simple_display = SimpleSegmentDisplay()
        simple_display.display_segment_details(road_data, segment_idx, uploaded_files)
        
        display_time = time.time() - start_time
        logger.info(f"Segment {segment_idx} displayed in {display_time:.3f}s")
    
    def _display_cached_segment_details(self, segment_idx: int, segment_cache):
        """Display segment details using cached visualizations."""
        import streamlit as st
        from PIL import Image
        import time
        import logging
        
        logger = logging.getLogger(__name__)
        start_time = time.time()
        
        logger.info(f"Loading cached segment {segment_idx}")
        
        # Get cached segment data
        segment_data = segment_cache.get_segment_data(segment_idx)
        if not segment_data:
            logger.error(f"No cached data available for segment {segment_idx}")
            st.error(f"No cached data available for segment {segment_idx}")
            return
        
        if 'error' in segment_data:
            logger.error(f"Error in segment data: {segment_data['error']}")
            st.error(f"Error in segment data: {segment_data['error']}")
            return
        
        logger.info(f"Cached data loaded for segment {segment_idx} in {time.time() - start_time:.3f}s")
        
        image_info = segment_data['image_info']
        
        # Display segment info  
        st.subheader(f"🔍 Segment {image_info['sequence']:03d} Details")
        
        # Create columns for different visualizations
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original Image**")
            st.image(segment_data['original_image'], 
                    caption=f"Segment {image_info['sequence']:03d}", 
                    use_container_width=True)
        
        with col2:
            st.markdown("**Grad-CAM Visualization**")
            if segment_data.get('gradcam_path') and segment_data['gradcam_path'].exists():
                st.image(str(segment_data['gradcam_path']), 
                        caption="Damage Attention Map",
                        use_container_width=True)
            else:
                st.info("Grad-CAM data not available for this segment")
        
        with col3:
            st.markdown("**Road Mask**")
            if segment_data.get('mask_path') and segment_data['mask_path'].exists():
                st.image(str(segment_data['mask_path']), 
                        caption="Road Segmentation",
                        use_container_width=True)
                
                # Display mask statistics
                mask_stats = segment_data.get('mask_stats', {})
                if mask_stats:
                    coverage = mask_stats.get('road_coverage', 0) * 100
                    st.caption(f"🛣️ Road Coverage: {coverage:.1f}%")
            else:
                st.info("Road mask not available for this segment")
        
        # Display predictions
        st.markdown("**Predictions**")
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        predictions = segment_data['predictions']
        
        with pred_col1:
            damage_pred = predictions['damage']
            color = "🔴" if damage_pred['prediction'] else "🟢"
            st.metric(
                f"{color} Damage",
                f"{damage_pred['probability']:.3f}",
                delta="DETECTED" if damage_pred['prediction'] else "None"
            )
        
        with pred_col2:
            occlusion_pred = predictions['occlusion']
            color = "🟡" if occlusion_pred['prediction'] else "🟢"
            st.metric(
                f"{color} Occlusion", 
                f"{occlusion_pred['probability']:.3f}",
                delta="DETECTED" if occlusion_pred['prediction'] else "None"
            )
        
        with pred_col3:
            crop_pred = predictions['crop']
            color = "🟠" if crop_pred['prediction'] else "🟢"
            st.metric(
                f"{color} Crop Issues",
                f"{crop_pred['probability']:.3f}",
                delta="DETECTED" if crop_pred['prediction'] else "None"
            )
        
        # Display coordinates
        st.markdown("**Location Information**")
        coord_col1, coord_col2 = st.columns(2)
        
        with coord_col1:
            st.info(f"📍 Latitude: {image_info['latitude']:.6f}")
        
        with coord_col2:
            st.info(f"📍 Longitude: {image_info['longitude']:.6f}")
    
    def _display_original_segment_details(self, road_data: Dict, segment_idx: int, uploaded_files):
        """Original segment display method as fallback."""
        import streamlit as st
        from PIL import Image
        import matplotlib.pyplot as plt
        import numpy as np
        
        if segment_idx >= len(road_data['images']):
            st.error("Invalid segment index")
            return
        
        # Get segment data
        image_info = road_data['images'][segment_idx]
        gradcam_info = None
        
        # Find corresponding grad-CAM data
        for gc_data in road_data['gradcam_data']:
            if gc_data['sequence'] == image_info['sequence']:
                gradcam_info = gc_data
                break
        
        # Find uploaded file
        uploaded_file = None
        for file in uploaded_files:
            if file.name == image_info['filename']:
                uploaded_file = file
                break
        
        if not uploaded_file:
            st.error(f"Could not find uploaded file: {image_info['filename']}")
            return
        
        # Create clean container for segment display
        with st.container():
            # Display segment info
            st.subheader(f"🔍 Segment {image_info['sequence']:03d} Details")
            
            # Load original image once for all visualizations
            original_image = Image.open(uploaded_file)
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            
            # Create columns for different visualizations
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(original_image, caption=f"Segment {image_info['sequence']:03d}", 
                        use_container_width=True)
            
            with col2:
                st.markdown("**Grad-CAM Visualization**")
                
                if gradcam_info and gradcam_info['gradcam'] is not None:
                    # Display Grad-CAM overlay
                    gradcam_data = gradcam_info['gradcam']
                    
                    # Create figure for grad-CAM
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(np.array(original_image))
                    ax.imshow(gradcam_data, alpha=0.6, cmap='jet')
                    ax.set_title('Damage Attention Map', fontsize=10)
                    ax.axis('off')
                    
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Grad-CAM data not available for this segment")
            
            with col3:
                st.markdown("**Road Mask**")
                
                # Check if we have mask data for this segment (stored in gradcam_data for simplicity)
                mask_data = None
                mask_overlay = None
                mask_stats = {}
                
                # Try to get mask data from the road_data if available
                if 'mask_data' in road_data and segment_idx < len(road_data['mask_data']):
                    mask_info = road_data['mask_data'][segment_idx]
                    mask_data = mask_info.get('mask')
                    mask_overlay = mask_info.get('mask_overlay')
                    mask_stats = mask_info.get('mask_stats', {})
                
                if mask_overlay is not None:
                    # Display mask overlay
                    st.image(mask_overlay, caption="Road Segmentation", 
                            use_container_width=True)
                    
                    # Display mask statistics
                    if mask_stats:
                        coverage = mask_stats.get('road_coverage', 0) * 100
                        st.caption(f"🛣️ Road Coverage: {coverage:.1f}%")
                elif mask_data is not None:
                    # Display binary mask if overlay not available
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(np.array(original_image))
                    ax.imshow(mask_data, alpha=0.4, cmap='Greens')
                    ax.set_title('Road Segmentation', fontsize=10)
                    ax.axis('off')
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    if mask_stats:
                        coverage = mask_stats.get('road_coverage', 0) * 100
                        st.caption(f"🛣️ Road Coverage: {coverage:.1f}%")
                else:
                    st.info("Road mask not available for this segment")
            
            # Display predictions
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
            
            # Display coordinates
            st.markdown("**Location Information**")
            coord_col1, coord_col2 = st.columns(2)
            
            with coord_col1:
                st.info(f"📍 Latitude: {image_info['latitude']:.6f}")
            
            with coord_col2:
                st.info(f"📍 Longitude: {image_info['longitude']:.6f}")
