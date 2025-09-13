#!/usr/bin/env python3
"""
Streamlit Web UI for Road Distress Classification
Date: 2025-08-01

A user-friendly web interface for the road distress inference pipeline.
Upload images and get instant damage analysis with confidence heatmaps.
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import io
import time
import json
import zipfile
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import cv2
import numpy as np
import re
import logging

# Configure logging for debugging
logging.basicConfig(
    level=logging.DEBUG,  # Enable debug level for detailed info
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import our pipeline components
try:
    from src import EnsembleInferenceEngine, HeatmapGenerator, RoadProcessor, RoadVisualizer, RoadMaskGenerator, SegmentCache
except ImportError as e:
    st.error(f"Failed to import inference components: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Road Distress Classifier",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .damage-detected {
        background: linear-gradient(145deg, #fef2f2 0%, #fecaca 20%, #ffffff 100%);
        border: 2px solid #f87171;
        color: #dc2626;
    }
    
    .damage-detected h3 {
        color: #dc2626 !important;
        font-weight: 600;
    }
    
    .damage-detected h2 {
        color: #b91c1c !important;
        font-weight: 700;
    }
    
    .no-damage {
        background: linear-gradient(145deg, #f0fdf4 0%, #bbf7d0 20%, #ffffff 100%);
        border: 2px solid #4ade80;
        color: #16a34a;
    }
    
    .no-damage h3 {
        color: #16a34a !important;
        font-weight: 600;
    }
    
    .no-damage h2 {
        color: #15803d !important;
        font-weight: 700;
    }
    
    .metric-card h3 {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #374151;
    }
    
    .metric-card h2 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: #1f2937;
    }
    
    .metric-card h4 {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0.5rem 0;
        color: #374151;
    }
    
    .metric-card p {
        font-size: 0.9rem;
        color: #6b7280;
        margin: 0;
    }
    
    .info-panel {
        background: linear-gradient(145deg, #1f2937 0%, #374151 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        color: #ffffff;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .info-panel h4 {
        color: #f3f4f6 !important;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    .info-panel p {
        color: #d1d5db !important;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    
    .info-panel strong {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
    }
    
    .interpretation-guide {
        background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #3b82f6;
        margin: 1rem 0;
        color: #ffffff;
        box-shadow: 0 8px 25px -5px rgba(59, 130, 246, 0.3);
    }
    
    .interpretation-guide h4 {
        color: #60a5fa !important;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .interpretation-guide p {
        color: #e2e8f0 !important;
        margin-bottom: 0.8rem;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    .interpretation-guide strong {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .color-legend {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 4px;
        margin-right: 8px;
        vertical-align: middle;
    }
    
    .legend-red { background: linear-gradient(45deg, #ef4444, #dc2626); }
    .legend-yellow { background: linear-gradient(45deg, #f59e0b, #d97706); }
    .legend-blue { background: linear-gradient(45deg, #3b82f6, #2563eb); }
    
    /* Analysis card styling for better visibility */
    .analysis-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        color: #f8fafc !important;
    }
    
    /* Fix Streamlit metrics visibility in dark cards */
    .analysis-card .stMetric {
        background: rgba(15, 23, 42, 0.7) !important;
        border-radius: 8px;
        padding: 12px !important;
        margin: 8px 0 !important;
        border: 1px solid #475569;
    }
    
    .analysis-card .stMetric label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    
    .analysis-card .stMetric [data-testid="metric-container"] > div {
        color: #ffffff !important;
        font-weight: bold !important;
    }
    
    .analysis-card .stMetric [data-testid="metric-container"] div[data-testid] {
        color: #ffffff !important;
    }
    
    /* Tab styling improvements */
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9 !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        color: #374151 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 700 !important;
    }
    
    /* Ensure all text in analysis sections is visible */
    .analysis-card h1, .analysis-card h2, .analysis-card h3, .analysis-card h4, .analysis-card h5, .analysis-card h6 {
        color: #f8fafc !important;
    }
    
    .analysis-card p, .analysis-card span, .analysis-card div {
        color: #e2e8f0 !important;
    }
    
    .analysis-card strong {
        color: #ffffff !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

@st.cache_resource
def load_ensemble_engine(_cache_buster="v3"):
    """Load the ensemble inference engine (cached for performance)."""
    try:
        with st.spinner("Loading Multi-Model Ensemble (Model B + Model H)..."):
            engine = EnsembleInferenceEngine()
            heatmap_gen = HeatmapGenerator()
        return engine, heatmap_gen
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.error("**Model checkpoint loading failed!**")
        
        # Provide helpful troubleshooting
        st.markdown("""
        **🔧 Troubleshooting Steps:**
        
        1. **Check if checkpoint exists:**
           ```bash
           python check_model.py
           ```
        
        2. **Verify experiments path:**
           - Expected: `../experiments/2025-07-05_hybrid_training/results/model_b/checkpoints/best_model.pth`
           - Make sure the training experiment completed successfully
        
        3. **PyTorch version compatibility:**
           - This error often occurs with PyTorch 2.6+ security changes
           - The model loader has been updated to handle this automatically
        
        4. **Alternative checkpoints:**
           - Try using `checkpoint_epoch_021.pth` (best epoch) instead of `best_model.pth`
        """)
        
        # Show a button to run the checker
        if st.button("🔍 Run Model Checker"):
            st.code("python check_model.py", language="bash")
            st.info("Run this command in your terminal to diagnose the issue")
        
        return None, None

def display_results_metrics(results):
    """Display prediction results in a nice metric format."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        damage_result = results['class_results']['damage']
        if damage_result['prediction']:
            st.markdown(f"""
            <div class="metric-card damage-detected">
                <h3>🚨 DAMAGE DETECTED</h3>
                <h2>{damage_result['probability']:.1%}</h2>
                <p>Confidence: {damage_result['confidence']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card no-damage">
                <h3>✅ NO DAMAGE</h3>
                <h2>{damage_result['probability']:.1%}</h2>
                <p>Confidence: {damage_result['confidence']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        occlusion_result = results['class_results']['occlusion']
        status = "🟡 DETECTED" if occlusion_result['prediction'] else "✅ CLEAR"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Occlusion</h3>
            <h4>{status}</h4>
            <p>Probability: {occlusion_result['probability']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        crop_result = results['class_results']['crop']
        status = "🔵 DETECTED" if crop_result['prediction'] else "✅ GOOD"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Crop Issues</h3>
            <h4>{status}</h4>
            <p>Probability: {crop_result['probability']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Overall Confidence</h3>
            <h2>{results['overall_confidence']:.1%}</h2>
            <p>Inference: {results['inference_time']:.3f}s</p>
        </div>
        """, unsafe_allow_html=True)

def create_probability_chart(results):
    """Create an interactive probability chart."""
    classes = ['Damage', 'Occlusion', 'Crop']
    probabilities = [
        results['class_results']['damage']['probability'],
        results['class_results']['occlusion']['probability'],
        results['class_results']['crop']['probability']
    ]
    predictions = [
        results['class_results']['damage']['prediction'],
        results['class_results']['occlusion']['prediction'],
        results['class_results']['crop']['prediction']
    ]
    
    colors = ['red' if pred else 'lightblue' for pred in predictions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probabilities,
            marker_color=colors,
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Class Probabilities",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                  annotation_text="Decision Threshold (50%)")
    
    return fig

def process_multi_model_image(uploaded_file, engine, heatmap_gen, thresholds, display_scale=1.0, heatmap_method='gradcam', gradcam_class='damage', gradcam_model='combined'):
    """Process a single uploaded image with multi-model ensemble."""
    if uploaded_file is None:
        return None
    
    # Update engine thresholds
    engine.update_thresholds(thresholds)
    
    # Load image
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Display original image info
    st.subheader("📷 Original Image")
    
    # Create layout with image and info side by side
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Apply scaling if requested
        display_image = image
        if display_scale != 1.0:
            new_size = (int(image.size[0] * display_scale), int(image.size[1] * display_scale))
            display_image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Display image in container with modern styling
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(display_image, caption=f"📁 {uploaded_file.name}", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Modern info panel
        st.markdown(f"""
        <div class="info-panel">
            <h4>📊 Image Details</h4>
            <p><strong>Original Size:</strong><br>{image.size[0]} × {image.size[1]} px</p>
            <p><strong>Display Scale:</strong><br>{display_scale:.1f}x</p>
            <p><strong>Format:</strong><br>{image.format or 'Unknown'}</p>
            <p><strong>Color Mode:</strong><br>{image.mode}</p>
            <p><strong>File Size:</strong><br>{len(uploaded_file.getvalue()) / 1024:.1f} KB</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Run multi-model ensemble inference
    st.subheader("🤖 Multi-Model Analysis Pipeline")
    
    # Step 1: Model B Analysis
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown("### 🅱️ Step 1: Model B Analysis")
    with st.spinner("Running Model B inference..."):
        model_b_results = engine.predict_single_model(np.array(image), 'model_b')
    
    # Display Model B results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🅱️ Model B Predictions:**")
        for class_name in ['damage', 'occlusion', 'crop']:
            result = model_b_results['class_results'][class_name]
            prob = result['probability']
            pred = result['prediction']
            thresh = result['threshold']
            
            # Color based on prediction
            color = "🔴" if pred else "🟢"
            status = "DETECTED" if pred else "NOT DETECTED"
            
            st.markdown(f"""
            <div style="padding: 12px; margin: 8px 0; border-radius: 8px; background: {'rgba(239, 68, 68, 0.1)' if pred else 'rgba(34, 197, 94, 0.1)'}; border: 1px solid {'#ef4444' if pred else '#22c55e'}; color: #f8fafc;">
                {color} <strong style="color: #ffffff;">{class_name.upper()}:</strong> <span style="color: #ffffff; font-weight: bold;">{prob:.1%}</span> <span style="color: #e2e8f0;">({status})</span><br>
                <small style="color: #cbd5e1;">Threshold: {thresh:.1%}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Model B probability chart
        prob_chart_b = create_probability_chart(model_b_results)
        prob_chart_b.update_layout(title="🅱️ Model B Probabilities", height=300)
        st.plotly_chart(prob_chart_b, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 2: Model H Analysis
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown("### 🅷 Step 2: Model H Analysis")
    with st.spinner("Running Model H inference..."):
        model_h_results = engine.predict_single_model(np.array(image), 'model_h')
    
    # Display Model H results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🅷 Model H Predictions:**")
        for class_name in ['damage', 'occlusion', 'crop']:
            result = model_h_results['class_results'][class_name]
            prob = result['probability']
            pred = result['prediction']
            thresh = result['threshold']
            
            # Color based on prediction
            color = "🔴" if pred else "🟢"
            status = "DETECTED" if pred else "NOT DETECTED"
            
            st.markdown(f"""
            <div style="padding: 12px; margin: 8px 0; border-radius: 8px; background: {'rgba(239, 68, 68, 0.1)' if pred else 'rgba(34, 197, 94, 0.1)'}; border: 1px solid {'#ef4444' if pred else '#22c55e'}; color: #f8fafc;">
                {color} <strong style="color: #ffffff;">{class_name.upper()}:</strong> <span style="color: #ffffff; font-weight: bold;">{prob:.1%}</span> <span style="color: #e2e8f0;">({status})</span><br>
                <small style="color: #cbd5e1;">Threshold: {thresh:.1%}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Model H probability chart
        prob_chart_h = create_probability_chart(model_h_results)
        prob_chart_h.update_layout(title="🅷 Model H Probabilities", height=300)
        st.plotly_chart(prob_chart_h, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 3: Ensemble Analysis
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Step 3: Ensemble Decision")
    with st.spinner("Computing ensemble predictions..."):
        ensemble_results = engine.predict_ensemble(np.array(image))
    
    # Display ensemble results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Ensemble Final Decision:**")
        for class_name in ['damage', 'occlusion', 'crop']:
            result = ensemble_results['ensemble_results']['class_results'][class_name]
            prob = result['probability']
            pred = result['prediction']
            thresh = result['threshold']
            
            # Color based on prediction
            color = "🔴" if pred else "🟢"
            status = "DETECTED" if pred else "NOT DETECTED"
            
            st.markdown(f"""
            <div style="padding: 12px; margin: 8px 0; border-radius: 8px; background: {'rgba(239, 68, 68, 0.2)' if pred else 'rgba(34, 197, 94, 0.2)'}; border: 2px solid {'#ef4444' if pred else '#22c55e'}; color: #f8fafc;">
                <strong style="color: #ffffff;">{color} {class_name.upper()}: <span style="font-size: 1.1em;">{prob:.1%}</span> ({status})</strong><br>
                <small style="color: #cbd5e1;">Combined from both models | Threshold: {thresh:.1%}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Ensemble probability chart
        prob_chart_ensemble = create_probability_chart({'class_results': ensemble_results['ensemble_results']['class_results']})
        prob_chart_ensemble.update_layout(title="🎯 Ensemble Final Probabilities", height=300)
        st.plotly_chart(prob_chart_ensemble, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Comparison Table
    st.subheader("📊 Model Comparison")
    
    comparison_data = []
    for class_name in ['damage', 'occlusion', 'crop']:
        model_b_prob = model_b_results['class_results'][class_name]['probability']
        model_h_prob = model_h_results['class_results'][class_name]['probability']
        ensemble_prob = ensemble_results['ensemble_results']['class_results'][class_name]['probability']
        
        model_b_pred = model_b_results['class_results'][class_name]['prediction']
        model_h_pred = model_h_results['class_results'][class_name]['prediction']
        ensemble_pred = ensemble_results['ensemble_results']['class_results'][class_name]['prediction']
        
        comparison_data.append({
            'Class': class_name.capitalize(),
            'Model B Prob': f"{model_b_prob:.1%}",
            'Model B Pred': "✅" if model_b_pred else "❌",
            'Model H Prob': f"{model_h_prob:.1%}",
            'Model H Pred': "✅" if model_h_pred else "❌",
            'Ensemble Prob': f"{ensemble_prob:.1%}",
            'Final Decision': "🔴 DETECTED" if ensemble_pred else "🟢 CLEAR"
        })
    
    st.dataframe(comparison_data, use_container_width=True)
    

    
    # Generate visualizations
    st.subheader("🎨 Damage Analysis Visualizations")
    
    # Show spinner messages for Grad-CAM methods
    if heatmap_method == "gradcam" and gradcam_class:
        class_emoji = {"damage": "🔥", "occlusion": "🌿", "crop": "✂️"}
        emoji = class_emoji.get(gradcam_class, "🔥")
        model_emoji = {"model_b": "🅱️", "model_h": "🅷", "combined": "🎯"}
        model_text = model_emoji.get(gradcam_model, "🤖")
        spinner_text = f"Generating Grad-CAM for {gradcam_class.upper()} {emoji} using {model_text} {gradcam_model.replace('_', ' ').title()}..."
    elif heatmap_method == "gradcam_all":
        spinner_text = "Generating multi-class Grad-CAM visualization... 🌈"
    else:
        spinner_text = "Generating visualization... 🎨"
    
    with st.spinner(spinner_text):
        # Get confidence map using selected method
        try:
            if heatmap_method == "gradcam" and gradcam_class:
                confidence_map, _ = engine.get_damage_confidence_map(np.array(image), method=heatmap_method, target_class=gradcam_class, model_name=gradcam_model)
            else:
                confidence_map, _ = engine.get_damage_confidence_map(np.array(image), method=heatmap_method, model_name=gradcam_model)
        except TypeError:
            # Fallback for cached engine without method parameter
            st.warning("Using fallback heatmap method. Please refresh the page to use advanced methods.")
            confidence_map, _ = engine.get_damage_confidence_map(np.array(image))
        
        # Create clean visualizations without text overlays
        clean_heatmap = heatmap_gen.create_clean_heatmap(
            np.array(ensemble_results['resized_image']), confidence_map, scale_factor=display_scale
        )
        
        pure_confidence = heatmap_gen.create_pure_confidence_map(
            confidence_map, scale_factor=display_scale
        )
        
        # Create original scaled image for comparison
        original_scaled = np.array(ensemble_results['resized_image'])
        if display_scale != 1.0:
            new_height = int(original_scaled.shape[0] * display_scale)
            new_width = int(original_scaled.shape[1] * display_scale)
            original_scaled = cv2.resize(original_scaled, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Display visualizations in a modern layout
    if heatmap_method == "gradcam" and gradcam_class:
        class_emoji = {"damage": "🔥", "occlusion": "🌿", "crop": "✂️"}
        emoji = class_emoji.get(gradcam_class, "🔥")
        model_emoji = {"model_b": "🅱️", "model_h": "🅷", "combined": "🎯"}
        model_text = model_emoji.get(gradcam_model, "🤖")
        model_display = gradcam_model.replace('_', ' ').title()
        tab_title = f"{emoji} {gradcam_class.capitalize()} Grad-CAM ({model_text} {model_display})"
        caption_text = f"{emoji} Grad-CAM: {gradcam_class.capitalize()} Attention from {model_text} {model_display}"
    else:
        tab_title = "🔥 Heatmap Overlay"
        caption_text = "🔥 Clean Confidence Heatmap"
    
    tab1, tab2, tab3 = st.tabs([tab_title, "🌡️ Pure Confidence", "🔍 Side by Side"])
    
    with tab1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(clean_heatmap, caption=caption_text, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add class-specific interpretation guide
        if heatmap_method == "gradcam" and gradcam_class:
            class_descriptions = {
                "damage": {
                    "emoji": "🔥",
                    "focus": "road cracks, potholes, and surface damage",
                    "tip": "Red areas show where the model detected damage features"
                },
                "occlusion": {
                    "emoji": "🌿", 
                    "focus": "vegetation, objects blocking road view",
                    "tip": "Red areas show where the model detected occlusion features"
                },
                "crop": {
                    "emoji": "✂️",
                    "focus": "image cropping and framing issues", 
                    "tip": "Red areas show where the model detected crop-related features"
                }
            }
            
            desc = class_descriptions[gradcam_class]
            st.markdown(f"""
            <div class="interpretation-guide">
                <h4>{desc['emoji']} How to Interpret {gradcam_class.capitalize()} Grad-CAM</h4>
                <p><span class="color-legend legend-red"></span><strong>Red/Hot Areas:</strong> High attention for {desc['focus']}</p>
                <p><span class="color-legend legend-yellow"></span><strong>Yellow Areas:</strong> Moderate attention for {gradcam_class}</p>
                <p><span class="color-legend legend-blue"></span><strong>Blue/Cool Areas:</strong> Low attention for {gradcam_class}</p>
                <p style="margin-top: 1rem; font-style: italic; color: #94a3b8;">
                    💡 {desc['tip']} (Real model attention!)
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="interpretation-guide">
                <h4>🎯 How to Interpret Heatmap</h4>
                <p><span class="color-legend legend-red"></span><strong>Red/Hot Areas:</strong> High confidence/probability</p>
                <p><span class="color-legend legend-yellow"></span><strong>Yellow Areas:</strong> Moderate confidence/probability</p>
                <p><span class="color-legend legend-blue"></span><strong>Blue/Cool Areas:</strong> Low confidence/probability</p>
                <p style="margin-top: 1rem; font-style: italic; color: #94a3b8;">
                    💡 Tip: Focus on red areas for key model predictions
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(pure_confidence, caption="🌡️ Pure Confidence Map", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="interpretation-guide">
            <h4>📊 Pure Confidence Map</h4>
            <p><strong>What you're seeing:</strong> Model confidence without image overlay</p>
            <p><strong>Warmer colors:</strong> Higher confidence in damage detection</p>
            <p><strong>Cooler colors:</strong> Lower confidence in damage detection</p>
            <p style="margin-top: 1rem; font-style: italic; color: #94a3b8;">
                💡 This view helps identify the most confident predictions
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        # Side by side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(original_scaled, caption="📷 Original Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(clean_heatmap, caption="🔥 Damage Heatmap", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'model_b_results': model_b_results,
        'model_h_results': model_h_results,
        'ensemble_results': ensemble_results,
        'visualizations': {
            'clean_heatmap': clean_heatmap,
            'pure_confidence': pure_confidence,
            'original_scaled': original_scaled
        }
    }

def process_batch_images(uploaded_files, engine, heatmap_gen, thresholds):
    """Process multiple uploaded images with multi-model ensemble."""
    if not uploaded_files:
        return None
    
    # Update engine thresholds
    engine.update_thresholds(thresholds)
    
    st.subheader(f"📁 Batch Processing ({len(uploaded_files)} images)")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = []
    all_visualizations = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
        
        try:
            # Load and process image
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Run multi-model inference
            model_b_results = engine.predict_single_model(np.array(image), 'model_b')
            model_h_results = engine.predict_single_model(np.array(image), 'model_h')
            ensemble_results = engine.predict_ensemble(np.array(image))
            
            # Package results for batch processing
            results = {
                'filename': uploaded_file.name,
                'model_b_results': model_b_results,
                'model_h_results': model_h_results,
                'ensemble_results': ensemble_results,
                'resized_image': ensemble_results['resized_image']  # Use ensemble resized image
            }
            all_results.append(results)
            
            # Generate key visualization using ensemble results
            confidence_map, _ = engine.get_damage_confidence_map(np.array(image), method='gradcam', target_class='damage', model_name='combined')
            damage_heatmap = heatmap_gen.create_clean_heatmap(
                np.array(ensemble_results['resized_image']), confidence_map
            )
            all_visualizations.append((uploaded_file.name, damage_heatmap))
            
        except Exception as e:
            st.warning(f"Failed to process {uploaded_file.name}: {e}")
        
        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("✅ Batch processing complete!")
    
    if not all_results:
        st.error("No images were successfully processed.")
        return None
    
    # Create batch summary using ensemble results
    damage_count = sum(1 for r in all_results if r['ensemble_results']['ensemble_results']['class_results']['damage']['prediction'])
    occlusion_count = sum(1 for r in all_results if r['ensemble_results']['ensemble_results']['class_results']['occlusion']['prediction'])
    crop_count = sum(1 for r in all_results if r['ensemble_results']['ensemble_results']['class_results']['crop']['prediction'])
    
    avg_damage_prob = np.mean([r['ensemble_results']['ensemble_results']['class_results']['damage']['probability'] for r in all_results])
    avg_confidence = np.mean([r['ensemble_results']['ensemble_results']['overall_confidence'] for r in all_results])
    
    # Display batch summary
    st.subheader("📈 Batch Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Images Processed", len(all_results))
    with col2:
        st.metric("Damage Detected", damage_count, f"{damage_count/len(all_results)*100:.1f}%")
    with col3:
        st.metric("Avg Damage Prob", f"{avg_damage_prob:.1%}")
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Batch statistics chart
    categories = ['Damage', 'Occlusion', 'Crop']
    counts = [damage_count, occlusion_count, crop_count]
    percentages = [c/len(all_results)*100 for c in counts]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Detection Counts', 'Detection Percentages'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(x=categories, y=counts, name="Count", marker_color=['red', 'orange', 'blue']),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=categories, y=percentages, name="Percentage", marker_color=['red', 'orange', 'blue']),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display individual results
    st.subheader("🖼️ Individual Results")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📋 Results Table", "🎨 Heatmap Gallery"])
    
    with tab1:
        # Results table
        table_data = []
        for result in all_results:
            ensemble_data = result['ensemble_results']['ensemble_results']
            table_data.append({
                'Filename': result['filename'],
                'Damage': '🚨' if ensemble_data['class_results']['damage']['prediction'] else '✅',
                'Damage Prob': f"{ensemble_data['class_results']['damage']['probability']:.1%}",
                'Occlusion': '🟡' if ensemble_data['class_results']['occlusion']['prediction'] else '✅',
                'Crop': '🔵' if ensemble_data['class_results']['crop']['prediction'] else '✅',
                'Overall Confidence': f"{ensemble_data['overall_confidence']:.1%}"
            })
        
        st.dataframe(table_data, use_container_width=True)
    
    with tab2:
        # Heatmap gallery
        cols = st.columns(3)
        for i, (filename, heatmap) in enumerate(all_visualizations):
            with cols[i % 3]:
                st.image(heatmap, caption=filename, use_column_width=True)
    
    return all_results, all_visualizations

def process_road_folder(uploaded_files, engine, heatmap_gen, thresholds):
    """Process entire road folder and display comprehensive results."""
    
    # Initialize processors
    with st.spinner("Initializing components..."):
        mask_model_path = "../checkpoints/best_model.pth"
        mask_generator = RoadMaskGenerator(model_path=mask_model_path)
        segment_cache = SegmentCache()
        
    road_processor = RoadProcessor(engine, heatmap_gen, mask_generator)
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
            
            # Check if we have valid data
            if not road_data['coordinates'] or len(road_data['coordinates']) == 0:
                progress_bar.empty()
                status_text.empty()
                st.error("❌ No valid GPS coordinates found!")
                st.markdown("""
                **Possible issues:**
                - Filename format incorrect (should be: `XXX_latitude_longitude.ext`)
                - Coordinates out of valid range (lat: -90 to 90, lon: -180 to 180)
                - File names may have wrong coordinate order
                
                **Expected format:** `000_31.296905_-97.543646.png`
                - First number: sequence (000)
                - Second number: latitude (31.296905)
                - Third number: longitude (-97.543646)
                """)
                return
            
            # Calculate road score
            scoring_data = road_processor.calculate_road_score(road_data)
            
            progress_bar.empty()
            status_text.empty()
            
            # Store data in session state to persist across reruns
            st.session_state.road_data = road_data
            st.session_state.scoring_data = scoring_data
            st.session_state.uploaded_files_cache = uploaded_files
            
            # Display results using simple approach (no complex caching)
            display_road_results(road_data, scoring_data, road_visualizer, uploaded_files, None)
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Road processing failed: {e}")
            
            # Add debugging info
            with st.expander("🔧 Debugging Information"):
                st.write("**Upload Info:**")
                st.write(f"- Files uploaded: {len(uploaded_files)}")
                
                st.write("**Filename Analysis:**")
                for i, file in enumerate(uploaded_files[:5]):  # Show first 5 files
                    st.write(f"- {file.name}")
                
                if len(uploaded_files) > 5:
                    st.write(f"... and {len(uploaded_files) - 5} more files")
            return

def display_road_results(road_data, scoring_data, visualizer, uploaded_files, segment_cache=None):
    """Display comprehensive road analysis results with optional segment caching."""
    
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
    
    # Add segment selector for detailed analysis
    st.subheader("🔍 Segment Details")
    
    # Create segment selector 
    segment_options = [f"Segment {i:03d}" for i in range(len(road_data['images']))]
    
    # Initialize selected segment in session state if not exists
    if 'selected_segment' not in st.session_state:
        st.session_state.selected_segment = 0
    
    # Use a simple slider without any buttons that cause reruns
    clicked_segment = st.slider(
        "Navigate Segments:",
        min_value=0,
        max_value=len(segment_options) - 1,
        value=st.session_state.selected_segment,
        step=1,
        help=f"Drag to navigate segments (0-{len(segment_options)-1})",
        key="segment_slider"
    )
    
    # Update session state (no rerun needed - slider handles this automatically)
    st.session_state.selected_segment = int(clicked_segment)
    
    # Show current segment info with performance tips
    st.info(f"📍 Viewing: Segment {clicked_segment:03d} of {len(segment_options)} total segments")
    
    # Add performance tip
    st.caption("💡 **Tip**: Use the slider for smooth navigation between segments. Images are cached for faster switching!")
    
    # Add logging for segment selection
    logger = logging.getLogger(__name__)
    logger.info(f"User selected segment: {clicked_segment} (Segment {clicked_segment:03d})")
    
    # Use fragment to isolate segment display and prevent full page reruns
    @st.fragment
    def display_segment_fragment():
        """Fragment to display segment details without triggering full page rerun."""
        col1, col2 = st.columns([3, 2])
        
        with col1:
            logger.info(f"Starting display of segment {clicked_segment}")
            start_time = time.time()
            
            visualizer.display_segment_details(
                road_data, 
                clicked_segment, 
                uploaded_files,
                segment_cache
            )
            
            display_time = time.time() - start_time
            logger.info(f"Completed display of segment {clicked_segment} in {display_time:.3f}s")
        
        with col2:
            # Show segment location info
            if clicked_segment < len(road_data['coordinates']):
                segment_coord = road_data['coordinates'][clicked_segment]
                
                st.markdown("**📍 Segment Location**")
                
                # Location metrics
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    st.metric("Latitude", f"{segment_coord[0]:.6f}")
                with col2_2:
                    st.metric("Longitude", f"{segment_coord[1]:.6f}")
                
                # Health score for this segment
                if 'health_scores' in scoring_data:
                    segment_score = scoring_data['health_scores'][clicked_segment]
                    score_color = "normal" if segment_score >= 70 else "inverse" if segment_score >= 40 else "off"
                    
                    st.metric(
                        "Health Score",
                        f"{segment_score:.1f}/100", 
                        help=f"Individual score for segment {clicked_segment:03d}"
                    )
                
                # Road statistics
                st.markdown("**📊 Road Analysis Summary**")
                st.write(f"**Total Segments:** {len(road_data['images'])}")
                st.write(f"**Current Position:** {clicked_segment + 1} of {len(road_data['images'])}")
                
                progress_pct = (clicked_segment + 1) / len(road_data['images'])
                st.progress(progress_pct)
                st.caption(f"Progress: {progress_pct:.1%} through road")
    
    # Execute the fragment
    display_segment_fragment()
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["📊 Detailed Analysis", "🔥 Damage Heatmap", "📋 Segment Details"])
    
    with tab1:
        breakdown_chart = visualizer.create_score_breakdown_chart(scoring_data)
        st.plotly_chart(breakdown_chart, use_container_width=True)
    
    with tab2:
        damage_heatmap = visualizer.create_damage_heatmap(road_data)
        st.plotly_chart(damage_heatmap, use_container_width=True)
    
    with tab3:
        # Segment details table
        segment_data = visualizer.create_segment_details_table(road_data, scoring_data, max_rows=20)
        st.dataframe(segment_data, use_container_width=True)
        
        if len(scoring_data['segments']) > 20:
            st.info(f"Showing first 20 of {len(scoring_data['segments'])} segments")
    
    # Download section
    st.subheader("💾 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # Road report JSON
        road_report = {
            'road_data': convert_numpy_types(road_data),
            'scoring_data': convert_numpy_types(scoring_data),
            'metadata': {
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_images': len(road_data['images'])
            }
        }
        
        json_str = json.dumps(road_report, indent=2)
        st.download_button(
            label="📄 Download Road Report",
            data=json_str,
            file_name=f"road_analysis_{int(time.time())}.json",
            mime="application/json"
        )
    
    with col2:
        st.info("📊 Additional exports (maps, charts) available in full version")

def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🛣️ Multi-Model Road Distress Classifier</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by Ensemble** - Model B (80.6% F1) + Model H (78.1% F1) | EfficientNet-B3 | Advanced Decision Pipeline")
    
    # Load ensemble inference engine
    engine, heatmap_gen = load_ensemble_engine()
    if engine is None:
        st.stop()
    
    # Sidebar configuration
    st.sidebar.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    # Processing mode
    processing_mode = st.sidebar.radio(
        "Processing Mode",
        ["Single Image", "Batch Processing", "Road Folder Analysis"],
        help="Choose between analyzing one image, multiple images, or entire road folder"
    )
    
    # Image display options
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🖼️ Display Options**")
    
    # Image scaling
    display_scale = st.sidebar.slider(
        "Image Scale",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Scale factor for displaying images (1.0 = original size)"
    )
    
    # Heatmap opacity
    heatmap_alpha = st.sidebar.slider(
        "Heatmap Opacity",
        min_value=0.3,
        max_value=0.9,
        value=0.6,
        step=0.05,
        help="Transparency of heatmap overlay (higher = more visible)"
    )
    
    # Heatmap method selection (Grad-CAM only - the only real method!)
    heatmap_method = st.sidebar.selectbox(
        "Heatmap Visualization",
        ["gradcam", "gradcam_all"],
        index=0,
        help="Grad-CAM shows real model attention - the only method that reveals where your model is actually looking!"
    )
    
    # Class selection for Grad-CAM only (the only method that actually shows model attention)
    gradcam_class = None
    gradcam_model = 'combined'  # Default value
    if heatmap_method == "gradcam":
        gradcam_class = st.sidebar.selectbox(
            "Grad-CAM Target Class",
            ["damage", "occlusion", "crop"],
            index=0,
            help="Which class to visualize with Grad-CAM (shows real model attention)"
        )
        
        # Grad-CAM model selection
        gradcam_model = st.sidebar.selectbox(
            "🤖 Grad-CAM Model Source",
            options=['model_b', 'model_h', 'combined'],
            index=2,
            format_func=lambda x: {
                'model_b': '🅱️ Model B Only',
                'model_h': '🅷 Model H Only', 
                'combined': '🎯 Combined (Both Models)'
            }[x],
            help="Choose which model's attention to visualize"
        )
        
        class_emoji = {"damage": "🔥", "occlusion": "🌿", "crop": "✂️"}
        emoji = class_emoji.get(gradcam_class, "🔥")
        st.sidebar.markdown(f"**{emoji} Real Model Attention: {gradcam_class.upper()}**")
        
        st.sidebar.info("💡 **Only Grad-CAM shows real model attention!** Other methods create artificial patterns.")
        
        # Show model selection explanation
        if gradcam_model == 'model_b':
            st.sidebar.success("🅱️ **Model B**: Best performing model (No masks, no CLAHE)")
        elif gradcam_model == 'model_h':
            st.sidebar.success("🅷 **Model H**: Enhanced model (CLAHE + partial masks)")
        else:  # combined
            st.sidebar.success("🎯 **Combined**: Weighted average of both models' attention")
    
    # Per-class adjustable thresholds
    st.sidebar.subheader("⚖️ Decision Thresholds")
    
    damage_threshold = st.sidebar.slider(
        "🔥 Damage Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Probability threshold for damage detection (Balanced: P=0.54, R=0.66, Acc=0.79)"
    )
    
    occlusion_threshold = st.sidebar.slider(
        "🌿 Occlusion Threshold", 
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05,
        help="Probability threshold for occlusion detection (Balanced: P=0.80, R=0.75, Acc=0.93)"
    )
    
    crop_threshold = st.sidebar.slider(
        "✂️ Crop Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.49,
        step=0.05,
        help="Probability threshold for crop detection (Balanced: P=0.99, R=0.86, Acc=0.99)"
    )
    
    thresholds = {
        'damage': damage_threshold,
        'occlusion': occlusion_threshold,
        'crop': crop_threshold
    }
    
    with st.sidebar.expander("🔥 Grad-CAM Visualization"):
        st.write("**🔥 Grad-CAM**: Shows where model looks for specific class")
        st.write("**🌈 Grad-CAM All**: Multi-class attention visualization")
        
        if heatmap_method == "gradcam":
            st.success("🔥 Shows actual model attention - the ONLY method that reveals where your model is really looking!")
        elif heatmap_method == "gradcam_all":
            st.success("🌈 Multi-class visualization showing all model attention patterns")
    
    # Multi-model information
    with st.sidebar.expander("🤖 Multi-Model Ensemble"):
        st.write("**🅱️ Model B (Best Performer):**")
        st.write("- Macro F1: 80.6%")
        st.write("- Training: Augmentation only")
        st.write("- No masks, no CLAHE")
        
        st.write("**🅷 Model H (CLAHE Enhanced):**")
        st.write("- Macro F1: 78.1%") 
        st.write("- Training: CLAHE + partial masks")
        st.write("- Enhanced preprocessing")
        
        st.write("**🎯 Ensemble Benefits:**")
        st.write("- Combines both models")
        st.write("- Better generalization")
        st.write("- Adjustable thresholds")
        st.write("- 3-step analysis pipeline")
    
    # Visualization options
    with st.sidebar.expander("🎨 Visualization Options"):
        st.info("All visualizations are automatically generated")
        st.write("- Damage confidence heatmap")
        st.write("- Multi-class prediction overlay")
        st.write("- Comparison grid view")
    
    # Main content area
    if processing_mode == "Single Image":
        st.header("📤 Upload Single Image")
        
        uploaded_file = st.file_uploader(
            "Choose a road image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image of a road to analyze for damage"
        )
        
        if uploaded_file is not None:
            # Update heatmap generator with user settings
            heatmap_gen.alpha = heatmap_alpha
            
            results = process_multi_model_image(uploaded_file, engine, heatmap_gen, thresholds, display_scale, heatmap_method, gradcam_class, gradcam_model)
            
            # Download options
            if results:
                st.subheader("💾 Download Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Multi-model JSON results (convert numpy types to native Python types)
                    results_data = {
                        'filename': uploaded_file.name,
                        'model_b_results': {
                            'probabilities': results['model_b_results']['probabilities'],
                            'predictions': results['model_b_results']['predictions'],
                            'class_results': results['model_b_results']['class_results'],
                            'overall_confidence': results['model_b_results']['overall_confidence'],
                            'inference_time': results['model_b_results']['inference_time']
                        },
                        'model_h_results': {
                            'probabilities': results['model_h_results']['probabilities'],
                            'predictions': results['model_h_results']['predictions'],
                            'class_results': results['model_h_results']['class_results'],
                            'overall_confidence': results['model_h_results']['overall_confidence'],
                            'inference_time': results['model_h_results']['inference_time']
                        },
                        'ensemble_results': results['ensemble_results']['ensemble_results']
                    }
                    
                    # Convert numpy types to JSON-serializable types
                    json_serializable_data = convert_numpy_types(results_data)
                    json_str = json.dumps(json_serializable_data, indent=2)
                    
                    st.download_button(
                        label="📄 Download Multi-Model Results",
                        data=json_str,
                        file_name=f"{Path(uploaded_file.name).stem}_ensemble_results.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Heatmap image
                    heatmap_bytes = io.BytesIO()
                    Image.fromarray(results['visualizations']['clean_heatmap']).save(heatmap_bytes, format='PNG')
                    
                    st.download_button(
                        label="🖼️ Download Grad-CAM Heatmap",
                        data=heatmap_bytes.getvalue(),
                        file_name=f"{Path(uploaded_file.name).stem}_gradcam_heatmap.png",
                        mime="image/png"
                    )
    
    elif processing_mode == "Batch Processing":
        st.header("📁 Upload Multiple Images")
        
        uploaded_files = st.file_uploader(
            "Choose road images...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload multiple road images for batch analysis"
        )
        
        if uploaded_files:
            if st.button("🚀 Start Batch Processing", type="primary"):
                batch_results, batch_visualizations = process_batch_images(uploaded_files, engine, heatmap_gen, thresholds)
                
                if batch_results:
                    # Create downloadable batch results
                    st.subheader("💾 Download Batch Results")
                    
                    # Prepare batch JSON
                    batch_json = {
                        'summary': {
                            'total_images': len(batch_results),
                            'damage_detected': sum(1 for r in batch_results if r['ensemble_results']['ensemble_results']['class_results']['damage']['prediction']),
                            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
                        },
                        'results': []
                    }
                    
                    for result in batch_results:
                        ensemble_data = result['ensemble_results']['ensemble_results']
                        batch_json['results'].append({
                            'filename': result['filename'],
                            'probabilities': [float(p) for p in ensemble_data['probabilities']],
                            'predictions': [bool(p) for p in ensemble_data['predictions']],
                            'class_results': {
                                k: {
                                    'probability': float(v['probability']),
                                    'prediction': bool(v['prediction']),
                                    'confidence': float(v['confidence']),
                                    'threshold': float(v['threshold'])
                                } for k, v in ensemble_data['class_results'].items()
                            },
                            'overall_confidence': float(ensemble_data['overall_confidence'])
                        })
                    
                    json_str = json.dumps(batch_json, indent=2)
                    st.download_button(
                        label="📄 Download Batch Results (JSON)",
                        data=json_str,
                        file_name=f"batch_results_{int(time.time())}.json",
                        mime="application/json"
                    )
    
    else:  # Road Folder Analysis
        st.header("🛣️ Road Folder Analysis")
        
        uploaded_files = st.file_uploader(
            "Choose road images...",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload all images from a road folder with format: XXX_longitude_latitude.ext"
        )
        
        if uploaded_files:
            # Clear cached results when new files are uploaded
            current_files = [f.name for f in uploaded_files]
            cached_files = st.session_state.get('cached_file_names', [])
            
            if current_files != cached_files:
                # New files uploaded, clear cache
                for key in ['road_data', 'scoring_data', 'uploaded_files_cache', 'cached_file_names']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.cached_file_names = current_files
            
            # Validate filename format
            valid_count = 0
            for file in uploaded_files:
                if re.match(r'^\d+_-?\d+\.?\d*_-?\d+\.?\d*\.(png|jpg|jpeg)$', file.name, re.IGNORECASE):
                    valid_count += 1

            st.info(f"Found {valid_count}/{len(uploaded_files)} valid road images")
            
            if valid_count < 2:
                st.error("Need at least 2 valid road images with correct filename format: XXX_longitude_latitude.ext")
                st.markdown("""
                **Expected filename format:**
                - `000_31.296905_-97.543646.png`
                - `001_31.296954_-97.543848.jpg`
                - Sequence number + longitude + latitude + extension
                """)
            else:
                # Check if we have cached results first
                if 'road_data' in st.session_state and 'scoring_data' in st.session_state:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.success("✅ Using cached analysis results")
                    with col2:
                        if st.button("🗑️ Clear Cache", help="Clear cached results and start fresh"):
                            for key in ['road_data', 'scoring_data', 'uploaded_files_cache', 'cached_file_names', 'selected_segment', 'last_displayed_segment']:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.rerun()
                    
                    # Create road visualizer for cached results
                    road_visualizer = RoadVisualizer()
                    
                    display_road_results(
                        st.session_state.road_data, 
                        st.session_state.scoring_data, 
                        road_visualizer, 
                        st.session_state.get('uploaded_files_cache', uploaded_files),
                        None  # No segment cache for session state results (fallback mode)
                    )
                else:
                    # Show analyze button if no cached results
                    if st.button("🚀 Analyze Road", type="primary"):
                        process_road_folder(uploaded_files, engine, heatmap_gen, thresholds)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🛣️ Road Distress Classification Pipeline | Built with Streamlit</p>
        <p>Model B: EfficientNet-B3 | 80.6% Macro F1 Performance</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()