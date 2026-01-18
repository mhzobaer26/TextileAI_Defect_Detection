"""
🧵 TextileAI Defect Detection Web Application
Upload a fabric image to detect defects using AI
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from model_utils import preprocess_image, predict_defect, get_detailed_analysis
import os

# Page configuration
st.set_page_config(
    page_title="TextileAI - Defect Detection",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .defect-box {
        background: linear-gradient(135deg, #FEE2E2 0%, #FCA5A5 100%);
        border-left: 5px solid #DC2626;
    }
    .no-defect-box {
        background: linear-gradient(135deg, #D1FAE5 0%, #6EE7B7 100%);
        border-left: 5px solid #059669;
    }
    .metric-card {
        background: #F8FAFC;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #E2E8F0;
        margin: 0.5rem 0;
    }
    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        background: #E2E8F0;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load model (cached for performance)
@st.cache_resource
def load_model():
    """Load the trained model with compatibility handling"""
    # Use the newly trained model
    model_path = 'best_model.keras' if os.path.exists('best_model.keras') else 'textile_defect_model.keras'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info("📝 Please ensure the model file is in the same directory as app.py")
        st.stop()
    
    try:
        # Try loading with compile=False to avoid training-specific issues
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Recompile the model for inference
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("""
        **Troubleshooting Tips:**
        - The model may have been trained with a different TensorFlow version
        - Try re-training the model with the current TensorFlow version (2.20.0)
        - Or ensure the model file is not corrupted
        """)
        st.stop()

# Main app
def main():
    # Header
    st.markdown('<p class="main-header">🧵 TextileAI Defect Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a fabric image to detect defects using AI-powered analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.info("""
        **TextileAI** uses deep learning (EfficientNetB0) to detect defects in textile fabrics.
        
        **How to use:**
        1. Upload a fabric image (JPG, PNG, JPEG)
        2. Wait for AI analysis
        3. View detailed results
        
        **Classes:**
        - ✅ No Defect
        - ❌ Defect
        """)
        
        st.header("📊 Model Info")
        st.markdown("""
        - **Architecture**: EfficientNetB0
        - **Input Size**: 224x224
        - **Classes**: 2 (Defect, No Defect)
        - **Framework**: TensorFlow/Keras
        """)
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model = load_model()
    
    st.success("✅ Model loaded successfully!")
    
    # File upload
    st.markdown("---")
    st.subheader("📤 Upload Fabric Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the fabric for defect detection"
    )
    
    if uploaded_file is not None:
        # Display columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            # Image info
            st.markdown(f"""
            <div class="metric-card">
                <b>📏 Image Details:</b><br>
                Size: {image.size[0]} x {image.size[1]} px<br>
                Format: {image.format}<br>
                Mode: {image.mode}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🔍 AI Analysis")
            
            with st.spinner("🧠 Analyzing image..."):
                # Preprocess and predict
                processed_img = preprocess_image(image)
                prediction, confidence, class_name = predict_defect(model, processed_img)
                
                # Get detailed analysis
                analysis = get_detailed_analysis(class_name, confidence)
            
            # Display results
            if class_name == "No Defect":
                st.markdown(f"""
                <div class="result-box no-defect-box">
                    <h2 style="margin:0; color:#065F46;">✅ {class_name}</h2>
                    <p style="margin-top:0.5rem; font-size:1.1rem;">The fabric appears to be in good condition!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box defect-box">
                    <h2 style="margin:0; color:#991B1B;">❌ {class_name} Detected</h2>
                    <p style="margin-top:0.5rem; font-size:1.1rem;">The fabric has potential quality issues.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence score
            st.markdown("### 📊 Confidence Score")
            confidence_pct = int(confidence * 100)
            
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence_pct}%;">
                    {confidence_pct}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Confidence Level:** {confidence_pct}%")
            
            # Detailed prediction breakdown
            st.markdown("### 📈 Prediction Breakdown")
            
            defect_prob = float(prediction[0][0] * 100)
            no_defect_prob = float(prediction[0][1] * 100)
            
            st.markdown(f"""
            <div class="metric-card">
                <b>❌ Defect:</b> {defect_prob:.2f}%<br>
                <b>✅ No Defect:</b> {no_defect_prob:.2f}%
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bars for each class
            st.progress(float(defect_prob / 100), text=f"Defect: {defect_prob:.1f}%")
            st.progress(float(no_defect_prob / 100), text=f"No Defect: {no_defect_prob:.1f}%")
        
        # Detailed Analysis Section
        st.markdown("---")
        st.subheader("📝 Detailed Analysis")
        
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
        
        with analysis_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 Classification</h4>
                <p style="font-size:1.3rem; font-weight:bold; color:#1E3A8A;">{class_name}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with analysis_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Confidence</h4>
                <p style="font-size:1.3rem; font-weight:bold; color:#7C3AED;">{confidence_pct}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with analysis_col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>✨ Quality Status</h4>
                <p style="font-size:1.3rem; font-weight:bold; color:{'#059669' if class_name == 'No Defect' else '#DC2626'};">{analysis['status']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        st.info(analysis['recommendation'])
        
        # Additional details
        with st.expander("📋 View Technical Details"):
            st.json({
                "prediction_class": class_name,
                "confidence_score": f"{confidence:.4f}",
                "defect_probability": f"{defect_prob:.2f}%",
                "no_defect_probability": f"{no_defect_prob:.2f}%",
                "model_architecture": "EfficientNetB0 (Transfer Learning)",
                "input_size": "224x224",
                "prediction_threshold": "0.5"
            })
    
    else:
        # Show example when no image uploaded
        st.info("👆 Please upload a fabric image to begin analysis")
        
        st.markdown("### 🎯 Example Use Cases")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🏭 Manufacturing**
            - Quality control
            - Real-time inspection
            - Batch verification
            """)
        
        with col2:
            st.markdown("""
            **📦 Supply Chain**
            - Incoming inspection
            - Supplier quality check
            - Damage assessment
            """)
        
        with col3:
            st.markdown("""
            **🔬 R&D**
            - Material testing
            - Process optimization
            - Defect analysis
            """)

if __name__ == "__main__":
    main()
