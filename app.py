import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO

st.set_page_config(page_title="Malaria AI Detector", page_icon="🦠", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS & HTML FOR AURORA DASHBOARD ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Base Styles */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #050505 !important;
    color: #ffffff;
}

/* Hide Streamlit Defaults */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    display: none;
}
.block-container {
    padding-top: 1rem !important;
    max-width: 1100px;
}

/* Aurora Background */
.aurora-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -1;
    overflow: hidden;
    background-color: #050505;
}
.aurora-glow-1 {
    position: absolute;
    top: -10%;
    right: -10%;
    width: 70vw;
    height: 70vw;
    background: radial-gradient(circle, rgba(140,230,210,0.15) 0%, rgba(0,0,0,0) 60%);
    filter: blur(90px);
}
.aurora-glow-2 {
    position: absolute;
    bottom: -20%;
    left: -10%;
    width: 70vw;
    height: 70vw;
    background: radial-gradient(circle, rgba(200,240,230,0.1) 0%, rgba(0,0,0,0) 60%);
    filter: blur(90px);
}

/* Glassmorphic Nav */
.glass-nav-container {
    display: flex;
    justify-content: center;
    width: 100%;
    margin-bottom: 70px;
    margin-top: 10px;
}
.glass-nav {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 50px;
    padding: 12px 35px;
    display: flex;
    gap: 30px;
    align-items: center;
}
.glass-nav span {
    font-size: 0.9em;
    font-weight: 500;
    color: #c9d1d9;
    cursor: pointer;
    transition: color 0.3s ease;
}
.glass-nav span:hover {
    color: #ffffff;
}

/* Hero Section */
.hero-container {
    text-align: center;
    margin-bottom: 60px;
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    z-index: 1;
}
.hero-badge {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 10px 25px;
    border-radius: 50px;
    font-size: 0.9em;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 25px;
    display: inline-flex;
    align-items: center;
    gap: 10px;
}
.hero-title {
    font-size: 5em;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 25px;
    background: linear-gradient(180deg, #FFFFFF 0%, #A0AEC0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1.2em;
    color: #8b949e;
    max-width: 650px;
    line-height: 1.6;
}

/* Metric Cards Retained */
.metric-card-positive {
    background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(255, 107, 107, 0.2) 100%);
    border-left: 5px solid #FF6B6B;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
}
.metric-card-negative {
    background: linear-gradient(135deg, rgba(78, 205, 196, 0.1) 0%, rgba(78, 205, 196, 0.2) 100%);
    border-left: 5px solid #4ECDC4;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
}

/* File Uploader override to match dark theme better */
div[data-testid="stFileUploader"] section {
    background-color: rgba(255,255,255,0.03) !important;
    border: 1px dashed rgba(255,255,255,0.2) !important;
    border-radius: 15px !important;
}
</style>

<!-- Background Aurora Divs -->
<div class="aurora-bg">
    <div class="aurora-glow-1"></div>
    <div class="aurora-glow-2"></div>
</div>

<!-- Hero Section -->
<div class="hero-container">
    <div class="hero-badge">
        🧬 AI-Powered Diagnostic Tool
    </div>
    <div class="hero-title">Instant Malaria<br>Detection</div>
    <div class="hero-subtitle">Upload microscopic blood cell images below to instantly detect the presence of Plasmodium parasites. Our Convolutional Neural Network (CNN) analyzes cellular structures in real-time to assist in rapid malaria diagnosis.</div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ Settings & Info")
    st.info("This application uses a Convolutional Neural Network (CNN) to detect malaria infection in real-time blood cell imagery.")
    st.markdown("---")
    st.markdown("### How to use:")
    st.markdown("1. Upload one or more blood cell images.")
    st.markdown("2. Click the **Predict** button.")
    st.markdown("3. View the AI analysis below.")
    st.markdown("---")
    st.caption("Powered by TensorFlow & Streamlit")

# Load model with keras
@st.cache_resource(show_spinner="Loading deep learning model...")
def load_model():
    model = tf.keras.models.load_model("malaria_model.h5")
    return model

model = load_model()

# --- UPLOADER UI ---

# Use columns to center the content on wide layouts
col1, col2, col3 = st.columns([1, 2.5, 1])

with col2:
    # Multi-file uploader
    uploaded_files = st.file_uploader(
        "Upload Blood Cell Images (.png, .jpg)", 
        type=["jpg", "png", "jpeg"], 
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Predict AI Analysis", type="primary"):
        def image_to_base64(img):
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()

        for uploaded_file in uploaded_files:
            # Load image
            image = Image.open(uploaded_file).convert("RGB")
            img_b64 = image_to_base64(image)

            # Preprocess for model
            img_size = (64, 64)   
            img_array = image.resize(img_size)
            img_array = np.array(img_array) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(img_array, verbose=0)
            confidence = float(prediction[0][0]) 
            
            # Setup dynamic styling based on result
            if confidence > 0.5:
                conf_percent = confidence * 100
                bg_color = "rgba(78, 205, 196, 0.08)"
                border_color = "rgba(78, 205, 196, 0.2)"
                accent_color = "#4ECDC4"
                icon = "✅"
                status = "Malaria Negative"
                description = "The AI analysis indicates this cell is healthy."
            else:
                conf_percent = (1 - confidence) * 100
                bg_color = "rgba(255, 107, 107, 0.08)"
                border_color = "rgba(255, 107, 107, 0.2)"
                accent_color = "#FF6B6B"
                icon = "⚠️"
                status = "Malaria Positive"
                description = "The AI analysis has detected signs of malaria infection."

            img_html = f"""
<div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 10px; padding: 12px; display: inline-block; text-align: center;">
    <img src="data:image/png;base64,{img_b64}" style="width: 120px; height: 120px; object-fit: cover; border-radius: 6px; box-shadow: 0 2px 10px rgba(0,0,0,0.5);" />
    <p style="color:#8b949e; margin-top:8px; margin-bottom:0; font-size:0.85em; font-weight: 500; max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{uploaded_file.name}</p>
</div>
"""
            
            res_html = f"""
<div style="background-color: {bg_color}; border: 1px solid {border_color}; border-left: 4px solid {accent_color}; border-radius: 10px; padding: 15px 20px; max-width: 350px;">
    <h4 style="margin-top:0; margin-bottom: 8px; color:{accent_color}; font-size: 1.25em;">{icon} {status}</h4>
    <p style="color:#a0aec0; font-size: 0.95em; margin-bottom: 12px; line-height: 1.3;">{description}</p>
    <p style="margin-bottom:0; color:#8b949e; font-size: 0.9em;">AI Confidence: <strong style="color:#ffffff;">{conf_percent:.1f}%</strong></p>
</div>
"""
            
            with st.container():
                st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.1); margin: 20px 0;'>", unsafe_allow_html=True)
                img_col, res_col = st.columns([1, 1.5], gap="large")
                
                with img_col:
                    st.markdown(img_html, unsafe_allow_html=True)
                with res_col:
                    st.markdown(res_html, unsafe_allow_html=True)