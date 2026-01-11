
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="🧠 Brain Tumor Detection", layout="centered")

# --- Custom CSS for Layout ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stImage"] img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 300px !important; 
        border: 2px solid #ddd;
        border-radius: 15px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        font-weight: bold;
        background-color: #2E86C1;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_my_model():
    try:
        # Ensure the filename matches your saved model file
        model = tf.keras.models.load_model('Brain_Tumor_Detection_Model.keras', compile=False)
        return model
    except Exception as e:
        st.sidebar.error(f"Model Load Error: {e}")
        return None

model = load_my_model()

# --- Title and Header ---
st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI scan below to analyze for the presence of a tumor.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Scan')

    # Changed button text to be relevant
    if st.button("Run Diagnostic Analysis"):
        if model is not None:
            with st.spinner('Analyzing MRI features...'):
                # --- Preprocessing ---
                img = image.convert("RGB")
                
                # IMPORTANT: Ensure target_size matches what you used during training (e.g., 128, 150, or 224)
                target_size = (128, 128) 
                img = img.resize(target_size) 
                
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # --- Prediction ---
                try:
                    prediction = model.predict(img_array)
                    # For binary classification: 0 = No Tumor, 1 = Tumor (or vice-versa depending on your training)
                    confidence = float(prediction[0][0])
                    
                    st.markdown("---")
                    
                    # LOGIC: Assuming 0 = Healthy (No Tumor) and 1 = Tumor
                    # Adjust the 0.5 threshold if you used a different one during training
                    if confidence > 0.5:
                        st.error(f"### Result: Tumor Detected ⚠️")
                        st.write(f"Probability: **{confidence * 100:.2f}%**")
                    else:
                        st.success(f"### Result: No Tumor Detected ✅")
                        st.write(f"Probability: **{(1 - confidence) * 100:.2f}%**")
                
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    st.info("Tip: Check if the model input shape matches (128, 128, 3).")
        else:
            st.warning("Model file not found. Please ensure 'Brain_Tumor_Detection_Model.keras' is in the directory.")

# --- Footer Disclaimer ---
st.markdown("---")
st.caption("**Disclaimer:** This is an AI-assisted tool and should not be used as a primary medical diagnosis. Please consult a professional radiologist.")