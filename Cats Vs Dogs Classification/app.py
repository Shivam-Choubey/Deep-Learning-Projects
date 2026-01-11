import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(page_title="Cats vs Dogs Classifier", layout="centered")

# --- Custom CSS for Layout ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stImage"] img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 250px !important; 
        border: 2px solid #ddd;
        border-radius: 15px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        font-weight: bold;
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model('cats_vs_dogs_model_1.keras', compile=False)
        return model
    except Exception as e:
        st.sidebar.error(f"Model Load Error: {e}")
        return None

model = load_my_model()

# --- Title and Header ---
st.title("🐱Cats vs 🐶Dogs Classifier")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')

    if st.button("Identify Pet"):
        if model is not None:
            with st.spinner('Analyzing features...'):
                # --- Preprocessing ---
                img = image.convert("RGB")
                
                # TRYING 128x128 based on the 86528 error logic
                target_size = (128, 128) 
                img = img.resize(target_size) 
                
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # --- Prediction ---
                try:
                    prediction = model.predict(img_array)
                    confidence = float(prediction[0][0])
                    
                    st.markdown("---")
                    if confidence > 0.5:
                        st.info(f"### Result: It's a Dog 🐶")
                        st.write(f"Confidence Level: **{confidence * 100:.2f}%**")
                    else:
                        st.success(f"### Result: It's a Cat 🐱")
                        st.write(f"Confidence Level: **{(1 - confidence) * 100:.2f}%**")
                
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    # Helpful debug info for the user
                    st.write("Debug Info: The model expects a different input size.")
                    if hasattr(model, 'input_shape'):
                        st.write(f"Model Input Shape: {model.input_shape}")
        else:
            st.warning("Model not found.")