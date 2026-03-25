import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# -------------------------------------------------
# Load Model (cached for performance)
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# -------------------------------------------------
# Load Class Indices
# -------------------------------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.title("🌱 About")
    st.markdown(
        """
        **Plant Disease Detection System**

        - CNN-based image classifier  
        - Trained on PlantVillage dataset  
        - Achieves ~92% validation accuracy  

        **Tech Stack**
        - TensorFlow / Keras  
        - Python  
        - Streamlit  
        """
    )

    st.markdown("---")
    st.caption("Developed for Hobby Project")

# -------------------------------------------------
# Main Title
# -------------------------------------------------
st.title("🌿 Plant Disease Detection")
st.subheader("Upload a leaf image to detect disease")

# -------------------------------------------------
# File Uploader
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a leaf image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# Prediction Logic
# -------------------------------------------------
if uploaded_file is not None:

    col1, col2 = st.columns([1, 1])

    # -------- Image Column --------
    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(
            img,
            caption="Uploaded Leaf Image",
            width=350
        )

    # -------- Preprocessing --------
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------- Prediction --------
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100

    # -------- Result Column --------
    with col2:
        st.markdown("### 🧠 Prediction Result")

        st.success(f"**{predicted_class}**")

        st.markdown("**Confidence Level**")
        st.progress(int(confidence))
        st.caption(f"{confidence:.2f}% confidence")

        if "Healthy" in predicted_class:
            st.info("✅ The leaf appears to be healthy.")
        else:
            st.warning("⚠️ Disease detected. Early treatment recommended.")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("⚡ CNN-based Plant Disease Detection | Demo Application")