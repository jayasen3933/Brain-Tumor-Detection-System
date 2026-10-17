
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

MODEL_PATH = "brain_tumor_model.keras"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

st.title("🧠 Brain Tumor Classification")
st.write("Upload a brain MRI image to classify it using the trained Xception model.")

st.warning(
    "Educational project only. This application is not a medical diagnostic tool. "
    "Predictions may be incorrect and must not guide medical decisions."
)

try:
    model = load_model()
    st.success("Trained model loaded successfully.")
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded MRI")
            st.image(image, caption="Uploaded image", use_container_width=True)

        with col2:
            st.subheader("Classification result")

            resized_image = image.resize(IMAGE_SIZE)
            image_array = np.asarray(resized_image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            with st.spinner("Analyzing image..."):
                probabilities = model.predict(image_array, verbose=0)[0]

            predicted_index = int(np.argmax(probabilities))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(probabilities[predicted_index]) * 100

            if predicted_class == "notumor":
                st.info("The model classified this image as: No Tumor")
            else:
                st.info(f"Model classification: {predicted_class.title()}")

            st.metric("Model confidence", f"{confidence:.2f}%")

            st.subheader("Class probabilities")

            for name, probability in zip(CLASS_NAMES, probabilities):
                display_name = "No Tumor" if name == "notumor" else name.title()
                st.write(f"{display_name}: {probability * 100:.2f}%")
                st.progress(float(probability))

            st.caption(
                "These probabilities are model outputs, not verified "
                "clinical probabilities or a diagnosis."
            )

    except Exception as e:
        st.error(f"Unable to process this image: {e}")
