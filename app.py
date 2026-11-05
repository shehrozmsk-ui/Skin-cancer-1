
import streamlit as st
import tensorflow as tf
import numpy as np
import json

from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "best_cnn_model.keras"
META_PATH = ROOT / "model_metadata.json"

st.set_page_config(
    page_title="Skin Cancer CNN Classifier",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #10131c;
    color: white;
}
h1, h2, h3 {
    color: white;
}
.stButton > button {
    background-color: #e11d48;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("🔬 Skin Cancer Image Classification")

st.write(
    "Educational image classification using a compact CNN."
)

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Model or metadata file is missing.")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

with open(META_PATH, "r") as f:
    metadata = json.load(f)

model = load_model()

classes = metadata["class_names"]
height = metadata["image_height"]
width = metadata["image_width"]
threshold = metadata.get("threshold", 0.5)

st.subheader("Upload a skin image")

uploaded_file = st.file_uploader(
    "Choose JPG, JPEG or PNG",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        width=350
    )

    if st.button("Predict Image"):

        resized = image.resize((width, height))

        image_array = np.asarray(
            resized,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        probability = float(
            model.predict(
                image_array,
                verbose=0
            )[0][0]
        )

        index = int(probability >= threshold)

        st.subheader("Prediction Result")

        st.success(
            f"Predicted category: {classes[index]}"
        )

        st.write(
            f"{classes[0]} score: "
            f"{(1-probability)*100:.2f}%"
        )

        st.write(
            f"{classes[1]} score: "
            f"{probability*100:.2f}%"
        )

st.divider()

st.subheader("Model Information")

st.write("Model: Compact CNN")
st.write(f"Image size: {height} × {width}")

st.write(
    f"Test accuracy: "
    f"{metadata['test_accuracy']*100:.2f}%"
)

st.warning(
    "Educational demonstration only. This is not a medical "
    "diagnostic tool. Do not use predictions to diagnose "
    "skin cancer or make treatment decisions. Consult a "
    "qualified healthcare professional for medical concerns."
)
