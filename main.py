import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_tbc.h5")

model = load_model()
class_labels = ["Normal", "TBC"]

st.title("🩻 Klasifikasi Citra X-Ray TBC vs Normal")

uploaded_file = st.file_uploader("Upload Gambar X-Ray", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar diupload", use_container_width=True)

    img = image.resize((64,64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred)

    st.subheader("🔍 Hasil Prediksi")
    st.write(f"👉 **{class_labels[pred_class]}** (Keyakinan: {confidence:.2f})")
