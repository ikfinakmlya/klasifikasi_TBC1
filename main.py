import streamlit as st
import tensorflow as tf
from keras.layers import LeakyReLU 
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model (.keras)
@st.cache_resource  # supaya model cuma diload sekali
def load_model():
    return tf.keras.models.load_model("model_tbc.keras")

model = load_model()
class_labels = ["Normal", "TBC"]

# Judul Aplikasi
st.title("🩻 Klasifikasi Citra X-Ray TBC vs Normal")
st.write("Upload gambar X-ray untuk mendeteksi apakah termasuk **TBC** atau **Normal**.")

# Upload file
uploaded_file = st.file_uploader("📤 Upload gambar X-ray (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar diupload", use_container_width=True)

    # Preprocessing
    img_resized = img.resize((64, 64))  # sesuai input model
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # tambah dimensi batch

    # Prediksi
    prediction = model.predict(img_array)

    # Handle sigmoid (1 output neuron) atau softmax (2 output neuron)
    if prediction.shape[1] == 1:
        prob_tbc = float(prediction[0][0])
        prob_normal = 1 - prob_tbc
        label = "TBC" if prob_tbc > 0.5 else "Normal"
        confidence = max(prob_tbc, prob_normal)
    else:
        pred_class = np.argmax(prediction, axis=1)[0]
        label = class_labels[pred_class]
        confidence = np.max(prediction)

    # Hasil
    st.subheader("🔍 Hasil Prediksi")
    st.write(f"👉 **{label}** (Keyakinan: {confidence:.2f})")
