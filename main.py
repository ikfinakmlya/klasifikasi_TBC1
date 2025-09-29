import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.layers import LeakyReLU


# load model
model = tf.keras.models.load_model(
    "model_tbc.h5",
    custom_objects={"LeakyReLU": LeakyReLU}
)
class_labels = ["Normal", "TBC"]

st.title("Klasifikasi X-Ray TBC vs Normal")

# upload file
uploaded_file = st.file_uploader("Upload Gambar X-Ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # tampilkan gambar
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar diupload", use_column_width=True)

    # preprocessing
    img = image.resize((64, 64))                   # resize sesuai input model
    img_array = np.array(img) / 255.0              # normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # tambah dimensi batch

    # prediksi
    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred)

    # hasil
    st.write(f"Prediksi: **{class_labels[pred_class]}**")
    st.write(f"Tingkat keyakinan: {confidence:.2f}")
