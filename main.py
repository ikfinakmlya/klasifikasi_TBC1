import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# coba import LeakyReLU dari keras
try:
    from keras.layers import LeakyReLU
except:
    from keras.layers import LeakyReLU

# fungsi load model dengan fallback
def load_model_safe():
    model = None
    try:
        # coba load h5 dengan custom_objects
        model = tf.keras.models.load_model(
            "model_tbc.h5",
            custom_objects={"LeakyReLU": LeakyReLU}
        )
        st.success("Model berhasil dimuat dari model_tbc.h5")
    except Exception as e1:
        st.warning(f"Gagal load .h5 → {e1}")
        try:
            # fallback: coba load SavedModel folder
            model = tf.keras.models.load_model("model_tbc")
            st.success("Model berhasil dimuat dari folder model_tbc/")
        except Exception as e2:
            st.error(f"Gagal load model: {e2}")
    return model

# load model
model = load_model_safe()
class_labels = ["Normal", "TBC"]

st.title("Klasifikasi X-Ray TBC vs Normal")

# upload file
uploaded_file = st.file_uploader("Upload Gambar X-Ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
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
    st.subheader("Hasil Prediksi")
    st.write(f"👉 **{class_labels[pred_class]}**")
    st.write(f"Tingkat keyakinan: **{confidence:.2f}**")
