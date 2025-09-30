import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# load model aman
@st.cache_resource
def load_model_safe():
    model = None
    try:
        # coba load format .keras (recommended di Keras 3)
        model = tf.keras.models.load_model("model_tbc.keras")
        st.success("✅ Model berhasil dimuat dari model_tbc.keras")
    except Exception as e1:
        st.warning(f"Gagal load .keras → {e1}")
        try:
            # fallback: coba load legacy H5
            from keras.layers import LeakyReLU  # import untuk custom layer
            model = tf.keras.models.load_model(
                "model_tbc.h5",
                custom_objects={"LeakyReLU": LeakyReLU}
            )
            st.success("✅ Model berhasil dimuat dari model_tbc.h5")
        except Exception as e2:
            st.error(f"❌ Gagal load model: {e2}")
    return model

# panggil load model
model = load_model_safe()
class_labels = ["Normal", "TBC"]

# judul app
st.title("🩻 Klasifikasi X-Ray TBC vs Normal")

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
    st.subheader("🔍 Hasil Prediksi")
    st.write(f"👉 **{class_labels[pred_class]}**")
    st.write(f"Tingkat keyakinan: **{confidence:.2f}**")
