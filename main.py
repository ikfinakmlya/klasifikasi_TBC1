import streamlit as st
import tensorflow as tf
import numpy as np
from keras.layers import LeakyReLU
from PIL import Image

# =============================
# Fungsi load model aman
# =============================
@st.cache_resource
def load_model_safe():
    model = None
    try:
        # coba load H5 (lebih kompatibel di Keras 3)
        model = tf.keras.models.load_model(
            "model_tbc.h5",
            custom_objects={"LeakyReLU": LeakyReLU}
        )
        st.success("✅ Model berhasil dimuat dari model_tbc.h5")
    except Exception as e1:
        st.warning(f"Gagal load .h5 → {e1}")
        try:
            # fallback ke format .keras
            model = tf.keras.models.load_model(
                "model_tbc.keras",
                custom_objects={"LeakyReLU": LeakyReLU}
            )
            st.success("✅ Model berhasil dimuat dari model_tbc.keras")
        except Exception as e2:
            st.error(f"❌ Gagal load model: {e2}")
    return model

# =============================
# Load model
# =============================
model = load_model_safe()
class_labels = ["Normal", "TBC"]

# =============================
# UI Streamlit
# =============================
st.title("🩻 Klasifikasi Citra X-Ray TBC vs Normal")
st.write("Upload gambar X-ray untuk mendeteksi apakah termasuk **TBC** atau **Normal**.")

# Upload file
uploaded_file = st.file_uploader("📤 Upload gambar X-ray (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # tampilkan gambar
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar diupload", use_container_width=True)

    # preprocessing
    img_resized = image.resize((64, 64))  # ukuran sesuai input
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediksi
    pred = model.predict(img_array)

    # handle sigmoid (1 neuron) atau softmax (2 neuron)
    if pred.shape[1] == 1:  
        prob_tbc = float(pred[0][0])
        prob_normal = 1 - prob_tbc
        label = "TBC" if prob_tbc > 0.5 else "Normal"
        confidence = max(prob_tbc, prob_normal)
    else:
        pred_class = np.argmax(pred, axis=1)[0]
        label = class_labels[pred_class]
        confidence = np.max(pred)

    # hasil
    st.subheader("🔍 Hasil Prediksi")
    st.write(f"👉 **{label}** (Keyakinan: {confidence:.2f})")
