import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model.h5")
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

st.title("🖼️ Image Classifier (CIFAR-10 CNN)")
file = st.file_uploader("Upload an image (32x32 recommended)", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).resize((32, 32))
    st.image(img, caption="Uploaded Image")
    x = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(x)
    st.success(f"🧠 Prediction: {class_names[np.argmax(pred)]}")
