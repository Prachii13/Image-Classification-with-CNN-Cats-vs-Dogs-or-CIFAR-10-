import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

model = tf.keras.models.load_model("model.h5")
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

img_path = sys.argv[1]
img = image.load_img(img_path, target_size=(32, 32))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

pred = model.predict(x)
class_idx = np.argmax(pred[0])
print("🧠 Prediction:", class_names[class_idx])
