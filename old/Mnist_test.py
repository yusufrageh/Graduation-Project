import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('TF_model.keras')

img = image.load_img('8.png', color_mode='grayscale', target_size=(28, 28))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  
img_array /= 255.0  

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

print(f'Predicted Class: {predicted_class[0]}')