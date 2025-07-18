
import tensorflow as tf
from tensorflow.keras.models import load_model
loaded_model = load_model('D:\GraduationProject\Inference_tools_python\eye_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]
eye_model = converter.convert()

# Step 4: Save the optimized TFLite model
with open('eye_model.tflite', 'wb') as f:
    f.write(eye_model)
