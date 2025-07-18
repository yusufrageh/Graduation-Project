from tensorflow import keras
import tensorflow as tf
model = keras.models.load_model('D:\GraduationProject\Inference_tools_python\eye_model.h5')
saved_model_dir = "D:\GraduationProject\Inference_tools_python\saved_model"
tf.saved_model.save(model, saved_model_dir)