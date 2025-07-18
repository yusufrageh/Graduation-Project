import tensorflow as tf

# Define input and output node names for your model
input_node_names = ["conv2d_input"]  # Replace with your model's input node
output_node_names = ["dense_1"]  # Replace with your model's output node

# Specify the path to your saved model
model_path = "D:/GraduationProject/Inference_tools_python/saved_model"  # For SavedModel directory

# Create the converter
converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=model_path)

# Set precision mode and other configurations
converter.convert()

# Save the converted model
converter.save("D:/GraduationProject/Inference_tools_python/engine")
