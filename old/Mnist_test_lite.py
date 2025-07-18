import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras

# Load your TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="mnist_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the true labels for your images
true_labels = [2, 4, 5, 8]  

# Initialize variables for accuracy calculation
correct_predictions = 0
total_images = len(true_labels)

# Loop through all images
for i, image_path in enumerate(["2.png","4.png","5.png","8.png"]):
    # Load and preprocess the image
    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Post-process the output (e.g., get the predicted label)
    predicted_label = np.argmax(output_data)

    # Compare with the true label
    true_label = true_labels[i]
    if predicted_label == true_label:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_images
print("Accuracy:", accuracy)
