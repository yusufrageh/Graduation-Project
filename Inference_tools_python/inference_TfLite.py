import numpy as np
import cv2
import tensorflow as tf

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path="D:\GraduationProject\Inference_tools_python\face_landmark.tflite")
    interpreter.allocate_tensors()
except Exception as e:
    print("Error loading model:", e)
    exit()

# Load TFLite model
try:
    interpreter_eye = tf.lite.Interpreter(model_path="D:\GraduationProject\Inference_tools_python\face_landmark.tflite")
    interpreter_eye.allocate_tensors()
except Exception as e:
    print("Error loading model:", e)
    exit()
# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

eye_input_details = interpreter_eye.get_input_details()
eye_output_details = interpreter_eye.get_output_details()
eye_input_shape = eye_input_details[0]['shape']
# Open video capture
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = frame[0:720, 279:1000]
    input_data = cv2.resize(frame, (input_shape[2], input_shape[1]))

    # Normalize input frame
    input_data = (input_data.astype(np.float32)).astype(np.float32) / 255.0

    # Expand dimensions to create batch dimension
    input_data = np.expand_dims(input_data, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # Run inference
    try:
        interpreter.invoke()
    except Exception as e:
        print("Error running inference:", e)
        exit()

    # Get output and compine each point into [x,y,z]
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.reshape(output_data, (-1, 3))
    
    #crop left eye
    y_coord_top = int(output_data[299][1]*frame.shape[0]/input_shape[2])
    y_coord_bot = int(output_data[330][1]*frame.shape[0]/input_shape[2])
    x_coord_left = int(output_data[6][0]*frame.shape[1]/input_shape[1])
    x_coord_right = int(output_data[383][0]*frame.shape[1]/input_shape[1])
    left_eye = frame[y_coord_top:y_coord_bot,x_coord_left:x_coord_right]
    #crop right eye
    y_coord_top = int(output_data[69][1]*frame.shape[0]/input_shape[2])
    y_coord_bot = int(output_data[101][1]*frame.shape[0]/input_shape[2])
    x_coord_left = int(output_data[156][0]*frame.shape[1]/input_shape[1])
    x_coord_right = int(output_data[6][0]*frame.shape[1]/input_shape[1])
    right_eye = frame[y_coord_top:y_coord_bot,x_coord_left:x_coord_right]


    eyes=[right_eye,left_eye]
    # Get input and output tensors
    for eye in eyes:

        #input_data = tf.image.resize_with_pad(eye, 224, 224)
        input_data = cv2.resize(eye, (eye_input_shape[2], eye_input_shape[1]))
        # Normalize input frame
        input_data = (input_data.astype(np.float32)).astype(np.float32) / 255.0
        # Expand dimensions to create batch dimension
        input_data = np.expand_dims(input_data, axis=0)
        # Set input tensor
        interpreter_eye.set_tensor(eye_input_details[0]['index'], input_data)
        # Run inference
        interpreter_eye.invoke()
        output_data = interpreter_eye.get_tensor(eye_output_details[0]['index'])
        print(output_data)
        
