import numpy as np
import tensorflow as tf
import socket,cv2,base64,time
# Load TFLite model
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
try:
    interpreter = tf.lite.Interpreter(model_path="D:\GraduationProject\Inference_tools_python\\face_landmark.tflite")
    interpreter.allocate_tensors()
except Exception as e:
    print("Error loading model:", e)
    exit()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

#loading pretrained model to make predictions
model=tf.keras.models.load_model('D:\GraduationProject\Inference_tools_python\eye_model.h5')
target_size = (160, 160)
class_name = ['closed', 'open']


BUFF_SIZE = 65536
client_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
host_name = socket.gethostname()
host_ip = '192.168.1.8'#  socket.gethostbyname(host_name)
print(host_ip)
port = 9999
message = b'Hello'
client_socket.sendto(message,(host_ip,port))

while True:
    start_time = time.time()
    packet,_ = client_socket.recvfrom(BUFF_SIZE)
    data = base64.b64decode(packet,' /')
    npdata = np.fromstring(data,dtype=np.uint8)
    frame = cv2.imdecode(npdata,1)
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
    left_eye = tf.image.resize_with_pad(left_eye, target_size[0], target_size[1])
    # Resize with padding
    right_eye = tf.image.resize_with_pad(right_eye, target_size[0], target_size[1])
    left_eye = np.expand_dims(left_eye, axis=0)
    right_eye = np.expand_dims(right_eye, axis=0)
    imgs = np.vstack([left_eye, right_eye])
    predictions = model.predict(imgs)
    predicted_class_left = class_name[1 if predictions[0] > 0.5 else 0]
    predicted_class_right = class_name[1 if predictions[1] > 0.5 else 0]

    # Display eye state predictions
    cv2.putText(frame, f"Left Eye: {predicted_class_left}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Right Eye: {predicted_class_right}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw facial landmarks
    for point in output_data:
        x = int(point[0] * frame.shape[1] / input_shape[1])
        y = int(point[1] * frame.shape[0] / input_shape[2])
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    end_time = time.time()
    total_time = end_time - start_time
    fps = 1 / total_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Video with Inference Results", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key  == ord('q'):
        break
client_socket.close()


        
