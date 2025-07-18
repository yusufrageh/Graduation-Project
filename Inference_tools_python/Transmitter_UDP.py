import cv2
import imutils
import socket
import time
import base64
import numpy as np
import tensorflow as tf

BUFF_SIZE = 65536
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket.SOCK_DGRAM for UDP
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)  # to increase buffer size

Name = socket.gethostname()
host_ip = socket.gethostbyname(Name)  # replace with your IP address
print(host_ip)  # print server IP address

port = 9999  # port number
socket_address = (host_ip, port)  # socket address

server_socket.bind(socket_address)  # bind the socket to the address
print('Listening at:', socket_address)  # print the socket address

### setting up mediapipe face landmark to adjust face position accordind to detection
physical_devices = tf.config.list_physical_devices('GPU')
try:
    interpreter = tf.lite.Interpreter(model_path="/Users/mohamedkorayem/Offline_Data/GraduationProject/Inference_tools_python/face_landmark.tflite")
    interpreter.allocate_tensors()
except Exception as e:
    print("Error loading model:", e)
    exit()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

##
WIDTH = 720  # width of the frame

def send_video_frames(client_addr):
    vid = cv2.VideoCapture(0)
    fps, st, frames_to_count, cnt = (0, 0, 20, 0)  # for calculating FPS
    
    while vid.isOpened():
        _, frame = vid.read()  # read frame from camera
        frame = imutils.resize(frame, width=WIDTH)  # resize the frame
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.merge([frame1, frame1, frame1])
        frame = frame[0:405, 155:560]
        encoded, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # encode the frame
        message = base64.b64encode(buffer)  # encode the frame

        server_socket.sendto(message, client_addr)  # send the frame to client

        frame = cv2.putText(frame, 'FPS: ' + str(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # write the FPS on the frame
        
        ### face landmark
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
        # Draw facial landmarks
        for point in output_data:
            x = int(point[0] * frame.shape[1] / input_shape[1])
            y = int(point[1] * frame.shape[0] / input_shape[2])
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        ###
        cv2.imshow('TRANSMITTING VIDEO', frame)  # display the frame
        key = cv2.waitKey(1) & 0xFF  # wait for key press
        if key == ord('q'):  # if key is 'q' then break the loop
            break

        if cnt == frames_to_count:  # calculate FPS
            try:
                fps = round(frames_to_count / (time.time() - st))
                st = time.time()
                cnt = 0
            except:
                pass
        cnt += 1

    vid.release()
    cv2.destroyAllWindows()
    time.sleep(2)

while True:
    msg, client_addr = server_socket.recvfrom(BUFF_SIZE)  # receive data from client
    print('GOT connection from', client_addr)  # print client address
    print(msg)
    
    if msg == b'hello':
        send_video_frames(client_addr)
