import socket, cv2, base64, time
from playsound import playsound
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import threading
from IPython.display import clear_output

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

# Load pretrained model to make predictions
target_size = (160, 160)
class_name = ['closed', 'open']
saved_model_dir = "D:\GraduationProject\Inference_tools_python\saved_model"
loaded_model = tf.saved_model.load(saved_model_dir)

# Get the serving function
infer = loaded_model.signatures["serving_default"]

BUFF_SIZE = 65536
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
host_ip = '192.168.1.4'
print(host_ip)
port = 9999
message = b'hello'
client_socket.sendto(message, (host_ip, port))

def get_alpha(N):
    return 2 / (N + 1)

def update_ema(new_value, current_ema, alpha):
    return alpha * new_value + (1 - alpha) * current_ema

# Initialize EMA variables
N = 200  # Define the period for EMA
alpha = get_alpha(N)  # Define the smoothing factor

# Initial values for EMA (assuming first predictions will initialize these)
ema = None
results = []
ema_results = []
# Create a plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(10, 6))
update_plot_freq = 10  # Update plot every 10 iterations
iteration = 0
sound_file = 'D:\GraduationProject\Inference_tools_python\\alarm.mp3'

def play_alarm_sound(sound_file):
    playsound(sound_file)
    return

# Create a thread for playing the sound
alarm_time = None  
count = 0

def cleanup():
    plt.ioff()  # Turn off interactive mode
    plt.text(0.5, 0.5, 'Emergency declared and the system is shutting down. handing over to the emergency system', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=35,color='red',wrap=True)
    plt.show()  # Display the final plot
    client_socket.close()
    cv2.destroyAllWindows()
    print("Emergency flag is declared and the system is shutting down. handing over to the emergency system.")

try:
    while True:
        count += 1
        start_time = time.time()
        packet, _ = client_socket.recvfrom(BUFF_SIZE)
        data = base64.b64decode(packet, ' /')
        npdata = np.fromstring(data, dtype=np.uint8)
        frame = cv2.imdecode(npdata, 1)
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
            break
        # Get output and combine each point into [x,y,z]
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.reshape(output_data, (-1, 3))
        
        # Crop left eye
        y_coord_top = int(output_data[299][1] * frame.shape[0] / input_shape[2])
        y_coord_bot = int(output_data[330][1] * frame.shape[0] / input_shape[2])
        x_coord_left = int(output_data[6][0] * frame.shape[1] / input_shape[1])
        x_coord_right = int(output_data[383][0] * frame.shape[1] / input_shape[1])
        left_eye = frame[y_coord_top:y_coord_bot, x_coord_left:x_coord_right]
        # Crop right eye
        y_coord_top = int(output_data[69][1] * frame.shape[0] / input_shape[2])
        y_coord_bot = int(output_data[101][1] * frame.shape[0] / input_shape[2])
        x_coord_left = int(output_data[156][0] * frame.shape[1] / input_shape[1])
        x_coord_right = int(output_data[6][0] * frame.shape[1] / input_shape[1])
        right_eye = frame[y_coord_top:y_coord_bot, x_coord_left:x_coord_right]
        left_eye = tf.image.resize_with_pad(left_eye, target_size[0], target_size[1])
        # Resize with padding
        right_eye = tf.image.resize_with_pad(right_eye, target_size[0], target_size[1])
        left_eye = np.expand_dims(left_eye, axis=0)
        right_eye = np.expand_dims(right_eye, axis=0)
        ### Right eye inference
        input_tensor = tf.convert_to_tensor(right_eye, dtype=tf.float32)
        predictions = infer(input_tensor)
        result_right = predictions['dense_1'].numpy()
        ### Left eye inference
        input_tensor = tf.convert_to_tensor(left_eye, dtype=tf.float32)
        predictions = infer(input_tensor)
        result_left = predictions['dense_1'].numpy()

        predicted_class_left = class_name[1 if result_right > 0.6 else 0]
        predicted_class_right = class_name[1 if result_left > 0.6 else 0]  
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
        result = (result_right + result_left) / 2
        # Update EMA with the new prediction
        if ema is None:
            ema = 1  # Initialize EMA with 1 for first prediction
        else:
            ema = update_ema(result, ema, alpha)   
        # Store results for plotting
        if ema < 0.65:
            if count % 30 == 0:
                alarm_thread = threading.Thread(target=play_alarm_sound, args=(sound_file,))
                alarm_thread.start()
            if alarm_time is None:
                alarm_time = time.time()
                print("Alarm flag is set")
            elif time.time() - alarm_time > 7:
                print("Emergency flag is set")
                break
        elif ema > 0.75:
            alarm_time = None
        results.append(result)
        ema_results.append(ema)

        if iteration % update_plot_freq == 0:
            results_combined_array = np.array(results).flatten()
            ema_results_combined_array = np.array(ema_results).flatten()
            clear_output(wait=True)
            ax.cla()
            ax.plot(results_combined_array, label='Combined Predictions', color='blue')
            ax.plot(ema_results_combined_array, label='EMA Combined', color='orange')
            ax.axhline(y=0.65, color='red', linestyle='--', label='Threshold')
            ax.axhline(y=0.75, color='green', linestyle='--', label='Reset Threshold')
            ax.legend()
            ax.set_title('Combined Predictions and EMA')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Prediction')
            plt.tight_layout()
            plt.pause(0.01)
        iteration += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cleanup()
