import serial
import time
alarm_flag = bool(input("Enter alarm flag: "))
emergency_flag = bool(input("Enter emergency flag: "))
lane_deviation = "000000"
if alarm_flag == True and emergency_flag == False:
    eye = "10"
    print(eye)
elif alarm_flag == True and emergency_flag== True:
    eye = "11"
    print(eye)
else: 
    eye = "00"
    print(eye)

binary_data = lane_deviation + eye
# Configure the serial port
ser = serial.Serial('COM3', 9600, timeout=1)  # Replace 'COM3' with your Arduino's serial port
time.sleep(1.5)  # Allow some time for the serial connection to establish

# Send the integer
ser.write(binary_data.encode('ascii'))
ser.flush()
# Wait for the Arduino to process the data
# Close the serial port
ser.close()
