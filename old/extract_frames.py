import cv2
from PIL import Image, ImageDraw
import json
import os
from tqdm import tqdm
import time
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

# Load JSON metadata
json_filename = input('enter the name of the json file');
with open(json_filename, 'r') as json_file:
    metadata = json.load(json_file)

# Open the video file

video_path = input('enter the video path')
cap = cv2.VideoCapture(video_path)

# Check if the video capture object is opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video file.")
else:
    print("Video capture object opened successfully.")

# Output directory for labeled images
output_dir = 'output_images/'
os.makedirs(output_dir, exist_ok=True)

# For labeling, write the corresponding letter for the downloading group w Person's number
postName = input('enter group letter followed by the number of the subject')

# Face detection setup
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.2)

# Get the total number of frames
total_frames = metadata['openlabel']['frame_intervals'][0]['frame_end'] + 1
print(f'Total number of frames: {total_frames}')

# Process each action in the metadata
for frame_data in metadata['openlabel']['actions'].values():
    action_name = frame_data['type'].replace('/','_')
    frame_intervals = frame_data['frame_intervals']

    # Print the length of frame_intervals for each action

    # Create subdirectory for each action
    action_output_dir = os.path.join(output_dir, action_name)
    os.makedirs(action_output_dir, exist_ok=True)

    for interval in frame_intervals:
        frame_start = interval['frame_start']
        frame_end = interval['frame_end']

        # Extract frames from the video
        set_success = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        # Check if the frame position was set successfully
        if not set_success:
            print(f"Error: Unable to set frame position to {frame_start}. Skipping interval {interval}.")
            continue

        for frame_num in tqdm(range(frame_start, frame_end + 1), desc=f'Processing {action_name}'):
            ret, frame = cap.read()

            # Check if the frame is not empty
            if not ret or frame is None:
                print(f"Skipping empty frame at {frame_num}")
                continue
            # Face detection logic
            if frame_num % 5 == 0:  # Adjust the condition as needed
                image_rows, image_cols, _ = frame.shape
                mp_input = frame
                res = mp_face.process(mp_input)
                detection = res.detections
                 # Additional processing for labeling
                if detection:
                #Not empty detections
                    for det in detection:
                        if(det.score[0] > 0.2):
                            
                            print("detected")
                            loc = det.location_data
                            relative_bounding_box = loc.relative_bounding_box
                            rect_start_point = _normalized_to_pixel_coordinates(
                                relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols, image_rows)
                            rect_end_point = _normalized_to_pixel_coordinates(
                                relative_bounding_box.xmin + relative_bounding_box.width,
                                relative_bounding_box.ymin + relative_bounding_box.height, image_cols, image_rows)
        
                            xleft, ytop = rect_start_point
                            xright, ybot = rect_end_point
                            crop_img = mp_input[ytop:ybot, xleft:xright]
                            #####################################################################################################
                            # Here 34an aldata htOverwrite 3la b3d kda
                            cv2.imwrite(os.path.join(action_output_dir, f'{action_name}_Frame_{frame_num//5}{postName}.png'), crop_img) # change alrkm ale able #.png
                            # Group number, person number fooo2 name postName.
                            #####################################################################3
                        else :
                            print("No Face Detected")
                            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(img)
                            label_text = f'{action_name}_Frame_{frame_num}'
                            draw.text((10, 10), label_text, fill=(255, 255, 255))
                            img.save(f'{output_dir}{action_name}_Frame_{frame_num}.png')
                        # Replace cv2.waitKey with a delay (25 milliseconds)
                        time.sleep(0.025)

# Release video capture object
cap.release()
