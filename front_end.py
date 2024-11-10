import streamlit as st
import tempfile
import cv2
import numpy as np
import torch
from sort import Sort  # Make sure to include your tracking library, like SORT
import pathlib
import os
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load the YOLOv5 model
@st.cache_data
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt", force_reload=True)
    model.eval()
    return model

model = load_model()

# Function to process and track objects in video
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    tracker = Sort()
    cattle_cnt = 0
    while cap.isOpened():
        cattle_cnt_temp = 0
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLOv5 detection on the frame
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()  # Convert detections to NumPy array

        # Prepare detections for the tracker
        tracker_input = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf > 0.3:  # Confidence threshold
                tracker_input.append([x1, y1, x2, y2, conf])
                cattle_cnt_temp += 1

        #Update cattle count
        cattle_cnt = max(cattle_cnt, cattle_cnt_temp)
        
        # Update tracker and get tracked objects
        tracked_objects = tracker.update(np.array(tracker_input))

        # Draw bounding boxes and object IDs on the frame
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {int(obj_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the processed frame to the output video
        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
        out.write(frame)

    cap.release()
    out.release()
    return cattle_cnt

# Function to process and track objects in images
def process_image(image):
    results = model(image)
    detections = results.xyxy[0].cpu().numpy()  # Convert detections to NumPy array
    cattle_cnt = 0
    
    # Draw bounding boxes on the image
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.3:  # Confidence threshold
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f'Class: {int(cls)} Conf: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cattle_cnt += 1
    return image, cattle_cnt

# Streamlit UI
st.title("Cattle Grazing Detector")
option = st.selectbox("Upload an image (jpg, jpeg, png) or video (mp4, avi, mov) for detection", ["Image", "Video"])

uploaded_file = st.file_uploader("Upload a file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    if option == "Image" and uploaded_file.type.startswith("image"):
        # Process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        processed_image, cattle_cnt = process_image(image)

        # Display the processed image
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
        st.header(f'There are {cattle_cnt} cattles detected. ')
        
    elif option == "Video" and uploaded_file.type.startswith("video"):
        if 'video_processed' not in st.session_state:
            st.session_state['video_processed'] = False
        # Save the uploaded video to a temporary file
        temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input_file.write(uploaded_file.read())
        temp_input_file.close()

        # Create a temporary file for the output video
        temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_output_file.close()

        # Process the video
        if not st.session_state['video_processed']:
            with st.spinner("Processing video..."):
                st.session_state["cattle_cnt"] = process_video(temp_input_file.name, temp_output_file.name)
            st.success("Video processing complete.")
            st.session_state['output_video_path'] = temp_output_file.name
            st.session_state['video_processed'] = True
        
        # Display the processed video
        st.write("Video path:", st.session_state['output_video_path'])
        if os.path.exists(st.session_state['output_video_path']):
            st.video(st.session_state['output_video_path'])
        else:
            st.error("Video file not found.")
        st.header(f'There are {st.session_state["cattle_cnt"]} cattles detected. ')
        # Provide download link
        with open(st.session_state['output_video_path'], 'rb') as f:
            st.download_button("Download Processed Video", f, file_name="processed_video.mp4", mime="video/mp4")
        
        # os.remove(temp_input_file.name)
        # os.remove(temp_output_file.name)
    else:
        st.error("Please upload a valid image or video file.")


