import streamlit as st
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Ensure this library is installed for object tracking
import pathlib
import os
import torch

"""ONLY UNCOMMENT FOR WINDOWS LOCAL RUN"""
# Ensure compatibility with Windows paths
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

def check_weights_path(weights_path):
    """
    Check if the weights file exists and is accessible using platform-agnostic path handling
    Returns the full path if found, None otherwise
    """
    # Convert string path to Path object
    weights_path = pathlib.Path(weights_path)
    
    # List of possible paths to check using Path objects
    possible_paths = [
        weights_path,  # Original path
        pathlib.Path.cwd() / weights_path,  # Full path from current directory
        pathlib.Path.cwd() / 'weights' / 'yolov5l.pt',  # Explicit weights directory
        pathlib.Path('/mount/src/cattle_graze_model/weights/yolov5l.pt'),  # Absolute path
    ]
    
    for path in possible_paths:
        if path.is_file():
            return path  # Convert back to string for YOLO
    return None

@st.cache_data
def load_model():
    """
    Load the YOLO model with improved error handling
    """
    try:
        # Default weights path
        weights_path = 'yolov5l.pt'
        valid_path = 'weights/yolov5l.pt'
        if not os.path.isfile(valid_path):
            st.error(f""" Valid Path Failed
                     Current Directory: {os.getcwd()}
                     All files in directory: {os.listdir(os.getcwd())}
                     """)
        # # Check if weights file exists
        # valid_path = check_weights_path(weights_path)
        
        # if valid_path is None:
        #     st.error(f"""
        #         Could not find weights file. Please ensure:
        #         1. The weights file 'yolov5l.pt' exists in the 'weights' directory
        #         2. The file has correct permissions
        #         3. The full path is accessible
                
        #         Checked paths:
        #         - {weights_path},  # Original path
        #         - {pathlib.Path.cwd() / weights_path},  # Full path from current directory
        #         - {pathlib.Path.cwd() / 'weights' / 'yolov5l.pt'},  # Explicit weights directory
        #         - {pathlib.Path('/mount/src/cattle_graze_model/weights/yolov5l.pt')}
                
        #     """)
        #     return None
            
        # Load the model
        model = YOLO(valid_path)
        st.success("Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"""
            valid_path = {str(valid_path)}
            Error loading model: {str(e)}
            
            Current working directory: {os.getcwd()}
            Python path: {os.environ.get('PYTHONPATH', 'Not set')}
        """)
        return None

# Update the model loading call
model = load_model()
if model is None:
    st.error("""
        Failed to load the model. Please check:
        1. The model weights file is present
        2. You have sufficient permissions
        3. There is enough memory available
    """)
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

        # Run YOLO detection on the frame
        results = model.predict(frame, stream=True)
        
        tracker_input = []
        for result in results:
            for box in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box
                if conf > 0.3:  # Confidence threshold
                    tracker_input.append([x1, y1, x2, y2, conf])
                    cattle_cnt_temp += 1

        # Update cattle count
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
    results = model.predict(image)
    cattle_cnt = 0
    
    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box
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
        st.header(f'There are {cattle_cnt} cattles detected.')

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
        if os.path.exists(st.session_state['output_video_path']):
            st.video(st.session_state['output_video_path'])
        else:
            st.error("Video file not found.")
        
        st.header(f'There are {st.session_state["cattle_cnt"]} cattles detected.')

        # Provide download link
        with open(st.session_state['output_video_path'], 'rb') as f:
            st.download_button("Download Processed Video", f, file_name="processed_video.mp4", mime="video/mp4")

    else:
        st.error("Please upload a valid image or video file.")


