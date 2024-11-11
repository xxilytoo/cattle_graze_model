import streamlit as st

# Page Title
st.title("About the Project: Training the YOLOv11 Model for Cattle Detection")

# Objective Section
st.header("Objective")
st.write("""
The primary objective of this project was to develop a robust and efficient object detection model capable of identifying cattle in aerial images and video footage. This tool is intended to assist farmers and researchers in monitoring grazing patterns, assessing herd distribution, and ensuring the well-being of livestock.
""")

# Model Overview Section
st.header("Model Overview")
st.write("""
The project leverages a fine-tuned version of the YOLOv11 (You Only Look Once) architecture. YOLOv11 is known for its fast and accurate object detection capabilities, making it a suitable choice for real-time cattle identification from aerial views. The model was trained to detect cattle and distinguish them from other objects in diverse environments, including various terrains and lighting conditions.
""")

# Data Collection and Preparation Section
st.header("Data Collection and Preparation")
st.write("""
To train the YOLOv11 model effectively, a diverse and well-labeled dataset was crucial. The dataset included:
- **High-resolution aerial images and video frames**: Captured using drones and satellites to simulate real-world monitoring scenarios.
- **Annotations**: Each image was meticulously labeled with bounding boxes that identified cattle positions, ensuring that the model could learn to differentiate between cattle and background elements.
""")
st.write("""
Pre-processing steps involved:
- **Resizing** images to match the model’s input size.
- **Augmentation techniques**: Including random rotations, flips, and color adjustments to improve the model's generalization capabilities.
""")

# Training Methodology Section
st.header("Training Methodology")
st.write("""
The training process involved the following key steps:

1. **Initial Configuration**:
   - Configuring the YOLOv11 model with custom layers suited for cattle detection.
   - Setting up the training pipeline to run on high-performance GPUs for accelerated training.

2. **Hyperparameter Tuning**:
   - Optimizing learning rates, batch sizes, and anchor boxes.
   - Employing techniques like early stopping and learning rate schedulers to prevent overfitting.

3. **Fine-Tuning**:
   - Transfer learning was utilized by initializing with pre-trained weights on a large dataset, allowing the model to leverage existing knowledge.
   - Fine-tuning was done using our custom cattle dataset to adapt the model for specific detection requirements.
""")

# Model Performance and Evaluation Section
st.header("Model Performance and Evaluation")
st.write("""
The model was evaluated using metrics such as:
- **Precision and Recall**: To measure how accurately the model detected cattle.
- **Mean Average Precision (mAP)**: Ensured that the model maintained high performance across various thresholds.
- **Inference Time**: Tested for real-time performance to confirm usability in practical applications.
""")

st.subheader("Training Metrics")
st.write("Below is an illustration of the training metrics over the course of the training epochs:")
st.image("pictures/Yolo11ft_metrics.png")

st.subheader("Confusion Matrix")
st.write("The confusion matrix below highlights the model's performance in distinguishing between cattle and the background:")
st.image("pictures/Yolo11ft_Confusion_Matrix.png")

# Challenges and Solutions Section
st.header("Challenges and Solutions")
st.write("""
- **Varied Image Quality**: Addressed by training the model on images with different resolutions and using augmentation.
- **Background Complexity**: Enhanced detection robustness by training with diverse backgrounds to minimize false positives.
""")

# Future Improvements Section
st.header("Future Improvements")
st.write("""
Moving forward, potential improvements include:
- **Expanding the dataset** with more challenging scenarios and other livestock types.
- **Incorporating additional model features** for real-time tracking and analysis.
- **Deploying on cloud platforms** to provide scalable and accessible tools for end-users.
""")

# Credits section
st.header("Data source credits")
st.write("Data used to train from: https://universe.roboflow.com/detection-kvliu/cow_detect")

st.write("""
This Streamlit application enables users to upload and process their own aerial images and videos for cattle detection, demonstrating the model's effectiveness and practical use in the field.
""")


