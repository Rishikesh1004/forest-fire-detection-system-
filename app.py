import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import numpy as np

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # Replace with the path to your trained model

# Streamlit UI
st.set_page_config(page_title="Forest Fire Detection", layout="centered")
st.title("🌲🔥 Forest Fire Detection App")
st.write("Upload an image or video to detect forest fires using YOLOv8.")

option = st.selectbox("Choose input type", ("Image", "Video"))

if option == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to OpenCV format
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Run YOLOv8 inference
        results = model(image_cv)[0]

        # Plot the results on the image
        annotated_frame = results.plot()
        st.image(annotated_frame, caption="Detection Result", use_column_width=True)

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            results = model(frame)[0]
            annotated_frame = results.plot()

            # Convert annotated frame to RGB and show in Streamlit
            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()
        os.remove(video_path)
