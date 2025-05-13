import streamlit as st
import cv2
import tempfile
import numpy as np
from pose_utils import analyze_pose_video, analyze_pose
from PIL import Image

# Title and description
st.title("🏏 Bowling Action Analyzer")
st.write("Upload a bowling video to analyze its pose frame-by-frame.")

# Upload video
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
if uploaded_file is not None:
    # Open the video file using OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    # Get the original dimensions of the video
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Frame size adjustment slider
    frame_width = st.slider("Select Frame Width", min_value=100, max_value=original_width, value=original_width, step=10)

    # Calculate the new height while keeping the aspect ratio
    aspect_ratio = original_width / original_height
    frame_height = int(frame_width / aspect_ratio)

    # Display full video analysis
    feedback = analyze_pose_video(tfile.name)
    st.text_area("Full Video Analysis", value=feedback, height=200)

    # Frame selection slider
    num_frames = len(feedback.splitlines())  # Assuming feedback corresponds to frames
    selected_frame_index = st.slider("Select a Frame", min_value=0, max_value=num_frames - 1, value=0)

    # Load the selected frame and resize it based on the slider value
    cap = cv2.VideoCapture(tfile.name)
    for _ in range(selected_frame_index):
        cap.read()  # Skip to the selected frame
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Resize the frame according to the selected width while maintaining the aspect ratio
        resized_frame = cv2.resize(frame, (frame_width, frame_height))
        st.image(resized_frame, channels="BGR", use_container_width=True)

