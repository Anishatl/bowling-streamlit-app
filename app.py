import streamlit as st
import cv2
import tempfile
import numpy as np
from pose_utils import analyze_pose_video, analyze_pose

# Title and description
st.title("🏏 Bowling Action Analyzer")
st.markdown(
    """
    Upload a video file to analyze the bowler's action frame by frame.
    This app will detect and analyze key pose angles to give feedback on the bowler's form.
    """
)

# File uploader
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Saving the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Display feedback message
    st.write("Analyzing video, please wait...")

    # Load video with OpenCV to extract frames
    cap = cv2.VideoCapture(tfile.name)
    frames = []
    frame_count = 0

    # Read video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
    cap.release()

    st.write(f"Total frames in the video: {frame_count}")

    # Show frame slider every 5 frames
    # Display frames at every 5th index
    frame_slider = st.slider("Select a frame", 0, frame_count - 1, 0, step=5)

    # Select the frame and analyze pose
    selected_frame = frames[frame_slider]

    # Run pose analysis on the selected frame
    analyzed_frame, feedback = analyze_pose(selected_frame, draw_angles=True)

    # Show the analyzed frame
    st.image(analyzed_frame, channels="RGB", use_container_width=True)

    # Show feedback
    st.write(feedback)

    # Option to analyze the full video and get a summary
    analyze_full_video = st.button("Analyze Full Video")

    if analyze_full_video:
        feedback = analyze_pose_video(tfile.name)
        st.write("### Full Video Analysis:")
        st.write(feedback)
