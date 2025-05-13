import streamlit as st
import cv2
import tempfile
import numpy as np
from pose_utils import analyze_pose_video, analyze_pose

# Title and description
st.title("🏏 Bowling Action Analyzer")
st.write("Upload a video of a cricket bowling action, and the system will analyze the pose frame by frame.")

# File uploader for video input
uploaded_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])

# Display video feedback
if uploaded_file is not None:
    # Write the video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    # ✅ Analyze the full video for pose and angles
    st.write("Analyzing video, please wait...")
    feedback = analyze_pose_video(video_path)

    # ✅ Display feedback for the full video
    st.success("✅ Analysis complete!")
    st.text(feedback)

    # Optionally show the video with pose annotations frame by frame
    show_video = st.checkbox("Show video with frame-by-frame pose analysis")

    if show_video:
        # Open video using OpenCV
        cap = cv2.VideoCapture(video_path)
        frame_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Analyze pose for the current frame
            annotated_frame, _ = analyze_pose(frame, draw_angles=True)

            # Convert the frame to RGB for displaying in Streamlit
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame_rgb)

        cap.release()

        # Display video frames
        for frame in frame_list:
            st.image(frame, channels="RGB", use_container_width=True)
else:
    st.write("Please upload a video to analyze.")
