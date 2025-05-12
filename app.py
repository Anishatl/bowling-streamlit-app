import streamlit as st
import cv2
import tempfile
import numpy as np
from pose_utils import analyze_pose

st.title("🏏 Bowling Action Analyzer")

st.write("""
Upload a video of your bowling action.  
The app will analyze your pose and give feedback on injury risk.
""")

video_file = st.file_uploader("Upload your bowling video", type=["mp4", "mov", "avi"])

if video_file:
    # Save video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame using MediaPipe pose analysis
        annotated_frame, feedback = analyze_pose(frame)

        stframe.image(annotated_frame, channels="BGR")

    cap.release()
    st.success("✅ Analysis complete!")
    st.markdown(f"### Feedback:\n{feedback}")
