import streamlit as st
import cv2
import tempfile
import numpy as np
from pose_utils import analyze_pose_video  # ✅ import the new function

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

    st.video(tfile.name)  # ✅ Show the uploaded video

    st.write("Analyzing video, please wait...")

    # ✅ Analyze full video
    feedback = analyze_pose_video(tfile.name)

    # ✅ Display feedback
    st.success("✅ Analysis complete!")
    st.markdown("### Summary Feedback:")
    st.text(feedback)
