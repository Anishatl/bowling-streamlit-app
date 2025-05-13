import streamlit as st
import cv2
import tempfile
import numpy as np
from pose_utils import analyze_pose_video, analyze_pose
from PIL import Image

# Session state for rotation and frame size
if "rotation" not in st.session_state:
    st.session_state.rotation = 0
if "frame_width" not in st.session_state:
    st.session_state.frame_width = 400  # default

st.title("🏏 Bowling Action Analyzer")
st.write("Upload a bowling video to analyze its pose every 5 frames.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Load and analyze video only once
    if "feedback" not in st.session_state or "frames" not in st.session_state:
        st.session_state.feedback, st.session_state.frames = analyze_pose_video(tfile.name)

    feedback = st.session_state.feedback
    all_frames = st.session_state.frames

    st.text_area("Full Video Analysis", value=feedback, height=200)

    # Rotation button
    if st.button("🔄 Rotate Frame"):
        st.session_state.rotation = (st.session_state.rotation + 90) % 360

    # Frame size slider
    st.session_state.frame_width = st.slider(
        "Adjust Frame Width", min_value=100, max_value=800,
        value=st.session_state.frame_width, step=10
    )

    # Frame selector
    selected_frame_index = st.slider(
        "Select Frame", 0, len(all_frames) - 1, 0, step=1
    )

    # Display selected frame with rotation and resizing
    selected_frame = all_frames[selected_frame_index]
    if st.session_state.rotation != 0:
        # Rotate while keeping text upright
        pil_img = Image.fromarray(cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB))
        pil_img = pil_img.rotate(-st.session_state.rotation, expand=True)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB))

    # Resize while maintaining aspect ratio
    width = st.session_state.frame_width
    ratio = width / pil_img.width
    height = int(pil_img.height * ratio)
    pil_img = pil_img.resize((width, height))

    st.image(pil_img, use_container_width=False)
