import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from pose_utils import analyze_pose_video, analyze_pose

# Initialize session state
if "rotation" not in st.session_state:
    st.session_state.rotation = 0
if "frame_width" not in st.session_state:
    st.session_state.frame_width = 400
if "all_frames" not in st.session_state:
    st.session_state.all_frames = []
if "video_analyzed" not in st.session_state:
    st.session_state.video_analyzed = False
if "analysis_feedback" not in st.session_state:
    st.session_state.analysis_feedback = ""

st.title("🏏 Bowling Action Analyzer")

uploaded_file = st.file_uploader("Upload a bowling video", type=["mp4", "mov", "avi"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Only analyze video once
    if not st.session_state.video_analyzed:
        feedback, frames = analyze_pose_video(tfile.name, every_n_frames=5)
        st.session_state.analysis_feedback = feedback
        st.session_state.all_frames = frames
        st.session_state.video_analyzed = True

    # Display feedback
    st.markdown("## 📊 Full Video Analysis")
    st.success(st.session_state.analysis_feedback)

    st.markdown("---")
    st.markdown("## 🖼️ Frame-by-Frame Viewer")

    all_frames = st.session_state.all_frames
    total_frames = len(all_frames)

    if total_frames > 0:
        selected_frame_index = st.slider(
            "Select Frame", 0, total_frames - 1, 0, 1
        )

        # Rotation button
        if st.button("🔄 Rotate Frame"):
            st.session_state.rotation = (st.session_state.rotation + 90) % 360

        # Get the selected frame
        selected_frame = all_frames[selected_frame_index]

        # Convert to PIL for rotation
        pil_img = Image.fromarray(cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB))

        # Apply rotation
        if st.session_state.rotation != 0:
            pil_img = pil_img.rotate(-st.session_state.rotation, expand=True)

        # Resize
        width = st.session_state.frame_width
        ratio = width / pil_img.width
        height = int(pil_img.height * ratio)
        pil_img = pil_img.resize((width, height))

        # Show the image
        st.image(pil_img, use_container_width=False)

        # 👇 Width slider directly under image
        st.session_state.frame_width = st.slider(
            "Adjust Frame Width",
            min_value=100,
            max_value=800,
            value=st.session_state.frame_width,
            step=10,
        )
    else:
        st.warning("No frames extracted.")
