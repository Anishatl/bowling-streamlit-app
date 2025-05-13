import streamlit as st
import cv2
import tempfile
import numpy as np
from pose_utils import analyze_pose_video, analyze_pose
from PIL import Image

# Title and description
st.title("🏏 Bowling Action Analyzer")
st.write("Upload a video of your bowling action to get started.")

# Session state setup
if "angle_data" not in st.session_state:
    st.session_state.angle_data = None

if "frames" not in st.session_state:
    st.session_state.frames = []

if "rotated_frames" not in st.session_state:
    st.session_state.rotated_frames = []

if "rotation_angle" not in st.session_state:
    st.session_state.rotation_angle = 0

# Upload video
video_file = st.file_uploader("Upload Bowling Video", type=["mp4", "mov", "avi"])

# Rotate video
if st.button("🔁 Rotate Frame 90°"):
    st.session_state.rotation_angle = (st.session_state.rotation_angle + 90) % 360
    rotated = []
    for f in st.session_state.frames:
        img = Image.fromarray(f)
        rotated_img = img.rotate(-st.session_state.rotation_angle, expand=True)
        rotated.append(np.array(rotated_img))
    st.session_state.rotated_frames = rotated

# Process video
if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    if not st.session_state.angle_data:
        # Analyze once per video upload
        with st.spinner("Analyzing full video..."):
            angle_data, every_nth_frame = analyze_pose_video(tfile.name)
            st.session_state.angle_data = angle_data
            st.session_state.frames = every_nth_frame
            st.session_state.rotated_frames = every_nth_frame  # Initially not rotated

        st.success("✅ Analysis complete!")

    # Show full video feedback
    st.subheader("📊 Full Video Feedback")
    for key, values in st.session_state.angle_data.items():
        avg = np.mean(values)
        st.write(f"Average {key.replace('_', ' ').title()}: {avg:.2f}°")

    # Frame navigation
    st.subheader("🖼️ Frame-by-Frame Pose View")
    total_frames = len(st.session_state.rotated_frames)
    if total_frames > 0:
        frame_idx = st.slider("Select Frame", 0, total_frames - 1, 0)

        frame = st.session_state.rotated_frames[frame_idx]
        annotated_frame, _ = analyze_pose(frame, draw_angles=True)
        st.image(annotated_frame, channels="RGB", use_container_width=False)
