import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
import joblib
from pose_utils import analyze_pose_video, analyze_pose

# ----------------------------
# Load trained models & config
# ----------------------------
clip_model = joblib.load("clip_level_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

ADVICE_MAP = {
    "elbow": "Try to reduce hyperextension during delivery.",
    "knee": "Avoid excessive bending or locking of the front knee.",
    "shoulder": "Reduce shoulder abduction and external rotation.",
    "spine": "Maintain a more upright posture to reduce spine stress."
}

THRESHOLDS = {
    "elbow_angle": (70, 150),
    "spine_angle": (30, 90),
    "knee_angle": (90, 170),
    "shoulder_angle": (10, 90),
}

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🏏 Bowling Action Analyzer")
st.markdown("""
Upload a video file to analyze the bowler's action frame by frame.

This app detects and analyzes key pose angles and provides feedback on injury risk and how to improve form.
""")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

# Temporary container for full video analysis output
full_video_analysis_container = st.empty()

# Cache the analysis result to avoid recalculating
@st.cache_resource
def cached_video_analysis(file_path):
    return analyze_pose_video(file_path)

if 'rotation_angle' not in st.session_state:
    st.session_state.rotation_angle = 0  # Default rotation

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    with st.spinner('Analyzing video, please wait...'):
        # Read frames from video (every 5th frame)
        cap = cv2.VideoCapture(tfile.name)
        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 5 == 0:
                frames.append(frame)
            frame_count += 1
        cap.release()

        st.write(f"Total frames in the video: {frame_count}")
        st.write(f"Frames being analyzed (every 5th frame): {len(frames)}")

        # Run full video analysis
        if 'full_video_feedback' not in st.session_state:
            full_video_feedback = cached_video_analysis(tfile.name)
            st.session_state.full_video_feedback = full_video_feedback
            full_video_analysis_container.write("### Full Video Analysis:")
            full_video_analysis_container.write(full_video_feedback)
        else:
            full_video_analysis_container.write("### Full Video Analysis:")
            full_video_analysis_container.write(st.session_state.full_video_feedback)

        # ---- CLIP-LEVEL PREDICTION + ADVICE ----
        angle_features = ["elbow_angle", "spine_angle", "knee_angle", "shoulder_angle"]
        full_video_feedback = st.session_state.full_video_feedback

        if isinstance(full_video_feedback, list) and len(full_video_feedback) > 0:
            df_clip = pd.DataFrame(full_video_feedback)
            if all(col in df_clip.columns for col in angle_features):
                clip_avg = df_clip[angle_features].mean().values.reshape(1, -1)
                clip_scaled = scaler.transform(clip_avg)
                pred = clip_model.predict(clip_scaled)
                pred_label = label_encoder.inverse_transform(pred)[0]

                st.subheader("🩺 Clip-Level Injury Risk Prediction:")
                st.markdown(f"**Predicted Risk Level: {pred_label}**")

                if pred_label in ["Risky", "Severe"]:
                    st.subheader("📌 Injury Advice:")
                    for angle in angle_features:
                        val = clip_avg[0][angle_features.index(angle)]
                        low, high = THRESHOLDS[angle]
                        if not (low <= val <= high):
                            joint = angle.split("_")[0]
                            st.markdown(f"- **{joint.capitalize()}**: {ADVICE_MAP[joint]}")
            else:
                st.warning("Clip-level model could not run due to missing angle data.")
        else:
            st.warning("Insufficient frame-level angle data to evaluate clip.")

    # --- Individual Frame Selection ---
    frame_slider = st.slider("Select a frame", 0, len(frames) - 1, 0)
    selected_frame = frames[frame_slider]

    # Rotate frame
    if st.button("Rotate Frame 90°"):
        st.session_state.rotation_angle = (st.session_state.rotation_angle + 90) % 360

    if st.session_state.rotation_angle == 90:
        selected_frame = cv2.rotate(selected_frame, cv2.ROTATE_90_CLOCKWISE)
    elif st.session_state.rotation_angle == 180:
        selected_frame = cv2.rotate(selected_frame, cv2.ROTATE_180)
    elif st.session_state.rotation_angle == 270:
        selected_frame = cv2.rotate(selected_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Run pose analysis on selected frame
    analyzed_frame, feedback, debug_text = analyze_pose(selected_frame, draw_angles=True)

    st.image(analyzed_frame, channels="BGR", use_container_width=True)
    st.text_area("Pose Angles Debug Info", debug_text, height=100)
    st.write("Feedback:", feedback)
