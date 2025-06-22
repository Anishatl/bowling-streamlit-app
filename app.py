import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
from pose_utils import analyze_pose_video, analyze_pose
from model_loader import load_all_models

# Load all models once
models = load_all_models()
clip_model = models["clip_model"]
joint_models = models["joint_models"]
scaler = models["clip_scaler"]
clip_label_encoder = models["clip_label_encoder"]
joint_label_encoder = models["joint_label_encoder"]

# Advice map and angle thresholds for feedback
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

        # Run full video analysis (pose angles per frame)
        if 'full_video_feedback' not in st.session_state:
            full_video_feedback = cached_video_analysis(tfile.name)
            st.session_state.full_video_feedback = full_video_feedback
        else:
            full_video_feedback = st.session_state.full_video_feedback

        full_video_analysis_container.write("### Full Video Analysis (frame-level angles):")
        full_video_analysis_container.write(full_video_feedback)

        # ----------- CLIP-LEVEL PREDICTION -----------
        angle_features = ["elbow_angle", "spine_angle", "knee_angle", "shoulder_angle"]
        if isinstance(full_video_feedback, list) and len(full_video_feedback) > 0:
            df_clip = pd.DataFrame(full_video_feedback)

            if all(col in df_clip.columns for col in angle_features):
                # Mean angles per clip
                clip_avg = df_clip[angle_features].mean().values.reshape(1, -1)
                clip_scaled = scaler.transform(clip_avg)
                pred = clip_model.predict(clip_scaled)
                pred_label = clip_label_encoder.inverse_transform(pred)[0]

                st.subheader("🩺 Clip-Level Injury Risk Prediction:")
                st.markdown(f"**Predicted Risk Level: {pred_label}**")

                # Joint-level risk predictions & advice
                st.subheader("🦵 Joint-Level Injury Risk Analysis:")
                joint_risks = {}
                advice_given = False
                for joint in ["elbow", "spine", "knee", "shoulder"]:
                    angle_val = clip_avg[0][angle_features.index(f"{joint}_angle")]
                    # Joint model expects 2D input
                    joint_model = joint_models.get(joint)
                    if joint_model is not None:
                        joint_pred_encoded = joint_model.predict(np.array([[angle_val]]))
                        joint_pred_label = joint_label_encoder.inverse_transform(joint_pred_encoded)[0]
                        joint_risks[joint] = (joint_pred_label, angle_val)
                        # Show advice if risky or severe
                        if joint_pred_label in ["Risky", "Severe"]:
                            low, high = THRESHOLDS[f"{joint}_angle"]
                            if not (low <= angle_val <= high):
                                st.markdown(f"- **{joint.capitalize()}**: {ADVICE_MAP[joint]} (Angle: {angle_val:.1f}°)")
                                advice_given = True
                    else:
                        st.warning(f"No joint model found for {joint}")

                if not advice_given:
                    st.markdown("All joints are within safe biomechanical ranges!")

            else:
                st.warning("Clip-level model could not run due to missing angle data.")
        else:
            st.warning("Insufficient frame-level angle data to evaluate clip.")

    # --- Individual Frame Selection ---
    if len(frames) > 0:
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
    else:
        st.info("No frames extracted from video to display.")

