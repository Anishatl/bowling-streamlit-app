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

# Temporary container for full video analysis
full_video_analysis_container = st.empty()

# Cache the analysis result to avoid recalculating every time
@st.cache_resource
def cached_video_analysis(file_path):
    return analyze_pose_video(file_path)

if uploaded_file is not None:
    # Saving the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Create a loading spinner
    with st.spinner('Analyzing video, please wait...'):
        # Load video with OpenCV to extract frames
        cap = cv2.VideoCapture(tfile.name)
        frames = []
        frame_count = 0

        # Read video frames, but only store every 5th frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Only store every 5th frame
            if frame_count % 5 == 0:
                frames.append(frame)
            frame_count += 1
        cap.release()

        st.write(f"Total frames in the video: {frame_count}")
        st.write(f"Frames being analyzed (every 5th frame): {len(frames)}")

        # Option to analyze the full video (this section will remain at the top)
        if 'full_video_feedback' not in st.session_state:
            # Run the analysis only once, then store it in session state
            full_video_feedback = cached_video_analysis(tfile.name)
            st.session_state.full_video_feedback = full_video_feedback
            full_video_analysis_container.write("### Full Video Analysis:")
            full_video_analysis_container.write(full_video_feedback)
        else:
            # If analysis was already done, simply display it
            full_video_analysis_container.write("### Full Video Analysis:")
            full_video_analysis_container.write(st.session_state.full_video_feedback)

    # Show frame slider with the reduced number of frames (every 5th frame)
    frame_slider = st.slider("Select a frame", 0, len(frames) - 1, 0)

    # Select the frame and analyze pose
    selected_frame = frames[frame_slider]

    # Run pose analysis on the selected frame
    analyzed_frame, feedback = analyze_pose(selected_frame, draw_angles=True)

    # Show the analyzed frame
    st.image(analyzed_frame, channels="RGB", use_container_width=True)

    # Show feedback for the selected frame
    st.write(feedback)
