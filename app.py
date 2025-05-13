import cv2
import streamlit as st
from pose_utils import analyze_pose
from tqdm import tqdm

# === Existing Code for Streamlit UI ===

# Streamlit: Title and input video file
st.title("Cricket Bowling Pose Analysis")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Additional Streamlit code for other features (e.g., pose analysis, injury feedback, etc.)
# Example: Add other Streamlit features here

# === New Code for Video Annotation ===

def process_and_annotate_video(input_path, output_path, show_live=False):
    cap = cv2.VideoCapture(input_path)

    # Get original video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=frame_count, desc="Processing Frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Analyze pose and draw angles
            processed_frame, _ = analyze_pose(frame, draw_angles=True)

            # Write frame to output video
            out.write(processed_frame)

            # Optionally display video live
            if show_live:
                cv2.imshow('Pose Annotation', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            pbar.update(1)

    cap.release()
    out.release()
    if show_live:
        cv2.destroyAllWindows()

    print("✅ Done: Annotated video saved to", output_path)

# === Streamlit: Handling the video upload and processing ===
if uploaded_video is not None:
    # Save the uploaded video to a file
    input_video_path = f"temp_video/{uploaded_video.name}"
    with open(input_video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    # Option to show live preview during processing
    show_live = st.checkbox("Show live preview during processing", value=False)

    # Output path for annotated video
    output_video_path = f"temp_video/annotated_{uploaded_video.name}"

    # Run the video annotation process when the user presses the button
    if st.button("Process Video"):
        st.text("Processing your video... please wait.")
        process_and_annotate_video(input_video_path, output_video_path, show_live)

        # After processing, provide the download link
        st.text("Processing complete!")
        st.video(output_video_path)  # Streamlit video display
