# Export per-frame pose angles from a bowling video to a CSV file.
# Usage: python run_extractor.py path/to/video.mp4 [output.csv]
import sys

from pose_angle_extractor import extract_pose_angles_to_csv

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python run_extractor.py path/to/video.mp4 [output.csv]")

    video_path = sys.argv[1]
    csv_output_path = sys.argv[2] if len(sys.argv) > 2 else "output_angles.csv"

    extract_pose_angles_to_csv(video_path, csv_output_path, frame_skip=5)
    print(f"Pose angles saved to {csv_output_path}")
