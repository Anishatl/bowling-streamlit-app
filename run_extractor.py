#Test file to test extracting poses imported to a CSV, using video file of Mark Wood bowling
from pose_angle_extractor import extract_pose_angles_to_csv

if __name__ == "__main__":
    video_path = "C:/Users/anish/Downloads/wood.mp4"

    csv_output_path = "output_angles.csv"

    extract_pose_angles_to_csv(video_path, csv_output_path, frame_skip=5)
    print(f"Pose angles saved to {csv_output_path}")
