from pose_angle_extractor import extract_pose_angles_to_csv

if __name__ == "__main__":
    video_path = "C:\Users\anish\Downloads\Bowling_Training_Data/Mark_Wood_Video.mp4"  # Update this to your local video file path
    csv_output_path = "output_angles.csv"

    extract_pose_angles_to_csv(video_path, csv_output_path, frame_skip=5)
    print(f"Pose angles saved to {csv_output_path}")
