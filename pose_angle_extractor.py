import cv2
import csv
import mediapipe as mp
import numpy as np
from pose_utils import calculate_angle  # or define calculate_angle here if not importing
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def extract_pose_angles_to_csv(video_path, output_csv_path, frame_skip=5):
    import os
    print("File exists:", os.path.exists(video_path))

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "elbow_angle", "spine_angle", "knee_angle", "shoulder_angle"])

        frame_counter = 0
        successful_detections = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue

            # Resize for better detection (optional)
            frame = cv2.resize(frame, (640, 480))

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                def get_point(landmark): return (landmark.x, landmark.y)

                try:
                    r_shoulder = get_point(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
                    r_elbow = get_point(lm[mp_pose.PoseLandmark.RIGHT_ELBOW])
                    r_wrist = get_point(lm[mp_pose.PoseLandmark.RIGHT_WRIST])
                    r_hip = get_point(lm[mp_pose.PoseLandmark.RIGHT_HIP])
                    r_knee = get_point(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
                    r_ankle = get_point(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
                    l_hip = get_point(lm[mp_pose.PoseLandmark.LEFT_HIP])
                    l_shoulder = get_point(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])

                    elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                    spine_angle = calculate_angle(l_hip, r_hip, r_shoulder)
                    knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                    shoulder_angle = calculate_angle(r_elbow, r_shoulder, r_hip)

                    writer.writerow([
                        frame_counter,
                        round(elbow_angle, 2),
                        round(spine_angle, 2),
                        round(knee_angle, 2),
                        round(shoulder_angle, 2)
                    ])
                    successful_detections += 1

                except Exception as e:
                    print(f"Frame {frame_counter}: Error calculating angles - {e}")
            else:
                print(f"Frame {frame_counter}: No pose landmarks detected")

    cap.release()
    print(f"Finished. Successful pose detections: {successful_detections}")
