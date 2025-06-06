import csv
import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    """Calculate angle between three points in degrees."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

def extract_pose_angles_to_csv(video_path, csv_path, frame_skip=5):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(video_path)
    frame_counter = 0

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame', 'elbow_angle', 'spine_angle', 'knee_angle', 'shoulder_angle'])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1

            if frame_counter % frame_skip == 0:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark

                    def get_point(landmark):
                        return (landmark.x, landmark.y)

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

                    writer.writerow([frame_counter, elbow_angle, spine_angle, knee_angle, shoulder_angle])

    cap.release()
    pose.close()
