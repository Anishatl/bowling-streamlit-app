import cv2
import mediapipe as mp
import numpy as np

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

# Try the standard 'mp.solutions' API first, fall back if needed
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing

# Initialize MediaPipe Pose Model
pose = mp_pose.Pose()

#Analyze a single frame
def analyze_pose(frame, draw_angles=False):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    feedback = ""
    debug_text = ""

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        r_shoulder = (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
        r_elbow = (lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y)
        r_wrist = (lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y)

        debug_text += f"Right Shoulder (norm): {r_shoulder}\n"
        debug_text += f"Right Elbow (norm): {r_elbow}\n"
        debug_text += f"Right Wrist (norm): {r_wrist}\n"

        elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        debug_text += f"Elbow Angle: {elbow_angle:.2f}°\n"

        if draw_angles:
            h, w, _ = frame.shape
            r_shoulder_px = (int(r_shoulder[0]*w), int(r_shoulder[1]*h))
            r_elbow_px = (int(r_elbow[0]*w), int(r_elbow[1]*h))
            r_wrist_px = (int(r_wrist[0]*w), int(r_wrist[1]*h))

            cv2.circle(frame, r_shoulder_px, 5, (0,255,0), -1)
            cv2.circle(frame, r_elbow_px, 5, (0,255,0), -1)
            cv2.circle(frame, r_wrist_px, 5, (0,255,0), -1)

    else:
        debug_text = "No pose landmarks detected."

    return frame, feedback, debug_text


#Analyze full video and return list of per-frame angles
def analyze_pose_video(video_path):
    cap = cv2.VideoCapture(video_path)

    angles_list = []
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1
        if frame_counter % 5 != 0:
            continue

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

            angles_list.append({
                "elbow_angle": elbow_angle,
                "spine_angle": spine_angle,
                "knee_angle": knee_angle,
                "shoulder_angle": shoulder_angle
            })

    cap.release()
    return angles_list
