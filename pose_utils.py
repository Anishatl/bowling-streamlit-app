import cv2
import mediapipe as mp
import numpy as np

# === Function to calculate the angle between three points ===
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

# === Initialize MediaPipe Pose Model ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# === Frame-by-Frame Pose Analysis ===
def analyze_pose(frame, draw_angles=False):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    feedback = ""
    debug_text = ""

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Extract normalized coords for a few key points
        r_shoulder = (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
        r_elbow = (lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y)
        r_wrist = (lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y)

        # Add them to debug text
        debug_text += f"Right Shoulder (norm): {r_shoulder}\n"
        debug_text += f"Right Elbow (norm): {r_elbow}\n"
        debug_text += f"Right Wrist (norm): {r_wrist}\n"

        # Calculate elbow angle
        elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        debug_text += f"Elbow Angle: {elbow_angle:.2f}°\n"

        # Optional: draw landmarks on frame for visualization
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
    


# === Video Pose Analysis (entire video) ===
def analyze_pose_video(video_path):
    cap = cv2.VideoCapture(video_path)

    elbow_angles = []
    spine_angles = []
    knee_angles = []
    shoulder_angles = []
    
    frame_counter = 0  # Frame counter to track the frame number

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # Only analyze every 5th frame
        if frame_counter % 5 == 0:
            # Convert to RGB and process
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                def get_point(landmark):
                    return (landmark.x, landmark.y)

                # Extract points for the necessary landmarks
                r_shoulder = get_point(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
                r_elbow = get_point(lm[mp_pose.PoseLandmark.RIGHT_ELBOW])
                r_wrist = get_point(lm[mp_pose.PoseLandmark.RIGHT_WRIST])

                r_hip = get_point(lm[mp_pose.PoseLandmark.RIGHT_HIP])
                r_knee = get_point(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
                r_ankle = get_point(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])

                l_hip = get_point(lm[mp_pose.PoseLandmark.LEFT_HIP])
                l_shoulder = get_point(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])

                # Compute angles
                elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                spine_angle = calculate_angle(l_hip, r_hip, r_shoulder)
                knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                shoulder_angle = calculate_angle(r_elbow, r_shoulder, r_hip)

                # Store angles
                elbow_angles.append(elbow_angle)
                spine_angles.append(spine_angle)
                knee_angles.append(knee_angle)
                shoulder_angles.append(shoulder_angle)

    cap.release()

    # Summarize feedback based on angles
    def summarize(name, angles, safe_range, risky_threshold, inverse=False):
        avg = sum(angles) / len(angles)
        worst = min(angles) if not inverse else max(angles)

        status = "✅ Safe"
        if (not inverse and worst < risky_threshold) or (inverse and worst > risky_threshold):
            status = f"⚠️ Risk detected ({name} = {int(worst)}°)"

        return f"{name}:\n - Avg: {int(avg)}°\n - Peak: {int(worst)}°\n - {status}\n"

    # Customize thresholds and ranges here
    feedback = ""
    feedback += summarize("Elbow angle", elbow_angles, (165, 180), 150)
    feedback += summarize("Spine lean", spine_angles, (0, 30), 40, inverse=True)
    feedback += summarize("Knee bend", knee_angles, (160, 180), 145)
    feedback += summarize("Shoulder abduction", shoulder_angles, (90, 110), 120, inverse=True)

    return feedback
