import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    """Calculate angle between three points in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)

    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0  # Avoid division by zero

    cosine_angle = np.dot(ba, bc) / denom
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# For frame-by-frame live view (optional)
def analyze_pose(frame, draw_angles=False):
    height, width, _ = frame.shape
    with mp_pose.Pose() as pose:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        feedback = ""

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark

            def get_point(landmark):
                return (landmark.x * width, landmark.y * height)

            # === Points ===
            r_shoulder = get_point(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
            r_elbow = get_point(lm[mp_pose.PoseLandmark.RIGHT_ELBOW])
            r_wrist = get_point(lm[mp_pose.PoseLandmark.RIGHT_WRIST])
            r_hip = get_point(lm[mp_pose.PoseLandmark.RIGHT_HIP])
            r_knee = get_point(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
            r_ankle = get_point(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
            l_hip = get_point(lm[mp_pose.PoseLandmark.LEFT_HIP])
            l_shoulder = get_point(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])

            # === Angles ===
            elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            spine_angle = calculate_angle(l_hip, r_hip, r_shoulder)
            knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            shoulder_angle = calculate_angle(r_elbow, r_shoulder, r_hip)

            if elbow_angle < 150:
                feedback += f"⚠️ Elbow angle too low ({int(elbow_angle)}°) — may indicate illegal action or strain.\n"
            if spine_angle > 40:
                feedback += f"⚠️ Excessive spine lean ({int(spine_angle)}°). Risk of back strain.\n"
            if knee_angle < 145:
                feedback += f"⚠️ Front knee bend too deep ({int(knee_angle)}°). Risk of knee injury.\n"
            if shoulder_angle > 120:
                feedback += f"⚠️ Shoulder raised too far ({int(shoulder_angle)}°). Risk of shoulder impingement.\n"

            if draw_angles:
                cv2.putText(frame, f"Elbow: {int(elbow_angle)}°", (int(r_elbow[0]), int(r_elbow[1] - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Knee: {int(knee_angle)}°", (int(r_knee[0]), int(r_knee[1] - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f"Spine: {int(spine_angle)}°", (int(r_hip[0]), int(r_hip[1] - 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            if not feedback:
                feedback = "✅ No major pose issues detected."

    return frame, feedback

# For entire video summarized
def analyze_pose_video(video_path):
    cap = cv2.VideoCapture(video_path)

    elbow_angles = []
    spine_angles = []
    knee_angles = []
    shoulder_angles = []

    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                height, width, _ = frame.shape
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark

                    def get_point(landmark):
                        return (landmark.x * width, landmark.y * height)

                    r_shoulder = get_point(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
                    r_elbow = get_point(lm[mp_pose.PoseLandmark.RIGHT_ELBOW])
                    r_wrist = get_point(lm[mp_pose.PoseLandmark.RIGHT_WRIST])
                    r_hip = get_point(lm[mp_pose.PoseLandmark.RIGHT_HIP])
                    r_knee = get_point(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
                    r_ankle = get_point(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
                    l_hip = get_point(lm[mp_pose.PoseLandmark.LEFT_HIP])

                    elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                    spine_angle = calculate_angle(l_hip, r_hip, r_shoulder)
                    knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                    shoulder_angle = calculate_angle(r_elbow, r_shoulder, r_hip)

                    elbow_angles.append(elbow_angle)
                    spine_angles.append(spine_angle)
                    knee_angles.append(knee_angle)
                    shoulder_angles.append(shoulder_angle)

            except Exception as e:
                print("Error during pose processing:", e)
                continue

    cap.release()

    # === Compute summaries ===
    def summarize(name, angles, safe_range, risky_threshold, inverse=False):
        if not angles:
            return f"{name}:\n - No data\n"
        avg = sum(angles) / len(angles)
        worst = min(angles) if not inverse else max(angles)

        status = "✅ Safe"
        if (not inverse and worst < risky_threshold) or (inverse and worst > risky_threshold):
            status = f"⚠️ Risk detected ({name} = {int(worst)}°)"

        return f"{name}:\n - Avg: {int(avg)}°\n - Peak: {int(worst)}°\n - {status}\n"

    feedback = ""
    feedback += summarize("Elbow angle", elbow_angles, (165, 180), 150)
    feedback += summarize("Spine lean", spine_angles, (0, 30), 40, inverse=True)
    feedback += summarize("Knee bend", knee_angles, (160, 180), 145)
    feedback += summarize("Shoulder abduction", shoulder_angles, (90, 110), 120, inverse=True)

    return feedback
