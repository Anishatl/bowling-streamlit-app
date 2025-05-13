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
    """Analyze the pose on a single frame."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    feedback = ""

    if results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        lm = results.pose_landmarks.landmark

        # Convert landmarks to pixel coordinates
        def get_point(landmark):
            return (landmark.x, landmark.y)

        # === Joint Points ===
        r_shoulder = get_point(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
        r_elbow = get_point(lm[mp_pose.PoseLandmark.RIGHT_ELBOW])
        r_wrist = get_point(lm[mp_pose.PoseLandmark.RIGHT_WRIST])

        r_hip = get_point(lm[mp_pose.PoseLandmark.RIGHT_HIP])
        r_knee = get_point(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
        r_ankle = get_point(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])

        l_shoulder = get_point(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
        l_hip = get_point(lm[mp_pose.PoseLandmark.LEFT_HIP])

        # === Elbow Angle ===
        elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        if elbow_angle < 150:
            feedback += f"⚠️ Elbow angle is too low ({int(elbow_angle)}°) — may indicate illegal action or strain.\n"

        # === Spine Side Bend ===
        spine_angle = calculate_angle(l_hip, r_hip, r_shoulder)
        if spine_angle > 40:
            feedback += f"⚠️ Excessive spine lean detected ({int(spine_angle)}°). Can cause back strain.\n"

        # === Front Knee Bend ===
        knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        if knee_angle < 145:
            feedback += f"⚠️ Front knee bend is too deep ({int(knee_angle)}°). Risk of knee injury.\n"

        # === Shoulder Abduction ===
        shoulder_angle = calculate_angle(r_elbow, r_shoulder, r_hip)
        if shoulder_angle > 120:
            feedback += f"⚠️ Shoulder raised too far ({int(shoulder_angle)}°). Risk of shoulder impingement.\n"

        if feedback == "":
            feedback = "✅ No major pose issues detected."

        # === Draw angles if requested ===
        if draw_angles:
            cv2.putText(frame, f"Elbow Angle: {int(elbow_angle)}°", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Spine Angle: {int(spine_angle)}°", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Knee Angle: {int(knee_angle)}°", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Shoulder Angle: {int(shoulder_angle)}°", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame, feedback

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
