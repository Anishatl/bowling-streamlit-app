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


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


def analyze_pose(frame, draw_angles=False):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    feedback = ""

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        lm = results.pose_landmarks.landmark

        # Convert landmarks to pixel coordinates
        def get_point(landmark):
            return (landmark.x * frame.shape[1], landmark.y * frame.shape[0])

        # === Joint Points ===
        r_shoulder = get_point(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
        r_elbow = get_point(lm[mp_pose.PoseLandmark.RIGHT_ELBOW])
        r_wrist = get_point(lm[mp_pose.PoseLandmark.RIGHT_WRIST])

        r_hip = get_point(lm[mp_pose.PoseLandmark.RIGHT_HIP])
        r_knee = get_point(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
        r_ankle = get_point(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])

        l_shoulder = get_point(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
        l_hip = get_point(lm[mp_pose.PoseLandmark.LEFT_HIP])

        # === 1. Elbow Angle ===
        elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        if elbow_angle < 150:
            feedback += f"⚠️ Elbow angle is too low ({int(elbow_angle)}°) — may indicate illegal action or strain.\n"

        # === 2. Spine Side Bend ===
        spine_angle = calculate_angle(l_hip, r_hip, r_shoulder)
        if spine_angle > 40:
            feedback += f"⚠️ Excessive spine lean detected ({int(spine_angle)}°). Can cause back strain.\n"

        # === 3. Front Knee Bend ===
        knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        if knee_angle < 145:
            feedback += f"⚠️ Front knee bend is too deep ({int(knee_angle)}°). Risk of knee injury.\n"

        # === 4. Shoulder Abduction ===
        shoulder_angle = calculate_angle(r_elbow, r_shoulder, r_hip)
        if shoulder_angle > 120:
            feedback += f"⚠️ Shoulder raised too far ({int(shoulder_angle)}°). Risk of shoulder impingement.\n"

        if feedback == "":
            feedback = "✅ No major pose issues detected."

        # Optionally draw the angles
        if draw_angles:
            # Draw the angles on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Elbow Angle: {int(elbow_angle)}°", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Spine Angle: {int(spine_angle)}°", (50, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Knee Angle: {int(knee_angle)}°", (50, 150), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Shoulder Angle: {int(shoulder_angle)}°", (50, 200), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return frame, feedback

