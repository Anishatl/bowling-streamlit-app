import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def analyze_pose(frame):
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

        # === 1. Elbow Angle ===
        elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        if elbow_angle < 150:
            feedback += f"⚠️ Elbow angle is too low ({int(elbow_angle)}°) — may indicate illegal action or strain.\n"

        # === 2. Spine Side Bend ===
        # Using: left_hip – right_hip – right_shoulder
        spine_angle = calculate_angle(l_hip, r_hip, r_shoulder)
        if spine_angle > 40:
            feedback += f"⚠️ Excessive spine lean detected ({int(spine_angle)}°). Can cause back strain.\n"

        # === 3. Front Knee Bend (assuming right leg is front leg) ===
        knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        if knee_angle < 145:
            feedback += f"⚠️ Front knee bend is too deep ({int(knee_angle)}°). Risk of knee injury.\n"

        # === 4. Shoulder Abduction ===
        # Approximation using: elbow – shoulder – hip
        shoulder_angle = calculate_angle(r_elbow, r_shoulder, r_hip)
        if shoulder_angle > 120:
            feedback += f"⚠️ Shoulder raised too far ({int(shoulder_angle)}°). Risk of shoulder impingement.\n"

        if feedback == "":
            feedback = "✅ No major pose issues detected."

    return frame, feedback

