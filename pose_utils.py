import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

#For frame by frame, do not use rn
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

#For entire video summarized, using rn
def analyze_pose_video(video_path):
    cap = cv2.VideoCapture(video_path)

    elbow_angles = []
    spine_angles = []
    knee_angles = []
    shoulder_angles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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

    # === Compute summaries ===
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


