import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye_points):
    vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

cap = cv2.VideoCapture(0)
blink_count = 0
start_time = time.time()
ear_history = []

print("📷 Webcam initialized")
print("🔒 Processing locally (privacy-first edge AI)\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array(
                [[lm.x * frame.shape[1], lm.y * frame.shape[0]]
                 for lm in face_landmarks.landmark]
            )

            left_ear = eye_aspect_ratio(landmarks[LEFT_EYE])
            right_ear = eye_aspect_ratio(landmarks[RIGHT_EYE])
            ear = (left_ear + right_ear) / 2
            ear_history.append(ear)

            if ear < 0.23:
                blink_count += 1

    elapsed = time.time() - start_time
    if elapsed > 15:
        blink_rate = blink_count * 4
        posture = "Slouched" if np.mean(ear_history) < 0.25 else "Upright"

        cognitive_load = min(100, int((blink_rate * 1.5) + (20 if posture == "Slouched" else 0)))

        status = "LOW"
        if cognitive_load > 70:
            status = "HIGH"
        elif cognitive_load > 40:
            status = "MEDIUM"

        print(f"👀 Blink Rate: {blink_rate} blinks/min")
        print(f"🧍 Posture: {posture}")
        print(f"🧠 Cognitive Load Score: {cognitive_load}% ({status})")

        if status == "HIGH":
            print("⚠️ Recommendation: Take a short break\n")
        elif status == "MEDIUM":
            print("💡 Recommendation: Slow down pace\n")
        else:
            print("✅ You are focused\n")

        blink_count = 0
        ear_history = []
        start_time = time.time()

    cv2.imshow("Cognitive Load Tracker (Press Q to exit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
