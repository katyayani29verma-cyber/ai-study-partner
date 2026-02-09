import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    v1 = np.linalg.norm(eye[1] - eye[5])
    v2 = np.linalg.norm(eye[2] - eye[4])
    h = np.linalg.norm(eye[0] - eye[3])
    return (v1 + v2) / (2.0 * h)

cap = cv2.VideoCapture(0)

blink_count = 0
ear_history = []
start_time = time.time()

# values to display
blink_rate = 0
posture = "Upright"
cognitive_load = 0
status = "LOW"
recommendation = "You are focused"

print("\n📷 Webcam initialized")
print("🔒 Privacy-first edge processing enabled\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            landmarks = np.array([
                [lm.x * frame.shape[1], lm.y * frame.shape[0]]
                for lm in face.landmark
            ])

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

        cognitive_load = min(
            100,
            int((blink_rate * 1.5) + (20 if posture == "Slouched" else 0))
        )

        if cognitive_load > 70:
            status = "HIGH"
            recommendation = "⚠️ Take a short break"
        elif cognitive_load > 40:
            status = "MEDIUM"
            recommendation = "💡 Slow down pace"
        else:
            status = "LOW"
            recommendation = "✅ You are focused"

        # terminal output
        print(f"👀 Blink Rate: {blink_rate} blinks/min")
        print(f"🧍 Posture: {posture}")
        print(f"🧠 Cognitive Load Score: {cognitive_load}% ({status})")
        print(f"{recommendation}\n")

        blink_count = 0
        ear_history = []
        start_time = time.time()

    # ---- VIDEO OVERLAY ----
    y = 30
    dy = 35

    cv2.putText(frame, f"Blink Rate: {blink_rate} blinks/min",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, f"Posture: {posture}",
                (10, y+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.putText(frame, f"Cognitive Load: {cognitive_load}%  ({status})",
                (10, y+2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.putText(frame, recommendation,
                (10, y+3*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

    cv2.imshow("Cognitive Load Tracker | Press Q to exit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
