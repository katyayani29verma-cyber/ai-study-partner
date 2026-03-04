import cv2
import numpy as np
import time
from collections import deque
import winsound
import threading

try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except (AttributeError, ImportError):
    from mediapipe.python.solutions import holistic as mp_holistic
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
# EYE CONSTANTS
EAR_DROP_RATIO = 0.80    
MIN_BLINK_DURATION = 0.02 
BLINK_COOLDOWN = 0.2     
WINDOW_SIZE = 60
FATIGUE_BLINK_RATE = 20   
MICROSLEEP_DURATION = 3.0 

# POSTURE CONSTANTS
SLOUCH_DROP_RATIO = 0.75

# LANDMARK INDICES
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================
def calculate_ear(landmarks, eye_indices):
    points = [(landmarks[index].x, landmarks[index].y) for index in eye_indices]
    v_dist1 = np.linalg.norm(np.array(points[1]) - np.array(points[4]))
    v_dist2 = np.linalg.norm(np.array(points[2]) - np.array(points[5]))
    h_dist = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    if h_dist == 0: return 0.0
    return (v_dist1 + v_dist2) / (2 * h_dist)

def get_head_pose(landmarks, frame_width, frame_height):
    """
    Calculates 3D head rotation (Pitch & Yaw) using solvePnP.
    """
    face_2d = []
    
    # Generic 3D model points for a human face
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)

    # Correlating MediaPipe 2D landmarks: Nose(1), Chin(152), L_Eye(226), R_Eye(446), L_Mouth(57), R_Mouth(287)
    indices = [1, 152, 226, 446, 57, 287] 
    for idx in indices:
        lm = landmarks[idx]
        x, y = int(lm.x * frame_width), int(lm.y * frame_height)
        face_2d.append([x, y])
        
    face_2d = np.array(face_2d, dtype=np.float64)
    
    # Camera matrix calculation
    focal_length = 1 * frame_width
    cam_matrix = np.array([ [focal_length, 0, frame_height / 2],
                            [0, focal_length, frame_width / 2],
                            [0, 0, 1] ])
    
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    # Solve for 3D rotation
    success, rot_vec, trans_vec = cv2.solvePnP(model_points, face_2d, cam_matrix, dist_matrix)
    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    # Extract Pitch (Up/Down) and Yaw (Left/Right)
    pitch = angles[0]
    yaw = angles[1]
    
    # --- CONTEXTUAL DISTRACTION LOGIC ---
    if yaw > 15 or yaw < -15:
        return 'LOOKING_AROUND', pitch, yaw
    elif pitch > 15:
        return 'LOOKING_UP', pitch, yaw
    elif pitch < -10:
        return 'LOOKING_DOWN', pitch, yaw
    else:
        return 'SCREEN', pitch, yaw

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def process_video():
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    
    # Trackers
    blink_count, is_blinking = 0, False
    blink_start_time, last_blink_time, last_alarm_time = 0, 0, 0
    blink_times = deque(maxlen=100) 
    
    # Distraction history now uses the 3D states
    head_history = deque(maxlen=30) 
    
    ear_history = deque(maxlen=150) 
    dynamic_ear_threshold, baseline_ear = 0.0, 0.0
    state, state_color = "CALIBRATING...", (0, 255, 255) 
    
    posture_history = deque(maxlen=150)
    baseline_posture = 0.0
    dynamic_slouch_threshold = 0.0
    posture_state, posture_color = "CALIBRATING...", (0, 255, 255)
    
    print("Starting AI Study Partner Engine (3D Pose Active)...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue
        
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        debug_frame = frame.copy()
        
        results = holistic.process(rgb_frame)
        current_ear, blink_rate, current_posture_ratio = 0.0, 0, 0.0
        pitch, yaw = 0.0, 0.0
        
        # ==========================================
        # POSTURE LOGIC
        # ==========================================
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                debug_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            l_shldr = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            r_shldr = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
            
            shoulder_width = abs(l_shldr.x - r_shldr.x)
            if shoulder_width == 0: shoulder_width = 0.01 
            
            shldr_mid_y = (l_shldr.y + r_shldr.y) / 2.0
            neck_height = shldr_mid_y - nose.y 
            
            current_posture_ratio = neck_height / shoulder_width
            
            if baseline_posture == 0.0 or current_posture_ratio > dynamic_slouch_threshold:
                posture_history.append(current_posture_ratio)
                
            if len(posture_history) > 30:
                baseline_posture = np.percentile(list(posture_history), 90) 
                dynamic_slouch_threshold = baseline_posture * SLOUCH_DROP_RATIO
                
                if current_posture_ratio < dynamic_slouch_threshold:
                    posture_state, posture_color = "SLOUCHING", (0, 0, 255)
                else:
                    posture_state, posture_color = "UPRIGHT", (0, 255, 0)
        else:
            posture_state, posture_color = "BODY HIDDEN", (0, 165, 255)

        # ==========================================
        # EYE & HEAD TRACKING LOGIC
        # ==========================================
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                debug_frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(results.face_landmarks.landmark[idx].x * frame_width)
                y = int(results.face_landmarks.landmark[idx].y * frame_height)
                cv2.circle(debug_frame, (x, y), 2, (0, 0, 255), -1)

            # EAR Math
            left_ear = calculate_ear(results.face_landmarks.landmark, LEFT_EYE)
            right_ear = calculate_ear(results.face_landmarks.landmark, RIGHT_EYE)
            current_ear = (left_ear + right_ear) / 2.0
            
            current_time = time.time()
            
            if baseline_ear == 0.0 or current_ear > dynamic_ear_threshold:
                ear_history.append(current_ear)
            
            # 3D Head Pose Math
            head_direction, pitch, yaw = get_head_pose(results.face_landmarks.landmark, frame_width, frame_height)
            head_history.append(head_direction)
            
            if len(ear_history) > 30:
                baseline_ear = np.percentile(list(ear_history), 90)
                dynamic_ear_threshold = baseline_ear * EAR_DROP_RATIO
                is_sleeping = False
                
                if current_ear < dynamic_ear_threshold:
                    if not is_blinking:
                        is_blinking, blink_start_time = True, current_time
                    else:
                        if (current_time - blink_start_time) > MICROSLEEP_DURATION:
                            is_sleeping = True
                else:
                    if is_blinking:
                        if (current_time - blink_start_time) >= MIN_BLINK_DURATION:
                            if (current_time - last_blink_time) > BLINK_COOLDOWN:
                                blink_count += 1
                                blink_times.append(current_time)
                                last_blink_time = current_time
                        is_blinking = False
                
                blink_rate = sum(1 for t in blink_times if current_time - t < WINDOW_SIZE)
                
                # Check for bad distraction states (ignoring 'LOOKING_DOWN')
                recent_head_states = list(head_history)
                is_distracted = (
                    recent_head_states.count('LOOKING_AROUND') > 20 or 
                    recent_head_states.count('LOOKING_UP') > 20
                )
                
                if is_sleeping:
                    state, state_color = "SLEEPING!", (255, 0, 255) 
                    if current_time - last_alarm_time > 1.5:
                        threading.Thread(target=lambda: winsound.Beep(2500, 1000), daemon=True).start()
                        last_alarm_time = current_time
                elif blink_rate > FATIGUE_BLINK_RATE:
                    state, state_color = "FATIGUED", (0, 0, 255)
                elif is_distracted:
                    state, state_color = "DISTRACTED (AWAY)", (0, 165, 255)
                elif recent_head_states.count('LOOKING_DOWN') > 15:
                    state, state_color = "FOCUSED (NOTEBOOK)", (255, 255, 0) # Cyan/Yellowish
                else:
                    state, state_color = "FOCUSED (SCREEN)", (0, 255, 0)
            
            # --- UI RENDERING ---
            cv2.rectangle(frame, (10, 10), (450, 200), (30, 30, 30), -1)
            
            cv2.putText(frame, f"Mind: {state}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
            cv2.putText(frame, f"BPM: {blink_rate}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, f"Posture: {posture_state}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, posture_color, 2)
            
            # Base debug stats
            cv2.putText(frame, f"EAR Base: {baseline_ear:.3f} | Cur: {current_ear:.3f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Pitch: {pitch:.1f} | Yaw: {yaw:.1f}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
        cv2.imshow('AI Study Partner - Output', frame)
        cv2.imshow('AI Study Partner - Mesh Debug', debug_frame)
        if cv2.waitKey(5) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
