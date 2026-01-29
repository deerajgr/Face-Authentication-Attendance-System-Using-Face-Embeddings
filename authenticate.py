import cv2
import face_recognition
import numpy as np
import pickle
import os
import datetime
import time

# ---------- Setup storage ----------
DATA_DIR = "data"
KNOWN_ENCODINGS_PATH = os.path.join(DATA_DIR, "known_encodings.pkl")
ATTENDANCE_LOG_PATH = "attendance_log.txt"

# Similarity threshold for face match (lower = stricter)
TOLERANCE = 0.6

def load_known_encodings():
    if os.path.exists(KNOWN_ENCODINGS_PATH):
        with open(KNOWN_ENCODINGS_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def log_attendance(user_id, action):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - User: {user_id} - Action: {action}\n"
    with open(ATTENDANCE_LOG_PATH, "a") as f:
        f.write(log_entry)
    print(f"[LOG] {log_entry.strip()}")

def preprocess_frame(frame):
    """
    Preprocess for lighting robustness and format compatibility
    """
    # Convert to grayscale for equalization, then back to color
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    
    # Remove alpha if present
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    # Ensure 3-channel
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # To RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Cast to uint8
    rgb = np.array(rgb, dtype=np.uint8)
    
    return rgb, frame  # Return both RGB (for recognition) and BGR (for display)

def detect_blink(face_landmarks):
    """
    Basic liveness check: Compute Eye Aspect Ratio (EAR) for blink detection
    """
    if not face_landmarks:
        return False
    
    def eye_aspect_ratio(eye):
        # Vertical distances
        v1 = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        v2 = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        # Horizontal distance
        h = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        return (v1 + v2) / (2.0 * h)
    
    left_eye = face_landmarks[0]['left_eye']
    right_eye = face_landmarks[0]['right_eye']
    
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    
    return avg_ear < 0.3  # Blink if EAR below threshold

# ---------- Main Authentication ----------
def authenticate_and_mark_attendance(action="Punch-In"):
    known_encodings = load_known_encodings()
    if not known_encodings:
        print("[ERROR] No registered users found.")
        return
    
    known_ids = list(known_encodings.keys())
    known_faces = list(known_encodings.values())
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return
    
    print(f"\n[INFO] Starting face authentication for {action}. Look at the camera.\n")
    print("Detecting face... (Blink to pass liveness check)\n")
    
    blink_detected = False
    blink_frames = 0
    required_blinks = 3  # Consecutive frames with low EAR
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break
        
        rgb_frame, display_frame = preprocess_frame(frame)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
        
        identified = False
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=TOLERANCE)
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                user_id = known_ids[best_match_index]
                
                # Liveness check
                if detect_blink(face_landmarks):
                    blink_frames += 1
                    if blink_frames >= required_blinks:
                        blink_detected = True
                else:
                    blink_frames = 0
                
                if blink_detected:
                    print(f"[SUCCESS] Identified {user_id}. Marking {action}.")
                    log_attendance(user_id, action)
                    identified = True
                    
                    # Draw rectangle and label
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{user_id} - Authenticated", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Blink to verify", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(display_frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow("Authenticate Face", display_frame)
        
        if identified:
            time.sleep(2)  # Show success for 2 seconds
            break
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ---------- Run ----------
if __name__ == "__main__":
    action = input("Enter action (Punch-In or Punch-Out): ").strip().title()
    if action not in ["Punch-In", "Punch-Out"]:
        print("[ERROR] Invalid action. Use 'Punch-In' or 'Punch-Out'.")
    else:
        authenticate_and_mark_attendance(action)