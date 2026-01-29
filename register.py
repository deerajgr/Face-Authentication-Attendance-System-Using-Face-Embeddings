import cv2
import face_recognition
import numpy as np
import pickle
import os

# ---------- Setup storage ----------
DATA_DIR = "data"
KNOWN_ENCODINGS_PATH = os.path.join(DATA_DIR, "known_encodings.pkl")

os.makedirs(DATA_DIR, exist_ok=True)


def load_known_encodings():
    if os.path.exists(KNOWN_ENCODINGS_PATH):
        with open(KNOWN_ENCODINGS_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_known_encodings(encodings):
    with open(KNOWN_ENCODINGS_PATH, "wb") as f:
        pickle.dump(encodings, f)


def preprocess_frame(frame):
    """
    Force frame into strict RGB uint8 format for face_recognition
    """

    # Remove alpha channel if present (RGBA → RGB)
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Convert to 3-channel BGR if grayscale
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # HARD CAST to uint8 (no conditions)
    rgb = np.array(rgb, dtype=np.uint8)

    return rgb


# ---------- Main Registration ----------
def register_user(user_id):
    known_encodings = load_known_encodings()

    if user_id in known_encodings:
        print(f"[INFO] User '{user_id}' already registered.")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print(f"\n[INFO] Registering '{user_id}'")
    print("Look at the camera.")
    print("Press 's' to capture (need 5 images). Press 'q' to quit.\n")

    encodings = []
    captured = 0

    while captured < 5:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            rgb_frame = preprocess_frame(frame)
            if rgb_frame is None:
                print("[WARN] Unsupported frame format. Skipping.")
                continue

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if face_encodings:
                encodings.append(face_encodings[0])
                captured += 1
                print(f"[OK] Captured {captured}/5")
            else:
                print("[WARN] No face detected. Try again.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if encodings:
        avg_encoding = np.mean(encodings, axis=0)
        known_encodings[user_id] = avg_encoding
        save_known_encodings(known_encodings)
        print(f"\n[SUCCESS] User '{user_id}' registered successfully.\n")
    else:
        print("\n[FAILED] No valid face encodings captured.\n")


# ---------- Run ----------
if __name__ == "__main__":
    user_id = input("Enter User ID (your name): ").strip()
    register_user(user_id)
