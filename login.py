import cv2
import numpy as np
import os
import time
from collections import deque, Counter
from database import load_label_map, log_audit, load_database

GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN   = (255, 255, 0)
WHITE  = (255, 255, 255)
ORANGE = (0, 165, 255)


def get_confidence_threshold(label_to_name, recognizer, all_faces, all_names):
    """
    Dynamically compute a per-user confidence threshold based on
    how well the model recognizes each registered user's own samples.
    Returns a global threshold that works for ALL users.
    """
    if not all_faces or not all_names:
        return 100  

    name_to_label = {v: k for k, v in label_to_name.items()}
    user_confidences = {}

    for name, label in name_to_label.items():
        user_faces = [f for f, n in zip(all_faces, all_names) if n == name]
        if not user_faces:
            continue
        confs = []
        for face in user_faces[:10]: 
            try:
                pred_label, conf = recognizer.predict(face)
                if pred_label == label:
                    confs.append(conf)
            except:
                pass
        if confs:
            avg = sum(confs) / len(confs)
            user_confidences[name] = avg
            print(f"[THRESHOLD] {name} avg self-confidence: {avg:.1f}")

    if not user_confidences:
        return 100

    max_conf = max(user_confidences.values())
    # Training samples can return near-0 confidence, which makes threshold 0
    # and causes every real login frame to be rejected.
    if max_conf <= 1:
        threshold = 70
    else:
        threshold = min(max_conf * 1.5, 130)  # cap at 130
        threshold = max(threshold, 45)
    print(f"[THRESHOLD] Auto threshold set to: {threshold:.1f}")
    return threshold


def login():
    print("\n" + "="*60)
    print("     FACE LOGIN SYSTEM - MULTI USER")
    print("="*60)

    if not os.path.exists("face_model.yml"):
        print("ERROR: No face model found. Please register users first.")
        input("Press Enter...")
        return False

    if not os.path.exists("name_map.pkl"):
        print("ERROR: No registered users. Please register first.")
        input("Press Enter...")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_model.yml")

    label_to_name = load_label_map()
    if not label_to_name:
        print("ERROR: Label map is empty. Please register users first.")
        input("Press Enter...")
        return False

    print(f"\nRegistered users : {list(label_to_name.values())}")
    print(f"Label map        : {label_to_name}")

    
    all_names, all_faces, _ = load_database()

    
    CONFIDENCE_THRESHOLD = get_confidence_threshold(
        label_to_name, recognizer, all_faces, all_names
    )
    print(f"[INFO] Using confidence threshold: {CONFIDENCE_THRESHOLD:.1f}")


    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        input("Press Enter...")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print("\n" + "="*40)
    print("• Look directly at camera")
    print("• Hold still for 2-3 seconds")
    print("• Unknown faces will be REJECTED")
    print("• Press Q to quit")
    print("="*40 + "\n")

    
    REQUIRED_MATCHES   = 12
    HISTORY_SIZE       = 15
    CONSISTENCY_NEEDED = 60   
    MIN_FACE_SIZE      = 100

    match_history = deque(maxlen=HISTORY_SIZE)
    label_history = deque(maxlen=HISTORY_SIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()
        h_frame, w_frame = frame.shape[:2]

    
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )

    
        cv2.putText(display, "FACE LOGIN SYSTEM", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, CYAN, 2)

        
        cv2.putText(display, "Look at camera and hold still", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

        if len(faces) == 0:
            cv2.putText(display, "NO FACE DETECTED — Move closer", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
            match_history.clear()
            label_history.clear()

        else:
            
            largest = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest

            face_area  = w * h
            frame_area = w_frame * h_frame
            face_ratio = face_area / frame_area

            if face_ratio < 0.03:
                cv2.putText(display, "Move closer to camera", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2)
                match_history.clear()
                label_history.clear()
            else:
                cv2.rectangle(display, (x, y), (x+w, y+h), GREEN, 2)

                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))

                label, confidence = recognizer.predict(face_roi)

            
                predicted_name = label_to_name.get(label, "UNKNOWN")
                print(f"[DEBUG] Label:{label} | Conf:{confidence:.1f} | "
                      f"Name:{predicted_name} | Threshold:{CONFIDENCE_THRESHOLD:.1f}")

                match_history.append(confidence)
                label_history.append(label)

                conf_color = GREEN if confidence < CONFIDENCE_THRESHOLD else RED
                cv2.putText(display, f"Conf: {confidence:.1f} / {CONFIDENCE_THRESHOLD:.0f}",
                            (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
                cv2.putText(display, "Scanning face...", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1)

                if len(match_history) >= REQUIRED_MATCHES:
                    avg_confidence = sum(match_history) / len(match_history)
                    good_frames    = sum(1 for c in match_history
                                        if c < CONFIDENCE_THRESHOLD)

                    label_counts             = Counter(label_history)
                    best_label, best_count   = label_counts.most_common(1)[0]
                    consistency              = (best_count / len(label_history)) * 100
                    best_name                = label_to_name.get(best_label, "UNKNOWN")

                    cv2.putText(display, f"Avg conf : {avg_confidence:.1f}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
                    cv2.putText(display, f"Consistency: {consistency:.0f}%", (10, 172),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
                    cv2.putText(display, f"Good frames: {good_frames}/{REQUIRED_MATCHES}",
                                (10, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
                    cv2.putText(display, "Best match : checking...", (10, 216),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1)

                    if (avg_confidence < CONFIDENCE_THRESHOLD and
                            best_label in label_to_name and
                            consistency >= CONSISTENCY_NEEDED):

                        matched_name = label_to_name[best_label]

                        cv2.putText(display, f"MATCH: {matched_name}!", (10, 260),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2)
                        cv2.imshow("Face Login", display)
                        cv2.waitKey(800)

                        print(f"\n{'='*40}")
                        print(f"  LOGIN SUCCESSFUL!")
                        print(f"  User       : {matched_name}")
                        print(f"  Label      : {best_label}")
                        print(f"  Confidence : {avg_confidence:.1f}")
                        print(f"  Consistency: {consistency:.1f}%")
                        print(f"{'='*40}\n")

                        log_audit(matched_name, "LOGIN", True,
                                  f"conf_{avg_confidence:.1f}")

                        success = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(success, "LOGIN SUCCESSFUL!", (80, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, GREEN, 3)
                        cv2.putText(success, f"Welcome, {matched_name}!",
                                    (140, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2)
                        cv2.putText(success, f"Label ID: {best_label}",
                                    (220, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 1)
                        cv2.imshow("Face Login", success)
                        cv2.waitKey(3000)

                        cap.release()
                        cv2.destroyAllWindows()
                        return True

                    else:
                        reasons = []
                        if avg_confidence >= CONFIDENCE_THRESHOLD:
                            reasons.append(
                                f"conf {avg_confidence:.1f} > {CONFIDENCE_THRESHOLD:.0f}"
                            )
                        if best_label not in label_to_name:
                            reasons.append("not registered")
                        if consistency < CONSISTENCY_NEEDED:
                            reasons.append(f"inconsistent {consistency:.0f}%")

                        cv2.putText(display,
                                    f"REJECTED: {', '.join(reasons)}",
                                    (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1)

                        log_audit("unknown", "LOGIN", False, ', '.join(reasons))

                        if len(match_history) >= HISTORY_SIZE:
                            cv2.imshow("Face Login", display)
                            cv2.waitKey(500)

                            denied = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(denied, "ACCESS DENIED", (100, 180),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, RED, 3)
                            cv2.putText(denied, "Not Recognised in System",
                                        (80, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
                            cv2.putText(denied, f"Confidence: {avg_confidence:.1f}",
                                        (200, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 1)
                            cv2.imshow("Face Login", denied)
                            cv2.waitKey(2500)

                            match_history.clear()
                            label_history.clear()

                else:
                    remaining = REQUIRED_MATCHES - len(match_history)
                    cv2.putText(display,
                                f"Scanning: {len(match_history)}/{REQUIRED_MATCHES} frames",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, YELLOW, 2)

        bar_w = int((len(match_history) / REQUIRED_MATCHES) * 300)
        cv2.rectangle(display, (10, 420), (310, 440), (50, 50, 50), -1)
        cv2.rectangle(display,
                      (10, 420),
                      (10 + min(bar_w, 300), 440),
                      GREEN if len(match_history) >= REQUIRED_MATCHES else YELLOW,
                      -1)
        cv2.putText(display, f"{len(match_history)}/{REQUIRED_MATCHES}",
                    (320, 436), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        cv2.imshow("Face Login", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False


if __name__ == "__main__":
    login()
