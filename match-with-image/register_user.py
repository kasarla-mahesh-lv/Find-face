import cv2
import numpy as np
import os
import time
import mediapipe as mp
from database import save_database, load_database, load_label_map, save_label_map

GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN   = (255, 255, 0)
WHITE  = (255, 255, 255)
ORANGE = (0, 165, 255)

mp_face_mesh = mp.solutions.face_mesh


def calculate_head_pose(landmarks, w, h):
    if not landmarks:
        return "UNKNOWN", 0
    nose_tip     = landmarks[1]
    left_eye     = landmarks[33]
    right_eye    = landmarks[263]
    eye_center_x = (left_eye.x + right_eye.x) / 2
    turn_amount  = (nose_tip.x - eye_center_x) * 4
    if turn_amount < -0.25:
        return "LEFT", turn_amount
    elif turn_amount > 0.25:
        return "RIGHT", turn_amount
    else:
        return "FRONT", turn_amount


def check_face_quality(face_roi):
    if face_roi.size == 0:
        return False, 0
    laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
    brightness    = np.mean(face_roi)
    quality_score = 0
    if laplacian_var > 30:    quality_score += 40
    elif laplacian_var > 20:  quality_score += 30
    elif laplacian_var > 10:  quality_score += 20
    else:                     quality_score += 10
    if 40 < brightness < 220:    quality_score += 30
    elif 20 < brightness < 240:  quality_score += 20
    else:                        quality_score += 10
    return quality_score >= 40, quality_score


def register_user():
    print("\n" + "="*70)
    print("     FACE REGISTRATION SYSTEM - MULTI-ANGLE")
    print("="*70)
    print("     • FRONT  - Look straight at camera")
    print("     • LEFT   - Turn head slowly left")
    print("     • RIGHT  - Turn head slowly right")
    print("     • Only 20 photos needed (fast!)")
    print("="*70)

    name        = input("Enter your name to register: ").strip().lower()
    employee_id = input("Enter Employee ID: ").strip().upper()

    if not name or not employee_id:
        print("Name and Employee ID cannot be empty!")
        return

    existing_names, existing_faces, existing_empids = [], [], []
    if os.path.exists("faces_data.pkl"):
        try:
            existing_names, existing_faces, existing_empids = load_database()
            print(f"[INFO] Existing users: {list(set(existing_names))}")
        except Exception as e:
            print(f"[WARNING] Could not load existing data: {e}")

    label_to_name = load_label_map()
    name_to_label = {v: k for k, v in label_to_name.items()}
    print(f"[INFO] Current label map: {label_to_name}")

    if name in name_to_label:
        print(f"\n[WARNING] '{name}' is already registered!")
        choice = input("Re-register? (y/n): ").strip().lower()
        if choice != 'y':
            return
        combined = [
            (n, f, e)
            for n, f, e in zip(existing_names, existing_faces, existing_empids)
            if n != name
        ]
        if combined:
            existing_names, existing_faces, existing_empids = map(list, zip(*combined))
        else:
            existing_names, existing_faces, existing_empids = [], [], []
        print(f"[INFO] Removed old face data for '{name}'")

    if name in name_to_label:
        this_label = name_to_label[name]   
    else:
        this_label = (max(label_to_name.keys()) + 1) if label_to_name else 0
    print(f"[INFO] Label for '{name}': {this_label}")

    print("\nChecking camera...")
    cap = None
    for idx in [0, 1]:
        test_cap = cv2.VideoCapture(idx)
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret and frame is not None:
                print(f"Camera {idx} connected!")
                cap = test_cap
                break
            test_cap.release()

    if cap is None:
        print("Cannot open any camera!")
        input("Press Enter to continue...")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    angles = {
        "FRONT": {"count": 0, "target": 7,  "color": GREEN,  "instruction": "Look STRAIGHT at camera"},
        "LEFT":  {"count": 0, "target": 7,  "color": CYAN,   "instruction": "Turn head SLOWLY LEFT"},
        "RIGHT": {"count": 0, "target": 6,  "color": ORANGE, "instruction": "Turn head SLOWLY RIGHT"}
    }

    new_names, new_faces, new_empids = [], [], []
    total_captured = 0
    TOTAL_TARGET   = 20        
    last_capture   = 0
    capture_delay  = 0.4     

    print(f"\nCapturing {TOTAL_TARGET} photos for '{name}'...")
    print("Press 'q' to cancel.\n")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        while total_captured < TOTAL_TARGET:
            ret, frame = cap.read()
            if not ret:
                break

            frame        = cv2.flip(frame, 1)
            rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results      = face_mesh.process(rgb)
            gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w         = frame.shape[:2]
            current_time = time.time()

            cv2.putText(frame, f"Registering: {name}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
            cv2.putText(frame, f"Total: {total_captured}/{TOTAL_TARGET}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

            if results.multi_face_landmarks:
                landmarks        = results.multi_face_landmarks[0].landmark
                angle, turn_val  = calculate_head_pose(landmarks, w, h)

                xs = [int(lm.x * w) for lm in landmarks]
                ys = [int(lm.y * h) for lm in landmarks]
                x  = max(0, min(xs) - 20)
                y  = max(0, min(ys) - 20)
                x2 = min(w, max(xs) + 20)
                y2 = min(h, max(ys) + 20)

                face_roi         = gray[y:y2, x:x2]
                is_good, quality = check_face_quality(face_roi)

                
                face_w = x2 - x
                face_h = y2 - y
                if face_w < 80 or face_h < 80:
                    cv2.putText(frame, "Move closer to camera",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
                    cv2.imshow("Multi-Angle Face Registration", frame)
                    cv2.waitKey(1)
                    continue

                angle_color = angles[angle]["color"] if angle in angles else WHITE
                cv2.putText(frame, f"Angle: {angle} ({turn_val:.2f})",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, angle_color, 2)
                cv2.rectangle(frame, (x, y), (x2, y2), angle_color, 2)

                if angle in angles:
                    ad = angles[angle]
                    cv2.putText(frame, f"{angle}: {ad['count']}/{ad['target']}",
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, angle_color, 2)
                    cv2.putText(frame, ad['instruction'],
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, angle_color, 1)

                    if (is_good and
                        ad['count'] < ad['target'] and
                        (current_time - last_capture) > capture_delay):

                        face_resized = cv2.resize(face_roi, (200, 200))
                        new_names.append(name)
                        new_faces.append(face_resized)
                        new_empids.append(employee_id)

                        ad['count']    += 1
                        total_captured += 1
                        last_capture    = current_time

                        print(f"  Captured {angle}: {ad['count']}/{ad['target']} "
                              f"(Total: {total_captured}/{TOTAL_TARGET})")

                        overlay = frame.copy()
                        cv2.rectangle(overlay, (x, y), (x2, y2), angle_color, -1)
                        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            else:
                cv2.putText(frame, "No face — Look at camera",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

            
            for i, (aname, ad) in enumerate(angles.items()):
                xp  = 50 + i * 180
                bar = int((ad['count'] / ad['target']) * 150)
                cv2.rectangle(frame, (xp, 390), (xp+150, 408), (50,50,50), -1)
                cv2.rectangle(frame, (xp, 390), (xp+bar,  408), ad['color'], -1)
                cv2.putText(frame, f"{aname} {ad['count']}/{ad['target']}",
                            (xp+5, 386), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ad['color'], 1)

    
            bar_total = int((total_captured / TOTAL_TARGET) * 500)
            cv2.rectangle(frame, (50, 440), (550, 465), (100,100,100), -1)
            cv2.rectangle(frame, (50, 440), (50+bar_total, 465), CYAN, -1)
            cv2.putText(frame, f"TOTAL: {total_captured}/{TOTAL_TARGET}",
                        (230, 436), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

            cv2.imshow("Multi-Angle Face Registration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nRegistration cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

    if total_captured < TOTAL_TARGET:
        print(f"[ERROR] Only captured {total_captured}. Need {TOTAL_TARGET}.")
        return

    all_names  = list(existing_names)  + new_names
    all_faces  = list(existing_faces)  + new_faces
    all_empids = list(existing_empids) + new_empids

    label_to_name[this_label] = name
    name_to_label = {v: k for k, v in label_to_name.items()}
    print(f"\n[INFO] Final label map: {label_to_name}")

    training_faces, training_labels = [], []
    for n, f in zip(all_names, all_faces):
        if n in name_to_label:
            training_faces.append(f)
            training_labels.append(name_to_label[n])

    if not training_faces:
        print("[ERROR] No training data!")
        return

    print(f"[INFO] Training with {len(training_faces)} samples...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(training_faces, np.array(training_labels, dtype=np.int32))
    recognizer.save("face_model.yml")

    save_database(all_names, all_faces, all_empids)
    save_label_map(label_to_name)

    print(f"\n{'='*70}")
    print(f"  REGISTRATION SUCCESSFUL!")
    print(f"  Name     : {name}")
    print(f"  Emp ID   : {employee_id}")
    print(f"  Label    : {this_label}")
    print(f"  Samples  : {total_captured}")
    print(f"  All Users: {list(label_to_name.values())}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    register_user()