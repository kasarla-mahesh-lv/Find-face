import cv2
import numpy as np
import os
import time
from database import load_db, save_db

os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

try:
    cv2.setLogLevel(3) 
except Exception:
    pass

GREEN = (0, 255, 0)
RED   = (0, 0, 255)
CYAN  = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

POSE_PLAN = [
    ("FRONT", 6),
    ("LEFT", 5),
    ("RIGHT", 5),
    ("UP", 4),
    ("DOWN", 4),
]


def _open_camera_probe(idx, backend=None):
    cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        return None, -1.0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    score = -1.0
    valid_n = 0
    for _ in range(12):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m = float(np.mean(gray))
        s = float(np.std(gray))
        valid_n += 1
        score = max(score, m + 1.2 * s)

    if valid_n < 2:
        cap.release()
        return None, -1.0
    return cap, score


def open_registration_camera():
    
    backends = [None]
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)

    best_cap = None
    best_score = -1.0
    best_idx = None
    
    idx_order = [1, 2, 3, 0]

    for idx in idx_order:
        for backend in backends:
            cap, score = _open_camera_probe(idx, backend)
            if cap is None:
                continue
            if score > best_score:
                if best_cap is not None:
                    best_cap.release()
                best_cap = cap
                best_score = score
                best_idx = idx
            else:
                cap.release()
    return best_cap, best_idx


def open_external_webcam():
    backends = [None]
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)

    best_cap = None
    best_score = -1.0
    best_idx = None
    
    idx_order = [1, 2, 3, 4, 5]

    for idx in idx_order:
        for backend in backends:
            cap, score = _open_camera_probe(idx, backend)
            if cap is None:
                continue
            if score > best_score:
                if best_cap is not None:
                    best_cap.release()
                best_cap = cap
                best_score = score
                best_idx = idx
            else:
                cap.release()
    return best_cap, best_idx


def test_webcam():
    print("\n==================================================")
    print("  TEST WEBCAM (PREVIEW ONLY)")
    print("==================================================")
    cap, idx = open_external_webcam()
    if cap is None:
        print("  No webcam")
        return

    print(f"  Webcam opened on camera index: {idx}")
    print("  Preview only. No face recognition in this mode.")
    print("  Press Q to close preview.\n")

    fail_n = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            fail_n += 1
            if fail_n > 20:
                break
            continue
        fail_n = 0
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 36), BLACK, -1)
        cv2.putText(frame, f"Webcam preview only (index {idx})  Q=quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
        cv2.imshow("Webcam Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def deepface_ready():
    try:
        from deepface import DeepFace
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        DeepFace.represent(
            img_path=dummy,
            model_name="Facenet512",
            enforce_detection=False,
            detector_backend="skip"
        )
        return True
    except Exception as e:
        print("\n  ERROR: DeepFace/TensorFlow is not ready.")
        print(f"  Details: {e}")
        print("  Registration cannot save user until this is fixed.\n")
        return False


def get_emb(img_bgr):
    try:
        from deepface import DeepFace
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        r   = DeepFace.represent(img_path=rgb, model_name="Facenet512",
                                 enforce_detection=False,
                                 detector_backend="skip")
        if r:
            return np.array(r[0]["embedding"])
    except Exception:
        return None
    return None


def register():
    print("\n==================================================")
    print("  REGISTER NEW PERSON")
    print("==================================================")
    name = input("\n  Enter name: ").strip().lower()
    if not name:
        print("  Name cannot be empty!")
        return

    db = load_db()
    if name in db:
        print(f"  '{name}' already exists!")
        if input("  Re-register? (y/n): ").strip().lower() != 'y':
            return
        del db[name]

    
    if not deepface_ready():
        return

    cap, cam_idx = open_registration_camera()
    if cap is None:
        print("  Cannot open camera!")
        print("  Close Camera/Zoom/Meet and retry.")
        return
    print(f"  Camera selected: index {cam_idx}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    total_target = sum(c for _, c in POSE_PLAN)
    print(f"\n  Capturing {total_target} photos with pose guidance")
    print("  Press Q to cancel\n")

    images = []
    TARGET = total_target
    last_t = 0

    while len(images) < TARGET:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        now   = time.time()

        cv2.rectangle(frame, (0, 0), (w, 78), BLACK, -1)
        cv2.putText(frame, f"Registering: {name}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
        cv2.putText(frame, f"Captured: {len(images)}/{TARGET}",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

        done = len(images)
        running = 0
        current_pose = "FRONT"
        pose_done = 0
        pose_need = POSE_PLAN[0][1]
        for pose_name, pose_count in POSE_PLAN:
            if done < running + pose_count:
                current_pose = pose_name
                pose_done = done - running
                pose_need = pose_count
                break
            running += pose_count

        cv2.putText(frame, f"Pose: {current_pose} ({pose_done}/{pose_need})",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, CYAN, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw  = haar.detectMultiScale(
            cv2.equalizeHist(gray),
            scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        if len(raw) == 0:
            cv2.putText(frame, "No face - look at camera",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, RED, 2)
        else:
            x, y, fw, fh = max(raw, key=lambda f: f[2] * f[3])
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), GREEN, 2)
            crop = frame[y:y + fh, x:x + fw]
            lap  = cv2.Laplacian(gray[y:y + fh, x:x + fw], cv2.CV_64F).var()
            bri  = np.mean(gray[y:y + fh, x:x + fw])
            if lap > 8 and 20 < bri < 240 and now - last_t > 0.4:
                images.append(cv2.resize(crop, (160, 160)))
                last_t = now
                print(f"  Captured: {len(images)}/{TARGET}")

        bar = int((len(images) / TARGET) * (w - 40))
        cv2.rectangle(frame, (20, h - 30), (w - 20, h - 10), (80, 80, 80), -1)
        cv2.rectangle(frame, (20, h - 30), (20 + bar, h - 10), GREEN, -1)
        cv2.imshow("Register", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if len(images) < TARGET:
        return

    print("\n  Computing embeddings...")
    try:
        embs = [e for e in (get_emb(img) for img in images) if e is not None]
    except Exception as e:
        print(f"  ERROR: Embedding failed: {e}")
        print("  Registration not saved. Fix DeepFace/TensorFlow and retry.")
        return
    if len(embs) < 5:
        print("  ERROR: Could not build enough embeddings.")
        print("  Registration not saved. Fix DeepFace/TensorFlow and retry.")
        return

    db[name] = {"emb": np.mean(embs, axis=0), "imgs": images}
    save_db(db)
    print(f"  Registered: {name} ({len(embs)} embeddings)")
    print("  Saved to facedb.pkl\n")


if __name__ == "__main__":
    register()
