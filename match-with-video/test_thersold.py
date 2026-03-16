import cv2
import numpy as np
import time
from deepface import DeepFace
from database import load_encodings

MODEL = "Facenet512"


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return float(np.dot(a, b))


def get_embedding(img_bgr):
    try:
        rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = DeepFace.represent(
            img_path=rgb, model_name=MODEL,
            enforce_detection=False, detector_backend="skip"
        )
        if result:
            return np.array(result[0]["embedding"])
    except Exception:
        pass
    return None


def get_best_score(emb, enc_dict):
    best_name, best_score = "UNKNOWN", 0.0
    for name, encs in enc_dict.items():
        for enc in encs:
            s = cosine_sim(emb, enc)
            if s > best_score:
                best_score = s
                best_name  = name
    return best_name, best_score


print("\n" + "="*55)
print("  THRESHOLD TESTER")
print("  Shows REAL scores for known and unknown persons")
print("="*55)

enc_dict = load_encodings()
if not enc_dict:
    print("  No persons registered!")
    input("  Press Enter...")
    exit()

print(f"  Registered: {list(enc_dict.keys())}")
print("\n  Loading model...")
get_embedding(np.zeros((100,100,3), dtype=np.uint8))
print("  Model ready!\n")

cap = None
for idx in [1, 2, 3, 0]:
    c = cv2.VideoCapture(idx)
    if c.isOpened():
        ret, f = c.read()
        if ret and f is not None:
            cap = c
            break
        c.release()

if cap is None:
    print("  Cannot open camera!")
    input("  Press Enter...")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
CYAN   = (255, 255, 0)

known_scores   = []
unknown_scores = []
mode           = "known"  

print("="*55)
print("  STEP 1: Registered person — look at camera")
print("  Press SPACE to switch to unknown person test")
print("  Press Q to finish and see results")
print("="*55 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 55), BLACK, -1)
    if mode == "known":
        cv2.putText(frame, "STEP 1: REGISTERED PERSON — look at camera",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, GREEN, 2)
        cv2.putText(frame, "Press SPACE when done to test UNKNOWN person",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    else:
        cv2.putText(frame, "STEP 2: UNKNOWN PERSON — different person now",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, RED, 2)
        cv2.putText(frame, "Press Q to finish and see results",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2]*f[3])
        crop          = frame[y:y+fh, x:x+fw]
        emb           = get_embedding(crop)

        if emb is not None:
            name, score = get_best_score(emb, enc_dict)
            col         = GREEN if mode == "known" else RED

            cv2.rectangle(frame, (x, y), (x+fw, y+fh), col, 2)
            cv2.putText(frame, f"Score: {score:.3f}  ({name})",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

            if mode == "known":
                known_scores.append(score)
                avg = sum(known_scores)/len(known_scores)
                cv2.putText(frame,
                            f"Known avg: {avg:.3f}  samples:{len(known_scores)}",
                            (10, h-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
            else:
                unknown_scores.append(score)
                avg = sum(unknown_scores)/len(unknown_scores)
                cv2.putText(frame,
                            f"Unknown avg: {avg:.3f}  samples:{len(unknown_scores)}",
                            (10, h-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)

    cv2.imshow("Threshold Tester", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' '):
        mode = "unknown"
        print("  Switched to UNKNOWN person mode.")
        print("  Now have a DIFFERENT person stand in front of camera\n")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*55)
print("  RESULTS")
print("="*55)

if known_scores:
    k_avg = sum(known_scores) / len(known_scores)
    k_min = min(known_scores)
    print(f"\n  REGISTERED PERSON:")
    print(f"  Avg score : {k_avg:.3f}")
    print(f"  Min score : {k_min:.3f}")

if unknown_scores:
    u_avg = sum(unknown_scores) / len(unknown_scores)
    u_max = max(unknown_scores)
    print(f"\n  UNKNOWN PERSON:")
    print(f"  Avg score : {u_avg:.3f}")
    print(f"  Max score : {u_max:.3f}")

if known_scores and unknown_scores:
    k_min = min(known_scores)
    u_max = max(unknown_scores)
    ideal = (k_min + u_max) / 2

    print(f"\n  ════════════════════════════════")
    print(f"  Known   min score : {k_min:.3f}")
    print(f"  Unknown max score : {u_max:.3f}")
    print(f"\n  SET THIS IN cctv.py:")
    print(f"  THRESHOLD = {ideal:.2f}")
    print(f"  ════════════════════════════════")
    print(f"\n  Open cctv.py line ~15 and change:")
    print(f"  THRESHOLD = {ideal:.2f}")

print("\n" + "="*55)
input("  Press Enter to exit...")
