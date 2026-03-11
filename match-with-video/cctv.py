import cv2
import numpy as np
import os
import time
import threading
from datetime import datetime
from collections import deque, Counter
from database import load_db, log_entry, mark_verified

os.environ['TF_CPP_MIN_LOG_LEVEL']   = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'

os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN   = (255, 255, 0)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GRAY   = (160, 160, 160)

UNKNOWN     = "Unknown"
SMOOTH      = 8
MAX_DIST    = 140
THRESHOLD   = 0.52
MIN_GAP     = 0.06
KNOWN_RECHECK_SEC = 8
TRACK_TTL_FRAMES = 60
MAX_FRAME_WIDTH = 960
ALERT_COOLDOWN_SEC = 10
UNKNOWN_SNAPSHOT_COOLDOWN_SEC = 25
INCIDENT_DIR = "incidents"
PERSON_DETECT_INTERVAL = 4
BODY_MATCH_DIST = 180
BODY_MIN_IOU = 0.15
CACHED_DRAW_MAX_MISSED = 10
ENABLE_BODY_FALLBACK = False
MEDIAPIPE_MIN_CONF = 0.16
MEDIAPIPE_MIN_AREA = 0.0004
HAAR_MIN_NEIGHBORS = 3
HAAR_MIN_SIZE = (26, 26)
HAAR_MIN_AREA = 0.0008
MIN_HITS_TO_DRAW_UNKNOWN = 1
UNKNOWN_DRAW_MIN_SCORE = 0.08
UNKNOWN_RECHECK_SEC = 0.5
KNOWN_ACCEPT_SCORE = 0.54
KNOWN_MIN_GAP_ACCEPT = 0.10
DARK_FRAME_MEAN = 26.0
USE_MSMF_FALLBACK = False
MAX_FRAME_ERRORS = 8
CAMERA_INDEX_ORDER = [1, 2, 3, 0]
CAMERA_SCAN_MAX_INDEX = 5
TRACK_BOX_SMOOTH_NEW = 0.80
AUTO_LIGHT_TUNE = True
LOW_LIGHT_MEAN = 70.0
HIGH_LIGHT_MEAN = 185.0
LOW_CONTRAST_STD = 26.0
FAR_PASS1_INTERVAL = 2
FAR_PASS2_INTERVAL = 4

SENSITIVITY_PROFILES = {
    "strict": {
        "THRESHOLD": 0.60,
        "MIN_GAP": 0.08,
        "MEDIAPIPE_MIN_CONF": 0.24,
        "MEDIAPIPE_MIN_AREA": 0.0008,
        "MIN_HITS_TO_DRAW_UNKNOWN": 2,
        "UNKNOWN_DRAW_MIN_SCORE": 0.20,
        "KNOWN_ACCEPT_SCORE": 0.60,
    },
    "normal": {
        "THRESHOLD": 0.52,
        "MIN_GAP": 0.06,
        "MEDIAPIPE_MIN_CONF": 0.20,
        "MEDIAPIPE_MIN_AREA": 0.0006,
        "MIN_HITS_TO_DRAW_UNKNOWN": 1,
        "UNKNOWN_DRAW_MIN_SCORE": 0.12,
        "KNOWN_ACCEPT_SCORE": 0.52,
    },
    "lenient": {
        "THRESHOLD": 0.45,
        "MIN_GAP": 0.04,
        "MEDIAPIPE_MIN_CONF": 0.12,
        "MEDIAPIPE_MIN_AREA": 0.0002,
        "MIN_HITS_TO_DRAW_UNKNOWN": 1,
        "UNKNOWN_DRAW_MIN_SCORE": 0.05,
        "KNOWN_ACCEPT_SCORE": 0.45,
    },
}



_model_loaded = False
_detect_cycle = 0
_light_tune_cycle = 0


def load_model():
    global _model_loaded
    if _model_loaded:
        return True
    try:
        from deepface import DeepFace
    except Exception as e:
        print("\n  ERROR: DeepFace/TensorFlow could not be loaded.")
        print(f"  Details: {e}\n")
        return False
    dummy = np.zeros((160, 160, 3), dtype=np.uint8)
    try:
        DeepFace.represent(
            img_path=dummy, model_name="Facenet512",
            enforce_detection=False, detector_backend="skip"
        )
    except Exception as e:
        print("\n  ERROR: Face model warmup failed.")
        print(f"  Details: {e}\n")
        return False
    _model_loaded = True
    return True


def get_embedding(img_bgr):
    try:
        from deepface import DeepFace

        
        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        h, w = rgb.shape[:2]
        if h < 100 or w < 100:
            rgb = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_CUBIC)
        result = DeepFace.represent(
            img_path          = rgb,
            model_name        = "Facenet512",
            enforce_detection = False,
            detector_backend  = "skip"
        )
        if result:
            return np.array(result[0]["embedding"])
    except Exception:
        pass
    return None


def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return float(np.dot(a, b))


def _safe_name(name):
    out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))
    return out.strip("_") or "person"


def get_padded_crop(frame, x, y, w, h, pad_ratio=0.35):
    h0, w0 = frame.shape[:2]
    px = int(w * pad_ratio)
    py = int(h * pad_ratio)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(w0, x + w + px)
    y2 = min(h0, y + h + py)
    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return crop
    return np.ascontiguousarray(crop)


def is_valid_frame(frame):
    return (
        frame is not None and
        isinstance(frame, np.ndarray) and
        frame.ndim == 3 and
        frame.shape[0] > 0 and
        frame.shape[1] > 0 and
        frame.shape[2] >= 3
    )


def sanitize_bgr_frame(frame):
    if frame is None or not isinstance(frame, np.ndarray):
        return None
    if frame.ndim == 2:
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        except Exception:
            return None
    elif frame.ndim == 3 and frame.shape[2] == 4:
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except Exception:
            return None
    if not is_valid_frame(frame):
        return None
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def bgr_to_gray_safe(frame):
    frame = sanitize_bgr_frame(frame)
    if frame is None:
        return None
    
    b = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    r = frame[:, :, 2].astype(np.float32)
    gray = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
    return gray


def frame_mean_brightness(frame):
    frame = sanitize_bgr_frame(frame)
    if frame is None:
        return 0.0
    
    gray = bgr_to_gray_safe(frame)
    if gray is None:
        return 0.0
    return float(np.mean(gray))


def frame_std_brightness(frame):
    frame = sanitize_bgr_frame(frame)
    if frame is None:
        return 0.0
    gray = bgr_to_gray_safe(frame)
    if gray is None:
        return 0.0
    return float(np.std(gray))


def enhance_dark_frame(frame):
    frame = sanitize_bgr_frame(frame)
    if frame is None:
        return frame
    try:
        mean_bri = frame_mean_brightness(frame)
        if mean_bri >= DARK_FRAME_MEAN:
            return frame
    
        return cv2.convertScaleAbs(frame, alpha=1.45, beta=24)
    except cv2.error:
        return frame


def tune_frame_for_detection(frame):
    """Lightweight auto-tuning for difficult room lighting."""
    global _light_tune_cycle
    _light_tune_cycle += 1
    frame = sanitize_bgr_frame(frame)
    if frame is None or not AUTO_LIGHT_TUNE:
        return frame
    try:
        mean_bri = frame_mean_brightness(frame)
        std_bri = frame_std_brightness(frame)
        tuned = frame

        
        if mean_bri < LOW_LIGHT_MEAN:
            tuned = cv2.convertScaleAbs(tuned, alpha=1.30, beta=16)
        
        elif mean_bri > HIGH_LIGHT_MEAN:
            tuned = cv2.convertScaleAbs(tuned, alpha=0.82, beta=-16)


        if std_bri < LOW_CONTRAST_STD and (_light_tune_cycle % 3 == 0):
            lab = cv2.cvtColor(tuned, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            tuned = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        return tuned
    except cv2.error:
        return frame


def _open_camera_candidate(idx, backend):
    cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        return None, -1.0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    if hasattr(cv2, "CAP_PROP_AUTO_WB"):
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)

    score = 0.0
    valid_n = 0
    for _ in range(14):
        ret, frm = cap.read()
        frm = sanitize_bgr_frame(frm) if ret else None
        if frm is not None:
            m = frame_mean_brightness(frm)
            s = frame_std_brightness(frm)
            valid_n += 1
            score = max(score, m + 1.2 * s)
    if valid_n < 2:
        cap.release()
        return None, -1.0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap, score


def open_live_camera(preferred_index=None):
    
    backends = [None]
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if USE_MSMF_FALLBACK and hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)

    
    if preferred_index is not None:
        for backend in backends:
            cap, score = _open_camera_candidate(int(preferred_index), backend)
            if cap is not None:
                return cap, int(preferred_index)
        return None, None

    best_cap = None
    best_score = -1.0
    best_idx = None

    idx_list = []
    for i in CAMERA_INDEX_ORDER:
        if i not in idx_list:
            idx_list.append(i)
    for i in range(CAMERA_SCAN_MAX_INDEX + 1):
        if i not in idx_list:
            idx_list.append(i)

    for idx in idx_list:
        for backend in backends:
            cap, score = _open_camera_candidate(idx, backend)
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


def save_incident(frame, x, y, w, h, name, score, source):
    os.makedirs(INCIDENT_DIR, exist_ok=True)
    day_dir = os.path.join(INCIDENT_DIR, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(day_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sid = _safe_name(name)
    base = f"{ts}_{sid}_{score:.2f}"
    frame_path = os.path.join(day_dir, f"{base}_frame.jpg")
    face_path = os.path.join(day_dir, f"{base}_face.jpg")
    meta_path = os.path.join(day_dir, "incident_log.csv")


    marked = frame.copy()
    cv2.rectangle(marked, (x, y), (x+w, y+h), RED if name == UNKNOWN else GREEN, 2)
    cv2.putText(marked, f"{name} {score:.2f}", (x, max(20, y-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    cv2.imwrite(frame_path, marked)

    face = frame[max(0, y):max(0, y)+h, max(0, x):max(0, x)+w]
    if face.size != 0:
        cv2.imwrite(face_path, face)

    source_tag = _safe_name(source)
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{source_tag},{name},{score:.3f},{frame_path},{face_path}\n")




def build_references(db):
    refs = {}
    print("\n  Building face references...")
    for name, data in db.items():
        prototypes = []
        if isinstance(data, dict) and "emb" in data:
            prototypes.append(np.array(data["emb"]))
            imgs = data.get("imgs", [])
            if isinstance(imgs, list) and imgs:
                step = max(1, len(imgs) // 6)
                for img in imgs[::step][:6]:
                    e = get_embedding(img)
                    if e is not None:
                        prototypes.append(e)
        elif isinstance(data, list):
            embs = []
            for img in data:
                e = get_embedding(img)
                if e is not None:
                    embs.append(e)
            if embs:
                prototypes = [np.mean(embs, axis=0)] + embs[:6]
        if prototypes:
            refs[name] = prototypes
            print(f"  {name}: {len(prototypes)} reference vectors ready")
        else:
            print(f"  {name}: FAILED - no references")
    print(f"  References ready for: {list(refs.keys())}\n")
    return refs


def identify(embedding, refs):
    best_name  = UNKNOWN
    best_score = -1.0
    second_best = -1.0
    for name, ref_set in refs.items():
        if isinstance(ref_set, np.ndarray):
            ref_set = [ref_set]
        scores = [cosine(embedding, ref_emb) for ref_emb in ref_set if ref_emb is not None]
        if not scores:
            continue
        scores.sort(reverse=True)
        if len(scores) >= 2:
            score = 0.7 * scores[0] + 0.3 * scores[1]
        else:
            score = scores[0]
        if score > best_score:
            second_best = best_score
            best_score = score
            best_name  = name
        elif score > second_best:
            second_best = score
    best_gap = best_score - second_best
    if best_score >= THRESHOLD and best_gap >= MIN_GAP:
        return best_name, best_score, best_gap
    return UNKNOWN, best_score, best_gap



class FaceWorker:
    def __init__(self, refs):
        self.refs   = refs
        self.result = None
        self.busy   = False
        self.lock   = threading.Lock()
        self.token  = 0

    def submit(self, face_crop):
        if self.busy:
            return
        self.busy = True

        def run():
            crop = face_crop
            if crop.shape[0] < 80:
                crop = cv2.resize(crop, (160, 160),
                                  interpolation=cv2.INTER_CUBIC)
            emb = get_embedding(crop)
            if emb is not None:
                name, score, gap = identify(emb, self.refs)
                if name == UNKNOWN:
                    
                    flip = cv2.flip(crop, 1)
                    emb2 = get_embedding(flip)
                    if emb2 is not None:
                        name2, score2, gap2 = identify(emb2, self.refs)
                        if score2 > score:
                            name, score, gap = name2, score2, gap2
            else:
                name, score, gap = UNKNOWN, 0.0, 0.0
            with self.lock:
                self.token += 1
                self.result = (name, score, gap, self.token)
            self.busy = False

        threading.Thread(target=run, daemon=True).start()

    def get(self):
        with self.lock:
            return self.result




class FaceTracker:
    def __init__(self, refs):
        self.refs   = refs
        self.tracks = {}
        self.nid    = 0
        self.used_in_frame = set()

    def start_frame(self):
        self.used_in_frame.clear()

    def _nearest(self, x, y, w, h):
        cx = x + w // 2
        cy = y + h // 2
        cand_box = (x, y, w, h)
        best_id, best_score = None, float("inf")
        for tid, d in self.tracks.items():
            if tid in self.used_in_frame:
                continue
            px, py = d['pos']
            vx, vy = d.get('vel', (0.0, 0.0))
            
            ppx = px + vx
            ppy = py + vy
            dist = ((cx - ppx) ** 2 + (cy - ppy) ** 2) ** 0.5

            prev_box = d.get('bbox', (x, y, w, h))
            iou = bbox_iou(cand_box, prev_box)
            pw, ph = max(1, prev_box[2]), max(1, prev_box[3])
            size_ratio = (w * h) / float(pw * ph)
            if size_ratio < 0.45 or size_ratio > 2.2:
                continue

        
            score = dist - (220.0 * iou)
            if score < best_score:
                best_score  = score
                best_id = tid

        if best_id is not None and best_score > MAX_DIST:
            return None
        return best_id

    def get_id(self, x, y, w, h):
        cx  = x + w//2
        cy  = y + h//2
        tid = self._nearest(x, y, w, h)
        if tid is None:
            tid = self.nid
            self.nid += 1
            self.tracks[tid] = {
                'pos':    (cx, cy),
                'vel':    (0.0, 0.0),
                'bbox':   (x, y, w, h),
                'names':  deque(maxlen=SMOOTH),
                'worker': FaceWorker(self.refs),
                'name':   None,
                'score':  0.0,
                'color':  YELLOW,
                'logged': "",
                'last_token': 0,
                'last_verify_ts': 0.0,
                'last_unknown_verify_ts': 0.0,
                'verify_state': "Checking...",
                'missed': 0,
                'seen': 1
            }
        else:
            px, py = self.tracks[tid]['pos']
            ox, oy, ow, oh = self.tracks[tid].get('bbox', (x, y, w, h))
            a_new = max(0.5, min(0.95, TRACK_BOX_SMOOTH_NEW))
            a_old = 1.0 - a_new
            sx = int(a_old * ox + a_new * x)
            sy = int(a_old * oy + a_new * y)
            sw = int(a_old * ow + a_new * w)
            sh = int(a_old * oh + a_new * h)
            self.tracks[tid]['pos'] = (sx + sw // 2, sy + sh // 2)
            vx = cx - px
            vy = cy - py
            ovx, ovy = self.tracks[tid]['vel']
            self.tracks[tid]['vel'] = (0.7 * ovx + 0.3 * vx, 0.7 * ovy + 0.3 * vy)
            self.tracks[tid]['bbox'] = (sx, sy, max(1, sw), max(1, sh))
            self.tracks[tid]['missed'] = 0
            self.tracks[tid]['seen'] = min(9999, self.tracks[tid].get('seen', 0) + 1)
        self.used_in_frame.add(tid)
        return tid

    def update(self, tid, face_crop):
        if tid not in self.tracks:
            return
        t = self.tracks[tid]
        now = time.time()

        if t['name'] and t['name'] != UNKNOWN:
            if (now - t['last_verify_ts']) < KNOWN_RECHECK_SEC:
                t['verify_state'] = "Verified (cached)"
                return t['name'], t['score'], t['last_token']
            t['verify_state'] = "Re-checking"
        else:
            if (now - t.get('last_unknown_verify_ts', 0.0)) < UNKNOWN_RECHECK_SEC:
                return t['name'], t['score'], t['last_token']
            t['last_unknown_verify_ts'] = now
            t['verify_state'] = "Checking..."

        t['worker'].submit(face_crop)
        result = t['worker'].get()
        if result is not None:
            if len(result) == 4:
                name, score, gap, token = result
            else:
                name, score, token = result
                gap = 0.0
            if token == t['last_token']:
                return result
            t['last_token'] = token
            t['names'].append(name)
            if len(t['names']) >= 3:
                cnt         = Counter(t['names'])
                best, count = cnt.most_common(1)[0]
                ratio = count / len(t['names'])
                if ratio >= 0.6:
                    if best != UNKNOWN and score >= KNOWN_ACCEPT_SCORE and gap >= KNOWN_MIN_GAP_ACCEPT:
                        t['name']  = best
                        t['color'] = GREEN
                        t['score'] = score
                        t['last_verify_ts'] = now
                        t['verify_state'] = "Verified"
                    else:
                        t['name']  = UNKNOWN
                        t['color'] = RED
                        t['score'] = score
                        t['verify_state'] = "Unknown"
        return result

    def should_log(self, tid):
        if tid not in self.tracks:
            return False, None, 0
        t    = self.tracks[tid]
        name = t['name']
        if name and name != UNKNOWN and name != t['logged'] and t.get('score', 0.0) >= KNOWN_ACCEPT_SCORE:
            t['logged'] = name
            score = t.get('score', 0.0)
            return True, name, score
        return False, None, 0

    def cleanup(self, current_ids):
        current_ids = set(current_ids)
        for tid in list(self.tracks):
            if tid in current_ids:
                self.tracks[tid]['missed'] = 0
                continue
            self.tracks[tid]['missed'] += 1
            self.tracks[tid]['seen'] = max(0, self.tracks[tid].get('seen', 0) - 1)
        
            t = self.tracks[tid]
            if t.get('name') and t.get('name') != UNKNOWN:
                x, y, w, h = t.get('bbox', (0, 0, 0, 0))
                vx, vy = t.get('vel', (0.0, 0.0))
                nx = int(x + vx)
                ny = int(y + vy)
                t['bbox'] = (nx, ny, w, h)
                t['pos'] = (nx + w // 2, ny + h // 2)
                t['verify_state'] = "Tracking (cached)"
            if self.tracks[tid]['missed'] > TRACK_TTL_FRAMES:
                del self.tracks[tid]

    def reset(self):
        self.tracks.clear()


def draw_box(frame, x, y, w, h, name, color, score, state=None):
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    L = 16
    for px, py, dx, dy in [
        (x,   y,    1,  1), (x+w, y,   -1,  1),
        (x,   y+h,  1, -1), (x+w, y+h, -1, -1)
    ]:
        cv2.line(frame, (px, py), (px+dx*L, py), color, 3)
        cv2.line(frame, (px, py), (px, py+dy*L), color, 3)

    label = name if name else "..."
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    ly = y-12 if y > th+60 else y+h+th+14
    cv2.rectangle(frame, (x, ly-th-8), (x+tw+12, ly+5), color, -1)
    cv2.putText(frame, label, (x+6, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, BLACK, 2)
    if name and name != UNKNOWN and score > 0:
        cv2.putText(frame, f"{score:.2f}",
                    (x+4, y+h+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    if state and name != UNKNOWN:
        sy = y+h+30 if (name and name != UNKNOWN and score > 0) else y+h+16
        cv2.putText(frame, state,
                    (x+4, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)


def draw_overlay(frame, n, fps, src, logs, paused):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 48), BLACK, -1)
    cv2.putText(frame, f"CCTV  [{src}]",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, CYAN, 2)
    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                (w-245, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    if int(time.time()*2) % 2 == 0:
        cv2.circle(frame, (w-28, 24), 9, RED, -1)
        cv2.putText(frame, "REC", (w-63, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
    cv2.rectangle(frame, (0, h-36), (w, h), BLACK, -1)
    cv2.putText(frame, f"FPS:{fps:.0f}  Faces:{n}",
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
    cv2.putText(frame, "SPACE=pause  Q=quit",
                (w-210, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1)
    if logs:
        px = w-230
        cv2.rectangle(frame, (px-4, 52),
                      (w, 52+26+len(logs)*24), BLACK, -1)
        cv2.putText(frame, "ENTRY LOG",
                    (px, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
        for i, (ts, nm) in enumerate(logs):
            cv2.putText(frame, f"{ts}  {nm}",
                        (px, 92+i*24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        GREEN if nm != UNKNOWN else RED, 1)
    if paused:
        cv2.putText(frame, "PAUSED", (w//2-80, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, YELLOW, 4)

def setup_detector():
    """MediaPipe-only detector for reliable multi-person face detection."""
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        detector = mp_face.FaceDetection(
            model_selection=1,          
            min_detection_confidence=MEDIAPIPE_MIN_CONF
        )
        print("  Detector: MediaPipe âœ“ (works up to 5m)")
        return "mediapipe", detector
    except Exception as e:
        print("\n  ERROR: MediaPipe is required for CCTV detection.")
        print("  Install with: pip install mediapipe")
        print(f"  Details: {e}\n")
        return None, None


def detect_faces(frame, detector_type, detector):
    """Returns list of (x, y, w, h) â€” only real human faces"""
    global _detect_cycle
    _detect_cycle += 1
    frame = sanitize_bgr_frame(frame)
    if frame is None:
        return []
    try:
        det_frame = tune_frame_for_detection(frame)
        if det_frame is None:
            det_frame = frame
        faces      = []
        h, w       = det_frame.shape[:2]
        frame_area = h * w

        if detector_type == "mediapipe":
            def run_mp_pass(src, scale_x=1.0, scale_y=1.0, min_conf=None, area_scale=1.0):
                out = []
                src = sanitize_bgr_frame(src)
                if src is None:
                    return out
                rgb = np.ascontiguousarray(src[:, :, ::-1])
                results = detector.process(rgb)
                if not results.detections:
                    return out
                sh, sw = src.shape[:2]
                s_area = sh * sw
                for det in results.detections:
                    score = det.score[0] if det.score else 0
                    conf_thr = MEDIAPIPE_MIN_CONF if min_conf is None else min_conf
                    if score < conf_thr:
                        continue
                    bb  = det.location_data.relative_bounding_box
                    x   = max(0, int(bb.xmin * sw))
                    y   = max(0, int(bb.ymin * sh))
                    bw  = min(int(bb.width  * sw), sw - x)
                    bh  = min(int(bb.height * sh), sh - y)
                    if bw < 16 or bh < 16:
                        continue
                    if bw * bh <= s_area * MEDIAPIPE_MIN_AREA * max(0.4, area_scale):
                        continue

                    ox = int(x * scale_x)
                    oy = int(y * scale_y)
                    ow = int(bw * scale_x)
                    oh = int(bh * scale_y)
                    ox = max(0, min(ox, w - 1))
                    oy = max(0, min(oy, h - 1))
                    ow = max(1, min(ow, w - ox))
                    oh = max(1, min(oh, h - oy))

                    if score >= 0.50:
                        out.append((ox, oy, ow, oh))
                        continue
                    crop = frame[oy:oy+oh, ox:ox+ow]
                    if is_face_like_crop(crop):
                        out.append((ox, oy, ow, oh))
                return out

            faces.extend(run_mp_pass(det_frame, 1.0, 1.0))

            
            if len(faces) <= 1 and (_detect_cycle % FAR_PASS1_INTERVAL == 0):
                up = cv2.resize(det_frame, None, fx=1.45, fy=1.45, interpolation=cv2.INTER_CUBIC)
                up = cv2.convertScaleAbs(up, alpha=1.18, beta=10)
                faces.extend(run_mp_pass(
                    up,
                    1.0/1.45, 1.0/1.45,
                    min_conf=max(0.08, MEDIAPIPE_MIN_CONF - 0.04),
                    area_scale=0.70
                ))
            if len(faces) <= 2 and (_detect_cycle % FAR_PASS2_INTERVAL == 0):
                up2 = cv2.resize(det_frame, None, fx=1.75, fy=1.75, interpolation=cv2.INTER_CUBIC)
                up2 = cv2.convertScaleAbs(up2, alpha=1.22, beta=14)
                faces.extend(run_mp_pass(
                    up2,
                    1.0/1.75, 1.0/1.75,
                    min_conf=max(0.07, MEDIAPIPE_MIN_CONF - 0.05),
                    area_scale=0.55
                ))

        elif detector_type == "haar":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frontal = detector["frontal"] if isinstance(detector, dict) else detector
            profile = detector.get("profile") if isinstance(detector, dict) else None

            raw  = frontal.detectMultiScale(
                cv2.equalizeHist(gray),
                scaleFactor=1.1,
                minNeighbors=HAAR_MIN_NEIGHBORS,
                minSize=HAAR_MIN_SIZE
            )
            raw_profile = []
            if (len(raw) == 0) and profile is not None and not profile.empty():
                rp = profile.detectMultiScale(
                    cv2.equalizeHist(gray),
                    scaleFactor=1.1,
                    minNeighbors=max(3, HAAR_MIN_NEIGHBORS - 1),
                    minSize=HAAR_MIN_SIZE
                )
                raw_profile = list(rp) if rp is not None else []
                gray_flip = cv2.flip(gray, 1)
                rpf = profile.detectMultiScale(
                    cv2.equalizeHist(gray_flip),
                    scaleFactor=1.1,
                    minNeighbors=max(3, HAAR_MIN_NEIGHBORS - 1),
                    minSize=HAAR_MIN_SIZE
                )
                raw_profile_flip = list(rpf) if rpf is not None else []
                for (x, y, fw, fh) in raw_profile_flip:
                    raw_profile.append((w - x - fw, y, fw, fh))

            if len(raw) > 0:
                for (x, y, fw, fh) in raw:
                    if fw * fh <= frame_area * HAAR_MIN_AREA:
                        continue
                    crop = frame[y:y+fh, x:x+fw]
                    if is_face_like_crop(crop):
                        faces.append((x, y, fw, fh))
            if len(raw_profile) > 0:
                for (x, y, fw, fh) in raw_profile:
                    if fw * fh <= frame_area * HAAR_MIN_AREA:
                        continue
                    crop = frame[y:y+fh, x:x+fw]
                    if is_face_like_crop(crop):
                        faces.append((x, y, fw, fh))

        return clean_face_boxes(faces)
    except cv2.error:
        return []


def setup_person_detector():
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print("  Person tracker: HOG enabled")
        return hog
    except Exception:
        print("  Person tracker: disabled")
        return None


def detect_persons(frame, person_detector):
    if person_detector is None or not is_valid_frame(frame):
        return []
    try:
        boxes, _ = person_detector.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )
    except Exception:
        return []

    persons = []
    h, w = frame.shape[:2]
    for (x, y, bw, bh) in boxes:
        x = max(0, int(x))
        y = max(0, int(y))
        bw = int(min(bw, w - x))
        bh = int(min(bh, h - y))
        if bw > 0 and bh > 0:
            persons.append((x, y, bw, bh))
    return persons


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / max(union, 1))


def is_face_like_crop(crop):
    
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    if h < 24 or w < 24:
        return False
    ar = w / max(h, 1)
    if ar < 0.55 or ar > 1.6:
        return False

    
    if min(h, w) < 52:
        gray = bgr_to_gray_safe(crop)
        if gray is None:
            return False
        mean_bri = float(np.mean(gray))
        if mean_bri < 8 or mean_bri > 250:
            return False
        if float(np.std(gray)) < 8.0:
            return False
        return True

    gray = bgr_to_gray_safe(crop)
    if gray is None:
        return False
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 12.0:
        return False

    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    except cv2.error:
        return False
    
    skin_mask = cv2.inRange(hsv, (0, 18, 35), (35, 255, 255))
    skin_ratio = float(np.count_nonzero(skin_mask)) / float(h * w)
    return 0.03 <= skin_ratio <= 0.98


def nms_boxes(boxes, iou_thr=0.35):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        nxt = []
        for b in boxes:
            if bbox_iou(best, b) < iou_thr:
                nxt.append(b)
        boxes = nxt
    return keep


def _center_in_box(inner, outer):
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    cx = ix + iw * 0.5
    cy = iy + ih * 0.5
    return (ox <= cx <= ox + ow) and (oy <= cy <= oy + oh)


def clean_face_boxes(boxes):
    
    boxes = nms_boxes(boxes, iou_thr=0.30)
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    keep = []
    for b in boxes:
        bx, by, bw, bh = b
        b_area = bw * bh
        drop = False
        for k in keep:
            k_area = k[2] * k[3]
            iou = bbox_iou(b, k)
            if iou > 0.22:
                drop = True
                break
    
            if b_area < 0.58 * k_area and _center_in_box(b, k):
                drop = True
                break
        if not drop:
            keep.append(b)
    return keep


def resolve_duplicate_known_names(faces_info):

    best_by_name = {}
    for i, item in enumerate(faces_info):
        _, _, _, _, name, _, score, _ = item
        if not name or name == UNKNOWN:
            continue
        if name not in best_by_name or score > best_by_name[name][1]:
            best_by_name[name] = (i, score)

    winners = {v[0] for v in best_by_name.values()}
    out = []
    for i, item in enumerate(faces_info):
        x, y, w, h, name, color, score, state = item
        if name and name != UNKNOWN and i not in winners:
            out.append((x, y, w, h, UNKNOWN, RED, score, "Re-checking"))
        else:
            out.append(item)
    return out


def apply_sensitivity_profile(mode):
    global THRESHOLD, MIN_GAP, MEDIAPIPE_MIN_CONF, MEDIAPIPE_MIN_AREA
    global MIN_HITS_TO_DRAW_UNKNOWN, UNKNOWN_DRAW_MIN_SCORE, KNOWN_ACCEPT_SCORE

    m = (mode or "normal").strip().lower()
    if m not in SENSITIVITY_PROFILES:
        m = "normal"
    p = SENSITIVITY_PROFILES[m]
    THRESHOLD = p["THRESHOLD"]
    MIN_GAP = p["MIN_GAP"]
    MEDIAPIPE_MIN_CONF = p["MEDIAPIPE_MIN_CONF"]
    MEDIAPIPE_MIN_AREA = p["MEDIAPIPE_MIN_AREA"]
    MIN_HITS_TO_DRAW_UNKNOWN = p["MIN_HITS_TO_DRAW_UNKNOWN"]
    UNKNOWN_DRAW_MIN_SCORE = p["UNKNOWN_DRAW_MIN_SCORE"]
    KNOWN_ACCEPT_SCORE = p["KNOWN_ACCEPT_SCORE"]
    return m



def cctv(sensitivity_mode="normal"):
    print("\n" + "="*50)
    print("  CCTV FACE RECOGNITION")
    print("="*50)
    active_mode = apply_sensitivity_profile(sensitivity_mode)
    print(f"  Sensitivity mode: {active_mode}")

    db = load_db()
    if not db:
        print("  No registered persons found.")
        ch = input("  Register a person now? (y/n): ").strip().lower()
        if ch == "y":
            try:
                from register import register
                register()
            except Exception:
                pass
            db = load_db()
        if not db:
            print("  ERROR: Register at least one person first.")
            input("  Press Enter...")
            return "no_registered_persons"

    print(f"  Registered: {list(db.keys())}")
    print("  Loading model (30 sec first time)...")
    if not load_model():
        input("  Press Enter...")
        return "deepface_unavailable"

    refs = build_references(db)
    if not refs:
        print("  ERROR: Could not build face references.")
        input("  Press Enter...")
        return "refs_failed"

    print("  1. Video file")
    print("  2. Camera (live)")
    choice = input("  Choice (1/2): ").strip()

    cam_cycle = CAMERA_INDEX_ORDER[:]
    cam_cycle_pos = 0

    def _next_live_index():
        nonlocal cam_cycle_pos
        if not cam_cycle:
            return None
        idx = cam_cycle[cam_cycle_pos % len(cam_cycle)]
        cam_cycle_pos = (cam_cycle_pos + 1) % len(cam_cycle)
        return idx

    if choice == "1":
        path = input("  Video path: ").strip().strip('"\'')
        if not os.path.exists(path):
            print("  Not found.")
            return "video_not_found"
        cap      = cv2.VideoCapture(path)
        source   = os.path.basename(path)
        is_video = True
    else:
        first_idx = _next_live_index()
        cap, current_cam_idx = open_live_camera(first_idx)
        if cap is None:
            cap, current_cam_idx = open_live_camera()
        if cap is None:
            print("  No camera found.")
            return "camera_not_found"
        print(f"  Camera selected: index {current_cam_idx}")
        source   = "LIVE CAMERA"
        is_video = False


    detector_type, detector = setup_detector()
    if detector_type is None or detector is None:
        input("  Press Enter...")
        return "mediapipe_missing"
    person_detector = setup_person_detector() if ENABLE_BODY_FALLBACK else None

    print("  Controls: SPACE=pause  Q=quit\n")

    tracker   = FaceTracker(refs)
    entry_log = deque(maxlen=7)
    prev_t    = time.time()
    fps       = 0.0
    paused    = False
    no_face_n = 0
    first     = True
    frame_no  = 0
    dark_n    = 0
    frame_err_n = 0
    read_fail_n = 0

    while True:
        try:
            frame_no += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                paused = not paused

            if paused:
                cv2.waitKey(80)
                continue

            if first:
                first = False
                ret, frame = cap.read()
                frame = sanitize_bgr_frame(frame) if ret else None
                if not ret or frame is None:
                    break
                if not is_video:
                    frame = cv2.flip(frame, 1)
            else:
                ret, frame = cap.read()
                frame = sanitize_bgr_frame(frame) if ret else None
                if not ret or frame is None:
                    if is_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        tracker.reset()
                        continue
                    read_fail_n += 1
                    if read_fail_n >= 18:
                        print("  Reopening camera: repeated read failures...")
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap, current_cam_idx = open_live_camera(_next_live_index())
                        if cap is None:
                            cap, current_cam_idx = open_live_camera()
                        read_fail_n = 0
                        if cap is None:
                            print("  Camera reopen failed.")
                            return "camera_not_found"
                    continue
                read_fail_n = 0
                if not is_video:
                    frame = cv2.flip(frame, 1)

            frame = sanitize_bgr_frame(frame)
            if frame is None:
                continue
            if not is_video:
                frame = enhance_dark_frame(frame)
                m_live = frame_mean_brightness(frame)
                s_live = frame_std_brightness(frame)
                if m_live < 8.0 or s_live < 3.0:
                    dark_n += 1
                else:
                    dark_n = 0
                if dark_n > 45:
                    print("  Reopening camera: stream too dark/flat...")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap, current_cam_idx = open_live_camera(_next_live_index())
                    if cap is None:
                        cap, current_cam_idx = open_live_camera()
                    dark_n = 0
                    if cap is None:
                        print("  Camera reopen failed.")
                        return "camera_not_found"
                    print(f"  Camera selected: index {current_cam_idx}")
                    continue

            hf, wf = frame.shape[:2]
            if wf > MAX_FRAME_WIDTH:
                s     = MAX_FRAME_WIDTH / wf
                frame = cv2.resize(frame, (MAX_FRAME_WIDTH, int(hf*s)))

            now    = time.time()
            fps    = 0.9*fps + 0.1/max(now-prev_t, 0.001)
            prev_t = now

            faces = detect_faces(frame, detector_type, detector)
            tracker.start_frame()

            current_ids = []
            current_id_set = set()
            faces_info  = []
            live_known_names = set()

            if not faces:
                no_face_n += 1
            else:
                no_face_n = 0
                for (x, y, w, h) in sorted(faces, key=lambda f: f[2]*f[3], reverse=True):
                    crop = get_padded_crop(frame, x, y, w, h, pad_ratio=0.35)
                    if crop is None or crop.size == 0:
                        continue

                    tid = tracker.get_id(x, y, w, h)
                    current_ids.append(tid)
                    current_id_set.add(tid)
                    tracker.update(tid, crop)

                    if tid in tracker.tracks:
                        t     = tracker.tracks[tid]
                        dname = t['name']
                        color = t['color']
                        score = t.get('score', 0.0)
                        state = t.get('verify_state', "")
                        seen  = t.get('seen', 0)

                        if dname is None:
                            continue
                        if dname == UNKNOWN and (seen < MIN_HITS_TO_DRAW_UNKNOWN or score < UNKNOWN_DRAW_MIN_SCORE):
                            continue

                        do_log, log_name, log_score = tracker.should_log(tid)
                        if do_log:
                            ts = datetime.now().strftime("%H:%M:%S")
                            entry_log.appendleft((ts, log_name))
                            log_entry(log_name, log_score)
                            mark_verified(log_name)
                            print(f"  [{ts}] {log_name}  score:{log_score:.2f}")

                        faces_info.append((x, y, w, h, dname, color, score, state))
                        if dname and dname != UNKNOWN:
                            live_known_names.add(dname)

            tracker.cleanup(current_ids)

            if person_detector is not None and frame_no % PERSON_DETECT_INTERVAL == 0:
                body_boxes = detect_persons(frame, person_detector)
                if body_boxes:
                    used_bodies = set()
                    for tid, t in tracker.tracks.items():
                        if tid in current_id_set:
                            continue
                        if not (t.get('name') and t.get('name') != UNKNOWN):
                            continue
                        if t.get('missed', 0) <= 0:
                            continue
                        tx, ty = t['pos']
                        prev_box = t.get('bbox', None)
                        best_i = None
                        best_score = 1e9
                        for i, (bx, by, bw, bh) in enumerate(body_boxes):
                            if i in used_bodies:
                                continue
                            bc = (bx + bw // 2, by + bh // 2)
                            d = ((tx - bc[0]) ** 2 + (ty - bc[1]) ** 2) ** 0.5
                            if d > BODY_MATCH_DIST:
                                continue
                            iou = bbox_iou(prev_box, (bx, by, bw, bh)) if prev_box else 0.0
                            if prev_box and iou < BODY_MIN_IOU:
                                continue
                            score = d - (180.0 * iou)
                            if score < best_score:
                                best_score = score
                                best_i = i
                        if best_i is not None:
                            bx, by, bw, bh = body_boxes[best_i]
                            used_bodies.add(best_i)
                            oldx, oldy, _, _ = t.get('bbox', (bx, by, bw, bh))
                            t['bbox'] = (bx, by, bw, bh)
                            t['pos'] = (bx + bw // 2, by + bh // 2)
                            t['vel'] = (0.7 * t['vel'][0] + 0.3 * (bx - oldx),
                                        0.7 * t['vel'][1] + 0.3 * (by - oldy))
                            t['verify_state'] = "Tracking (body)"

            fh, fw = frame.shape[:2]
            for tid, t in tracker.tracks.items():
                if tid in current_id_set:
                    continue
                if not (t.get('name') and t.get('name') != UNKNOWN):
                    continue
                if t.get('name') in live_known_names:
                    continue
                if t.get('missed', 0) <= 0 or t.get('missed', 0) > CACHED_DRAW_MAX_MISSED:
                    continue

                x, y, w, h = t.get('bbox', (0, 0, 0, 0))
                if w <= 0 or h <= 0:
                    continue
                x = max(0, min(x, fw - 1))
                y = max(0, min(y, fh - 1))
                if x + w > fw:
                    w = max(1, fw - x)
                if y + h > fh:
                    h = max(1, fh - y)
                faces_info.append((x, y, w, h, t['name'], t['color'],
                                   t.get('score', 0.0), "Tracking (cached)"))

            faces_info = resolve_duplicate_known_names(faces_info)

            for item in faces_info:
                draw_box(frame, *item)

            draw_overlay(frame, len(faces_info), fps, source, entry_log, paused)
            cv2.imshow("CCTV", frame)
            
            frame_err_n = 0

        except cv2.error as e:
            if frame_err_n % 5 == 0:
                print(f"  OpenCV frame error (recovered): {e}")
            frame_err_n += 1
            if not is_video and frame_err_n >= MAX_FRAME_ERRORS:
                print("  Reopening camera due to repeated frame errors...")
                try:
                    cap.release()
                except Exception:
                    pass
                cap, current_cam_idx = open_live_camera(_next_live_index())
                if cap is None:
                    cap, current_cam_idx = open_live_camera()
                frame_err_n = 0
                if cap is None:
                    print("  Camera reopen failed.")
                    return "camera_not_found"
                print(f"  Camera selected: index {current_cam_idx}")
            continue

    cap.release()
    cv2.destroyAllWindows()
    print("  Stopped.\n")
    return "ok"


if __name__ == "__main__":
    cctv()
