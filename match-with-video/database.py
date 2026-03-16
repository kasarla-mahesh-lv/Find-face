import os
import pickle
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE  = os.path.join(BASE_DIR, "facedb.pkl")
LOG_FILE = os.path.join(BASE_DIR, "entry_log.txt")


def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def save_db(data):
    with open(DB_FILE, "wb") as f:
        pickle.dump(data, f)


def is_empty():
    return len(load_db()) == 0


def get_persons():
    db = load_db()
    out = []
    for k, v in db.items():
        if isinstance(v, dict):
            n = len(v.get("imgs", [])) if isinstance(v.get("imgs", []), list) else 0
        elif isinstance(v, list):
            n = len(v)
        else:
            n = 0
        out.append({"name": k, "samples": n})
    return out


def load_encodings():
    """
    Backward-compatible helper for older scripts (e.g. test_thersold.py).
    Returns: {name: [embedding_vectors...]}
    """
    db = load_db()
    try:
        import numpy as np
    except Exception:
        return {}

    enc = {}
    for name, data in db.items():
        if isinstance(data, dict) and "emb" in data and data.get("emb") is not None:
            try:
                enc[name] = [np.array(data["emb"], dtype=np.float32)]
            except Exception:
                continue
        elif isinstance(data, list) and data:
            # Legacy format: list of embeddings (not images)
            first = data[0]
            if isinstance(first, (list, tuple)) and len(first) > 100:
                try:
                    enc[name] = [np.array(e, dtype=np.float32) for e in data]
                except Exception:
                    pass
    return enc


def show_persons():
    persons = get_persons()
    if not persons:
        print("\n  No persons registered.\n")
        return
    print("\n====== Registered Persons ======")
    for i, p in enumerate(persons, 1):
        print(f"  {i}. {p['name']:<20} {p['samples']} samples")
    print("================================\n")


def delete_all():
    for f in [DB_FILE, LOG_FILE]:
        if os.path.exists(f):
            os.remove(f)
    print("  All data deleted.")


def log_entry(name, score):
    with open(LOG_FILE, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts},{name},{score:.3f}\n")


def mark_verified(name):
    db = load_db()
    if name not in db:
        return
    person = db.get(name)
    if isinstance(person, dict):
        person["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db[name] = person
        save_db(db)


def get_log(limit=20):
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()[-limit:]
        return [{"time": l.split(",")[0],
                 "name": l.split(",")[1],
                 "score": l.split(",")[2].strip()}
                for l in lines if len(l.split(",")) >= 3]
    except Exception:
        return []
