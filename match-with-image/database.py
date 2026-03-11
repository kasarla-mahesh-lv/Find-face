import cv2
import numpy as np
import os
import pickle
from datetime import datetime

FACES_FOLDER  = "known_faces"
DATABASE_FILE = "faces_data.pkl"
AUDIT_FILE    = "audit_log.txt"


def setup():
    os.makedirs(FACES_FOLDER, exist_ok=True)


def load_label_map():
    """Load {label_int: name_str} dict from name_map.pkl"""
    if not os.path.exists("name_map.pkl"):
        return {}
    try:
        with open("name_map.pkl", "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            return {int(k): str(v) for k, v in data.items()}

        elif isinstance(data, list):
            converted = {i: str(n) for i, n in enumerate(data)}
            save_label_map(converted)
            print(f"[INFO] Converted old name_map list → dict: {converted}")
            return converted

        return {}
    except Exception as e:
        print(f"[WARNING] load_label_map failed: {e}")
        return {}


def save_label_map(label_to_name: dict):
    """Save {label_int: name_str} dict to name_map.pkl"""
    clean = {int(k): str(v) for k, v in label_to_name.items()}
    with open("name_map.pkl", "wb") as f:
        pickle.dump(clean, f)
    with open("name_map_backup.txt", "w") as f:
        for lbl, name in sorted(clean.items()):
            f.write(f"Label {lbl}: {name}\n")


def save_database(names, faces, emp_ids=None):
    if emp_ids is None:
        emp_ids = [f"EMP{i:03d}" for i in range(len(names))]

    if len(names) == 0 or len(faces) == 0:
        print("[WARNING] Attempting to save empty database!")
        return

    data = {
        "names":        names,
        "faces":        faces,
        "emp_ids":      emp_ids,
        "last_updated": str(datetime.now()),
        "user_count":   len(set(names))
    }

    try:
        with open(DATABASE_FILE, "wb") as f:
            pickle.dump(data, f)
        unique = list(dict.fromkeys(names))   
        print(f"[DATABASE] Saved {len(names)} samples from {len(unique)} user(s): {unique}")
    except Exception as e:
        print(f"[ERROR] Failed to save database: {e}")


def load_database():
    if not os.path.exists(DATABASE_FILE):
        return [], [], []

    try:
        with open(DATABASE_FILE, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            names   = data.get("names",   [])
            faces   = data.get("faces",   [])
            emp_ids = data.get("emp_ids", [])

            min_len = min(len(names), len(faces))
            if min_len == 0:
                return [], [], []

            names   = names[:min_len]
            faces   = faces[:min_len]
            emp_ids = (emp_ids if len(emp_ids) >= min_len
                       else [f"EMP{i:03d}" for i in range(min_len)])
            emp_ids = emp_ids[:min_len]
            return names, faces, emp_ids

        elif isinstance(data, tuple) and len(data) == 2:
            names, faces = data
            emp_ids = [f"EMP{i:03d}" for i in range(len(names))]
            return names, faces, emp_ids

        else:
            print(f"[ERROR] Unknown database format: {type(data)}")
            return [], [], []

    except Exception as e:
        print(f"[ERROR] Cannot load database: {e}")
        return [], [], []


def is_database_empty():
    names, _, _ = load_database()
    return len(names) == 0


def get_user_count():
    names, _, _ = load_database()
    return len(set(names))


def get_unique_names():
    label_map = load_label_map()
    if label_map:
        return [label_map[k] for k in sorted(label_map.keys())]
    names, _, _ = load_database()
    return list(dict.fromkeys(names))


def get_all_users():
    names, _, emp_ids = load_database()
    label_map = load_label_map()
    name_to_label = {v: k for k, v in label_map.items()}

    users = []
    seen  = set()
    for i, name in enumerate(names):
        if name not in seen:
            seen.add(name)
            users.append({
                'name':    name,
                'emp_id':  emp_ids[i] if i < len(emp_ids) else f"EMP{i:03d}",
                'samples': names.count(name),
                'label':   name_to_label.get(name, "?")
            })
    return users


def show_all_users():
    users     = get_all_users()
    label_map = load_label_map()

    if not users:
        print("\n[DATABASE] No users registered yet.")
    else:
        print("\n======= Registered Employees =======")
        for i, u in enumerate(users, 1):
            print(f"  {i}. {u['name']} (ID: {u['emp_id']}) "
                  f"— {u['samples']} samples  [Label: {u['label']}]")
        print(f"\n  Label map: {label_map}")
        print("====================================\n")



def delete_user(name):
    names, faces, emp_ids = load_database()
    label_map = load_label_map()           
    name_to_label = {v: k for k, v in label_map.items()}

    if name not in names:
        print(f"[DATABASE] '{name}' not found.")
        return

    remaining = [
        (n, f, e)
        for n, f, e in zip(names, faces, emp_ids)
        if n != name
    ]

    if remaining:
        new_names, new_faces, new_emp_ids = map(list, zip(*remaining))
    else:
        new_names, new_faces, new_emp_ids = [], [], []


    if name in name_to_label:
        del label_map[name_to_label[name]]

    save_database(new_names, new_faces, new_emp_ids)

    if new_names and label_map:
        
        current_name_to_label = {v: k for k, v in label_map.items()}
        training_faces, training_labels = [], []
        for n, f in zip(new_names, new_faces):
            if n in current_name_to_label:
                training_faces.append(f)
                training_labels.append(current_name_to_label[n])

        if training_faces:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(training_faces, np.array(training_labels, dtype=np.int32))
            recognizer.save("face_model.yml")
            save_label_map(label_map)
            print(f"[DATABASE] Retrained. Remaining label map: {label_map}")
        else:
            for f in ["face_model.yml", "name_map.pkl", "name_map_backup.txt"]:
                if os.path.exists(f): os.remove(f)
    else:
        for f in ["face_model.yml", "name_map.pkl", "name_map_backup.txt"]:
            if os.path.exists(f): os.remove(f)
        print("[DATABASE] No users remaining. Model files deleted.")

    print(f"[DATABASE] '{name}' deleted successfully.")
    log_audit("SYSTEM", "DELETE", True, f"Deleted user {name}")


def delete_all_users():
    try:
        deleted = []
        for fname in [DATABASE_FILE, "face_model.yml", "name_map.pkl",
                      AUDIT_FILE, "name_map_backup.txt"]:
            if os.path.exists(fname):
                os.remove(fname)
                deleted.append(fname)
        if deleted:
            print(f"[DATABASE] Deleted: {', '.join(deleted)}")
            print("[DATABASE] All user data removed successfully!")
        else:
            print("[DATABASE] No database files found.")
        return True
    except Exception as e:
        print(f"[ERROR] Cannot delete database: {e}")
        return False


def log_audit(employee_id, action, success, details=""):
    try:
        with open(AUDIT_FILE, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp},{employee_id},{action},{success},{details}\n")
    except:
        pass


def get_audit_log(limit=100):
    if not os.path.exists(AUDIT_FILE):
        return []
    try:
        with open(AUDIT_FILE, "r") as f:
            lines = f.readlines()
        if len(lines) > limit:
            lines = lines[-limit:]
        logs = []
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 4:
                    logs.append({
                        'timestamp': parts[0],
                        'employee':  parts[1],
                        'action':    parts[2],
                        'success':   parts[3].lower() in ('true', '1'),
                        'details':   parts[4] if len(parts) > 4 else ''
                    })
        return logs
    except Exception as e:
        print(f"Error reading audit log: {e}")
        return []