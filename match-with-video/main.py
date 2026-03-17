import os
import traceback
try:
    import msvcrt
except Exception:
    msvcrt = None
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    from register import register, test_webcam
    from cctv import cctv
    from database import show_persons, delete_all, is_empty, get_persons, get_log
except Exception as e:
    print(f"\nIMPORT ERROR: {e}")
    traceback.print_exc()
    input("\nPress Enter to exit...")
    exit()

def read_choice(prompt, valid):
    """Read a single-key menu choice on Windows terminals; fallback to input()."""
    valid = set(valid)
    if msvcrt is None:
        while True:
            value = input(prompt).strip()
            if value in valid:
                return value
            print(f"  Enter one of: {', '.join(sorted(valid))}")

    print(prompt, end="", flush=True)
    while True:
        ch = msvcrt.getwch()
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch in valid:
            print(ch)
            return ch


def main():
    sensitivity_mode = "lenient"
    while True:
        try:
            print("\n" + "="*50)
            print("   CCTV FACE RECOGNITION")
            print("="*50)
            persons = get_persons()
            if not persons:
                print("   No persons registered")
            else:
                print(f"   {len(persons)} person(s): "
                      f"{[p['name'] for p in persons]}")
            print(f"   Detection mode: {sensitivity_mode}")
            print("="*50)
            print("  1. Register Person")
            print("  2. Start CCTV")
            print("  3. View Persons")
            print("  4. View Entry Log")
            print("  5. Delete ALL Data")
            print("  6. Exit")
            print("  7. Set Detection Mode")
            print("  8. Test Webcam (preview only)")
            print("="*50)

            c = read_choice("  Choice (1-8): ", "12345678")

            if c == "1":
                register()
            elif c == "2":
                if is_empty():
                    print("  Register persons first.")
                else:
                    status = cctv(sensitivity_mode)
                    if status == "mediapipe_missing":
                        print("  CCTV did not start: MediaPipe is missing.")
                        print("  Run: python -m pip install mediapipe")
                    elif status == "deepface_unavailable":
                        print("  CCTV did not start: DeepFace/TensorFlow environment error.")
                        print("  Reinstall with:")
                        print("    python -m pip install --upgrade pip")
                        print("    python -m pip install deepface tensorflow")
                    elif status == "camera_not_found":
                        print("  CCTV did not start: camera not found or busy.")
                    elif status == "video_not_found":
                        print("  CCTV did not start: video file not found.")
                    elif status == "refs_failed":
                        print("  CCTV did not start: reference embeddings could not be built.")
            elif c == "3":
                show_persons()
            elif c == "4":
                logs = get_log(20)
                if not logs:
                    print("  No entries yet.")
                else:
                    print("\n===== Entry Log =====")
                    for e in logs:
                        print(f"  {e['time']}  {e['name']}  {e['score']}")
                    print("=====================\n")
            elif c == "5":
                if input("  Type DELETE ALL: ").strip() == "DELETE ALL":
                    delete_all()
            elif c == "6":
                print("  Goodbye!\n")
                break
            elif c == "7":
                print("  1. strict  (less mismatch, may miss faces)")
                print("  2. normal  (balanced)")
                print("  3. lenient (more faces, may mismatch)")
                m = read_choice("  Mode (1-3): ", "123")
                if m == "1":
                    sensitivity_mode = "strict"
                elif m == "2":
                    sensitivity_mode = "normal"
                elif m == "3":
                    sensitivity_mode = "lenient"
                else:
                    print("  Invalid mode, unchanged.")
            elif c == "8":
                test_webcam()
            else:
                print("  Enter 1-8.")
        except Exception as e:
            print(f"\nERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
