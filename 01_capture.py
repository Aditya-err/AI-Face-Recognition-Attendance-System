"""Standalone Face Capture & Registration CLI.

Uses YuNet (OpenCV DNN) for multi-angle face detection and SFace for 128-d
facial embedding extraction. Quality gates verify face size and sharpness
before saving embeddings into SQLite (faceattend.db).

Usage:
    python 01_capture.py
    python 01_capture.py --name "Alex Doe" --id "EMP001"
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from face_attendance import db, people
from face_attendance.config import Config
from face_attendance.pipeline import FacePipeline, l2_normalize


def calculate_sharpness(gray_crop: np.ndarray) -> float:
    """Variance of the Laplacian indicates image focus/sharpness."""
    if gray_crop is None or gray_crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def main():
    parser = argparse.ArgumentParser(description="Standalone Face Capture & Enrollment")
    parser.add_argument("--name", type=str, default=None, help="Full name of person")
    parser.add_argument("--id", type=str, default=None, help="Unique Person ID / Code")
    parser.add_argument("--samples", type=int, default=5, help="Number of face samples to capture (default: 5)")
    args = parser.parse_args()

    cfg = Config()
    db.init(str(cfg.db_path))

    if not cfg.models_available():
        print("\n[ERROR] Face models missing in models/ folder!")
        print("Run: python download_models.py\n")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("      FACE REGISTRATION & ENROLLMENT (YUNET + SFACE)")
    print("=" * 60)

    name = args.name
    while not name or not name.strip():
        name = input("\nEnter Full Name (e.g. Alex Doe): ").strip()

    code = args.id
    while not code or not code.strip():
        code = input("Enter Person ID (e.g. EMP001): ").strip()

    # Check for existing code
    existing = people.find_by_code(code)
    if existing:
        print(f"\n[!] Notice: Person ID '{code}' is already registered for '{existing['name']}'.")
        choice = input("Do you want to add more samples to this person? [y/N]: ").strip().lower()
        if choice != "y":
            print("Aborted.")
            sys.exit(0)
        re_enroll = True
        person_db_id = existing["id"]
    else:
        re_enroll = False
        person_db_id = None

    print(f"\nInitializing YuNet detector and SFace recognizer...")
    pipeline = FacePipeline(cfg.detector_model, cfg.recognizer_model)

    cam_idx = int(db.get_setting("camera_index", 0))
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_idx, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"\n[ERROR] Could not open camera {cam_idx}. Check webcam connection.\n")
        sys.exit(1)

    target_samples = max(3, min(args.samples, 10))
    captured_embeddings: list[np.ndarray] = []
    last_capture_time = 0.0

    print("\n------------------------------------------------------------")
    print(f" Starting enrollment for: {name} (ID: {code})")
    print(f" Goal: Capture {target_samples} high-quality face samples.")
    print(" Instructions:")
    print("  - Look directly at the camera.")
    print("  - Press [SPACEBAR] to capture a face sample.")
    print("  - Tilt or turn head slightly between captures.")
    print("  - Press [Q] or [ESC] to cancel.")
    print("------------------------------------------------------------\n")

    min_width = int(db.get_setting("min_face_width", 90))
    min_sharp = float(db.get_setting("min_sharpness", 30.0))

    window_name = f"Face Enrollment - {name} ({len(captured_embeddings)}/{target_samples})"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    feedback_msg = "Press [SPACE] to capture sample"
    feedback_color = (0, 255, 0)

    try:
        while len(captured_embeddings) < target_samples:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            display = frame.copy()
            h, w = frame.shape[:2]

            # Detect faces with YuNet
            faces = pipeline.detect(frame)
            current_face = None

            if len(faces) == 0:
                cv2.putText(display, "No face detected - Move closer", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif len(faces) > 1:
                cv2.putText(display, "Multiple faces detected - Only 1 person allowed", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                for f in faces:
                    cv2.rectangle(display, (f.x, f.y), (f.x + f.w, f.y + f.h), (0, 0, 255), 2)
            else:
                current_face = faces[0]
                box_color = (0, 255, 0)
                # Check quality
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fx1 = max(0, current_face.x)
                fy1 = max(0, current_face.y)
                fx2 = min(w, current_face.x + current_face.w)
                fy2 = min(h, current_face.y + current_face.h)
                crop = gray[fy1:fy2, fx1:fx2]
                sharpness = calculate_sharpness(crop)

                status_text = "READY TO CAPTURE"
                if current_face.w < min_width:
                    box_color = (0, 165, 255)
                    status_text = f"Too far! Move closer ({current_face.w}px < {min_width}px)"
                elif sharpness < min_sharp:
                    box_color = (0, 165, 255)
                    status_text = f"Hold still! (Blurry: {sharpness:.0f} < {min_sharp:.0f})"

                cv2.rectangle(display, (current_face.x, current_face.y),
                              (current_face.x + current_face.w, current_face.y + current_face.h),
                              box_color, 2)
                cv2.putText(display, status_text, (current_face.x, max(25, current_face.y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # Header info
            progress_text = f"Samples: {len(captured_embeddings)} / {target_samples}"
            cv2.putText(display, progress_text, (20, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(display, "[SPACE]: Capture  |  [Q]: Quit", (w - 280, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                print("\nEnrollment canceled by user.")
                return

            if key == 32:  # Spacebar
                now = time.time()
                if now - last_capture_time < 0.3:
                    continue
                last_capture_time = now

                if len(faces) == 0:
                    print("  [!] No face detected. Please position yourself in frame.")
                    continue
                if len(faces) > 1:
                    print("  [!] Multiple faces detected. Exactly one face must be visible.")
                    continue

                if current_face.w < min_width:
                    print(f"  [!] Face too small ({current_face.w}px). Minimum required is {min_width}px.")
                    continue

                if sharpness < min_sharp:
                    print(f"  [!] Face is blurry ({sharpness:.1f}). Hold still.")
                    continue

                emb = pipeline.embed(frame, current_face)
                captured_embeddings.append(l2_normalize(emb))
                print(f"  [+] Captured sample {len(captured_embeddings)}/{target_samples} (sharpness={sharpness:.1f}, width={current_face.w}px)")
                cv2.setWindowTitle(window_name, f"Face Enrollment - {name} ({len(captured_embeddings)}/{target_samples})")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(captured_embeddings) >= target_samples:
        print("\nSaving face embeddings to database...")
        try:
            if re_enroll:
                max_total = int(db.get_setting("max_embeddings_per_person", 24))
                people.add_embeddings(person_db_id, captured_embeddings, max_total)
                print(f"\n[SUCCESS] Successfully added {len(captured_embeddings)} new samples for '{name}' (ID: {code})!")
            else:
                p = people.create_person(code, name, captured_embeddings)
                print(f"\n[SUCCESS] Enrolled '{name}' (ID: {code}) successfully with {len(captured_embeddings)} samples!")
            print("Zero retraining needed - user is recognized immediately in live recognition.")
            print("Run 'python 03_recognize.py' or 'python run.py' to test.\n")
        except Exception as exc:
            print(f"\n[ERROR] Failed to save enrollment: {exc}\n")


if __name__ == "__main__":
    main()
