"""Standalone Live Face Recognition & Attendance CLI.

Runs a desktop window (cv2.imshow) using YuNet for multi-face detection,
SFace for 128-d embeddings, and in-memory vectorized Cosine Similarity
against registered identities in faceattend.db.

Features:
- Instant matching without retraining
- Visual feedback:
    * GREEN: Confident match (>= 0.50) + Attendance Marked
    * AMBER: Uncertain match (0.35 - 0.50)
    * RED:   Unknown face (< 0.35)
- Multi-frame confirmation window (default: 2 consecutive frames)
- Concurrency-safe atomic attendance recording in SQLite
- Real-time FPS counter

Usage:
    python 03_recognize.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from face_attendance import attendance as attendance_mod
from face_attendance import db, people
from face_attendance.config import Config
from face_attendance.pipeline import FacePipeline

# Color definitions (BGR)
GREEN = (60, 200, 120)
AMBER = (40, 170, 240)
RED = (70, 70, 230)
WHITE = (255, 255, 255)
DARK = (20, 20, 20)


def draw_label(img: np.ndarray, text: str, pt: tuple[int, int], color: tuple[int, int, int]):
    """Draw a clean filled rectangle background behind text."""
    x, y = pt
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    y_top = max(0, y - th - 10)
    cv2.rectangle(img, (x, y_top), (x + tw + 10, y_top + th + 10), color, -1)
    cv2.putText(img, text, (x + 5, y_top + th + 4), font, scale, WHITE, thick, cv2.LINE_AA)


def main():
    cfg = Config()
    db.init(str(cfg.db_path))

    if not cfg.models_available():
        print("\n[ERROR] ONNX face models missing in models/ folder!")
        print("Run: python download_models.py\n")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("      LIVE AI FACE RECOGNITION & ATTENDANCE (DESKTOP)")
    print("=" * 60)

    # Load thresholds from database
    sim_threshold = float(db.get_setting("similarity_threshold", 0.50))
    uncertain_threshold = float(db.get_setting("uncertain_threshold", 0.35))
    confirm_frames = int(db.get_setting("confirm_frames", 2))
    check_interval = float(db.get_setting("recognition_interval_seconds", 0.7))
    cam_idx = int(db.get_setting("camera_index", 0))

    gallery = people.get_gallery()
    print(f"Loaded Gallery: {len(gallery)} face embeddings registered.")
    print(f"Match Threshold: {sim_threshold:.2f} | Confirm Frames: {confirm_frames}")

    pipeline = FacePipeline(cfg.detector_model, cfg.recognizer_model)

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_idx, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"\n[ERROR] Could not open camera {cam_idx}. Check webcam connection.\n")
        sys.exit(1)

    window_name = "AI Face Recognition & Attendance (Press 'Q' to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # State tracking
    consecutive_counts: dict[int, int] = {}  # person_db_id -> consecutive confirmations
    last_recognition_time = 0.0
    cached_faces: list[dict] = []
    fps_history: list[float] = []
    prev_time = time.time()
    toast_message = ""
    toast_expire = 0.0

    print("\nStarting video stream. Press [Q] or [ESC] to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps_history.append(1.0 / dt)
                if len(fps_history) > 20:
                    fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

            h, w = frame.shape[:2]
            display = frame.copy()

            # Time to run full recognition pass?
            run_recognition = (curr_time - last_recognition_time) >= check_interval

            if run_recognition:
                last_recognition_time = curr_time
                detected_faces = pipeline.detect(frame)
                cached_faces = []
                seen_person_ids = set()

                for f in detected_faces:
                    emb = pipeline.embed(frame, f)
                    best = gallery.best_match(emb)
                    decision, person, sim = people.classify(best, sim_threshold, uncertain_threshold)

                    face_info = {
                        "box": (f.x, f.y, f.w, f.h),
                        "decision": decision,
                        "person": person,
                        "similarity": sim,
                    }

                    if decision == "recognized" and person:
                        pid = person["person_id"]
                        seen_person_ids.add(pid)
                        consecutive_counts[pid] = consecutive_counts.get(pid, 0) + 1

                        # Threshold reached? Trigger attendance!
                        if consecutive_counts[pid] >= confirm_frames:
                            res = attendance_mod.mark_present(
                                person_db_id=pid,
                                code=person["code"],
                                name=person["name"],
                                confidence=sim,
                            )
                            if res.get("created"):
                                toast_message = f"ATTENDANCE MARKED: {person['name']}"
                                toast_expire = curr_time + 3.0
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] MARKED ATTENDANCE: {person['name']} ({sim * 100:.1f}%)")
                            else:
                                if toast_expire < curr_time:
                                    toast_message = f"ALREADY MARKED: {person['name']}"
                                    toast_expire = curr_time + 1.5
                    cached_faces.append(face_info)

                # Reset counts for people no longer in frame
                for pid in list(consecutive_counts.keys()):
                    if pid not in seen_person_ids:
                        del consecutive_counts[pid]

            # Render cached recognition annotations
            for item in cached_faces:
                bx, by, bw, bh = item["box"]
                decision = item["decision"]
                person = item["person"]
                sim = item["similarity"]

                if decision == "recognized":
                    color = GREEN
                    label = f"{person['name']} {sim * 100:.0f}%"
                elif decision == "uncertain":
                    color = AMBER
                    label = f"{person['name']}? {sim * 100:.0f}%"
                else:
                    color = RED
                    label = "Unknown"

                cv2.rectangle(display, (bx, by), (bx + bw, by + bh), color, 2)
                draw_label(display, label, (bx, by), color)

            # Draw banner / HUD
            cv2.putText(display, f"FPS: {avg_fps:.1f}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(display, f"Registered: {len(gallery)}", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Draw Toast announcement
            if toast_message and curr_time < toast_expire:
                t_w, t_h = 480, 45
                tx1 = (w - t_w) // 2
                ty1 = 15
                cv2.rectangle(display, (tx1, ty1), (tx1 + t_w, ty1 + t_h), GREEN, -1)
                cv2.putText(display, toast_message, (tx1 + 20, ty1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, DARK, 2, cv2.LINE_AA)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nRecognition closed.\n")


if __name__ == "__main__":
    main()
