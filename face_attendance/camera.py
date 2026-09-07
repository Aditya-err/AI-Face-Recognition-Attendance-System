"""Camera service: capture, live recognition and registration sessions.

Threading model
---------------
A single daemon thread reads the webcam and, for each frame that the CPU
budget allows, runs face detection and renders an annotated JPEG that the web
browser consumes as an MJPEG stream.  Embedding extraction + recognition is
deliberately *gated*:

- it only runs when at least one face has moved significantly since the last
  full recognition pass, or when ``recognition_interval_seconds`` elapsed, so
  a static face costs one recognition pass per interval instead of per frame;

- attendance is only marked after ``confirm_frames`` consecutive passes
  agree on the same registered person - a single lucky frame can never mark
  attendance.

Registration (used by the Register page) is a *session*: the page asks the
camera thread to capture one good sample at a time, and samples accumulate
until the session is finished or cancelled.
"""

import logging
import os
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from . import attendance as attendance_mod
from . import config as config_module
from . import db
from . import people as people_mod
from .pipeline import (Face, FacePipeline, ModelUnavailableError,
                       get_pipeline)

log = logging.getLogger("camera")

GREEN = (60, 200, 120)    # BGR
AMBER = (40, 170, 240)
RED = (70, 70, 230)
SLATE = (150, 160, 180)
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)


class CameraService:
    """Owns the webcam and produces live annotated JPEG frames."""

    def __init__(self, cfg: config_module.Config):
        self.cfg = cfg
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Camera / frame state
        self.state = "stopped"          # stopped|starting|running|error
        self.error: Optional[str] = None
        self.latest_raw: Optional[np.ndarray] = None
        self.latest_jpeg: Optional[bytes] = None
        self._raw_seq = 0
        self._frame_seq = 0
        self.fps = 0.0
        self.resolution = (0, 0)

        # Live recognition state
        self._pipeline: Optional[FacePipeline] = None
        self._last_full_time = 0.0
        self._cached: list[dict] = []      # results of last full pass
        self._chain: dict[int, int] = {}   # person_db_id -> consecutive passes
        self._absent_since: Optional[float] = None
        self._flash: Optional[dict] = None
        self._proc_last = 0.0

        # Registration session state (used by the Register page)
        self._session: Optional[dict] = None
        self._last_capture_raw_seq = -1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="camera",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()

    def restart(self) -> None:
        """Close the camera and reopen it (picks up new index/models)."""
        self.stop()
        time.sleep(0.4)
        self.start()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            # 1. Make sure the face models are loadable before touching camera.
            pipe, pipe_error = self._load_pipeline()
            if pipe is None:
                self._set_state("error", pipe_error)
                self._sleep_interruptible(3.0)
                continue
            self._pipeline = pipe

            # 2. Open the camera (with retries on failure).
            cap = self._open_camera()
            if cap is None:
                self._set_state("error",
                                "Could not open camera. Check that it is not "
                                "in use by another app and that the camera "
                                "index is correct.")
                self._sleep_interruptible(2.5)
                continue

            self._set_state("running", None)
            read_failures = 0
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    read_failures += 1
                    if read_failures >= 5:
                        break  # reopen the camera
                    self._sleep_interruptible(0.05)
                    continue
                read_failures = 0
                self._handle_frame(frame)
            cap.release()
            if not self._stop_event.is_set():
                self._set_state("error", "Camera disconnected. Reconnecting...")

        self._set_state("stopped", None)

    def _sleep_interruptible(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end and not self._stop_event.is_set():
            time.sleep(min(0.2, end - time.time()))

    def _load_pipeline(self):
        try:
            return get_pipeline(self.cfg.detector_model,
                                self.cfg.recognizer_model), None
        except ModelUnavailableError as exc:
            return None, str(exc)

    def _open_camera(self):
        index = int(db.get_setting("camera_index"))
        width = int(db.get_setting("camera_width"))
        height = int(db.get_setting("camera_height"))
        self.resolution = (width, height)

        backends = []
        backend_cfg = self.cfg.camera_backend
        if backend_cfg == "dshow":
            backends = [cv2.CAP_DSHOW]
        elif backend_cfg == "msmf":
            backends = [cv2.CAP_MSMF]
        elif backend_cfg == "v4l2":
            backends = [cv2.CAP_V4L2]
        else:
            if os.name == "nt":
                backends = [cv2.CAP_DSHOW, 0]  # 0 = OpenCV default backend
            else:
                backends = [0]

        tried = []
        for backend in backends:
            try:
                if backend == 0:
                    cap = cv2.VideoCapture(index)
                else:
                    cap = cv2.VideoCapture(index, backend)
            except cv2.error:
                cap = None
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                return cap
            if cap is not None:
                cap.release()
            tried.append(str(backend))
        log.warning("Camera open failed, tried backends: %s", tried)
        return None

    # ------------------------------------------------------------------
    # Per-frame handling
    # ------------------------------------------------------------------

    def _handle_frame(self, frame: np.ndarray) -> None:
        """Store the raw frame and, when the CPU budget allows, annotate it."""
        if frame.shape[1] != self.resolution[0] or \
           frame.shape[0] != self.resolution[1]:
            frame = cv2.resize(frame, self.resolution)

        with self._lock:
            self.latest_raw = frame.copy()
            self._raw_seq += 1

        # Throttle processing to process_fps.
        now = time.time()
        min_interval = 1.0 / max(2, int(db.get_setting("process_fps")))
        if now - self._proc_last < min_interval:
            return

        try:
            pipe = self._pipeline
            if pipe is None:
                return
            if self._session is not None:
                annotated = self._annotate_register(frame, pipe)
            else:
                annotated = self._annotate_live(frame, pipe)

            ok, buf = cv2.imencode(".jpg", annotated,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._cond:
                    self.latest_jpeg = buf.tobytes()
                    self._frame_seq += 1
                    self._cond.notify_all()
            self._proc_last = now
            if self._proc_last:
                self.fps = 0.85 * self.fps + 0.15 * (
                    1.0 / max(1e-3, now - self._proc_last))
        except ModelUnavailableError:
            self._set_state("error", "Face model files are missing or broken.")
        except Exception:  # never let the camera thread die silently
            log.exception("Error while processing frame")
            self._set_state("error", "Internal processing error - see log.")

    # ------------------------------------------------------------------
    # Live recognition
    # ------------------------------------------------------------------

    def _annotate_live(self, frame: np.ndarray,
                       pipe: FacePipeline) -> np.ndarray:
        t = time.time()
        faces = pipe.detect(frame)

        if not faces:
            if self._absent_since is None:
                self._absent_since = t
            elif t - self._absent_since > 1.0:
                self._cached = []
                self._chain.clear()
            self._draw_flash(frame)
            self._draw_status_bar(frame, "LIVE RECOGNITION", [])
            return frame  # clean frame, nothing else to annotate

        self._absent_since = None
        recompute = self._should_recompute(faces, t)
        if recompute:
            results = self._recognize_pass(frame, pipe, faces, t)
            self._cached = results
            self._last_full_time = t

        # Draw current faces using the freshest cached results (IoU-matched).
        used = set()
        for f in faces:
            row = self._best_cached(f, used)
            if row is None:
                self._draw_face(frame, f, SLATE, "detecting...", None)
            else:
                self._draw_face(frame, f, row["color"], row["display"],
                                row.get("person"))
        self._draw_flash(frame)
        self._draw_status_bar(frame, "LIVE RECOGNITION", faces)
        return frame

    def _should_recompute(self, faces: list[Face], t: float) -> bool:
        interval = float(db.get_setting("recognition_interval_seconds"))
        if t - self._last_full_time >= interval:
            return True
        if not self._cached:
            return True
        for f in faces:
            matched = any(
                f.iou(c["face"]) >= 0.45 for c in self._cached)
            if not matched:
                return True
        return False

    def _recognize_pass(self, frame, pipe, faces, t) -> list[dict]:
        sim_thr = float(db.get_setting("similarity_threshold"))
        unc_thr = float(db.get_setting("uncertain_threshold"))
        confirm = int(db.get_setting("confirm_frames"))

        gallery = people_mod.get_gallery()
        rows = []
        recognized_people = set()
        for f in faces:
            emb = pipe.embed(frame, f)
            best = gallery.best_match(emb)
            label, person, sim = people_mod.classify(best, sim_thr, unc_thr)
            if label == "recognized":
                recognized_people.add(person["person_id"])
            if label == "recognized":
                row = {"face": f, "person": person, "display":
                       f"{person['name']} {int(round(sim * 100))}%",
                       "color": GREEN, "label": label}
            elif label == "uncertain":
                row = {"face": f, "person": person, "display":
                       f"{person['name']}? {int(round(sim * 100))}%",
                       "color": AMBER, "label": label}
            else:
                row = {"face": f, "person": None, "display": "Unknown",
                       "color": RED, "label": label}
            rows.append(row)

        # Consecutive-agreement chain -> attendance.
        for pid in list(self._chain):
            if pid not in recognized_people:
                del self._chain[pid]
        for pid in recognized_people:
            self._chain[pid] = self._chain.get(pid, 0) + 1
            if self._chain[pid] >= confirm:
                self._chain[pid] = 0
                person = people_mod.get_person(pid)
                if person is None:
                    continue
                result = attendance_mod.mark_present(
                    pid, person["code"], person["name"],
                    self._best_sim_for(rows, pid))
                kind = "attendance" if result["created"] else "recognized"
                attendance_mod.log_activity(
                    kind, pid, person["name"],
                    self._best_sim_for(rows, pid))
                if result["created"]:
                    self._set_flash(
                        f"ATTENDANCE MARKED - {person['name']} "
                        f"{datetime.now().strftime('%H:%M:%S')}", GREEN)
                else:
                    self._set_flash(result["message"], SLATE)

        # Activity feed (coalesced server side).
        for row in rows:
            person = row.get("person")
            attendance_mod.log_activity(
                row["label"],
                person["person_id"] if person else None,
                person["name"] if person else None,
                person["similarity"] if person else None)
        return rows

    @staticmethod
    def _best_sim_for(rows: list[dict], pid: int) -> float:
        sims = [r["person"]["similarity"] for r in rows
                if r.get("person") and r["person"]["person_id"] == pid]
        return max(sims) if sims else 0.0

    def _best_cached(self, face: Face, used: set) -> Optional[dict]:
        best, best_iou = None, 0.45
        for i, row in enumerate(self._cached):
            if i in used:
                continue
            iou = face.iou(row["face"])
            if iou >= best_iou:
                best_iou, best = iou, (i, row)
        if best:
            used.add(best[0])
            return best[1]
        return None

    # ------------------------------------------------------------------
    # Registration sessions
    # ------------------------------------------------------------------

    def begin_registration(self, mode: str, name: str, code: str,
                           person_db_id: Optional[int] = None) -> None:
        """Start (or replace) a registration session."""
        with self._lock:
            self._session = {
                "mode": mode,                 # new | collect (existing person)
                "name": (name or "").strip(),
                "code": (code or "").strip(),
                "person_db_id": person_db_id,
                "samples": [],
                "started_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._last_capture_raw_seq = -1

    def session(self) -> Optional[dict]:
        with self._lock:
            return self._session.copy() if self._session else None

    def _session_locked(self) -> Optional[dict]:
        return self._session

    def pending_samples(self) -> list[np.ndarray]:
        with self._lock:
            return [s["embedding"] for s in self._session["samples"]] \
                if self._session else []

    def cancel_registration(self) -> None:
        with self._lock:
            self._session = None
            self._flash = None

    def _annotate_register(self, frame, pipe) -> np.ndarray:
        # NOTE: detection for *samples* happens in capture_registration_sample
        # on the exact stored raw frame; here we only annotate for preview.
        faces = pipe.detect(frame)
        session = self._session_locked()
        if session is None:
            return frame
        color = GREEN if len(faces) == 1 else (SLATE if faces else RED)
        for f in faces:
            self._draw_face(frame, f, color, "face ok" if len(faces) == 1
                            else f"{len(faces)} faces", None)
        self._draw_register_header(frame, len(faces))
        self._draw_flash(frame, y0=40)
        return frame

    def capture_registration_sample(self) -> dict:
        """Capture one good face sample from the latest camera frame.

        Runs on the request thread but all inference goes through the shared
        (locked) pipeline.  Returns a dict with ``ok`` and a message code:
        captured / no_face / multiple_faces / too_blurry / too_small /
        no_new_frame / no_camera / no_models / no_session.
        """
        session = self.session()
        if session is None:
            return {"ok": False, "code": "no_session",
                    "message": "No active registration session."}
        if self.state != "running" or self.latest_raw is None:
            return {"ok": False, "code": "no_camera",
                    "message": "Camera is not available. Check camera "
                               "permissions or the camera index in Settings."}

        pipe, pipe_error = self._load_pipeline()
        if pipe is None:
            return {"ok": False, "code": "no_models", "message": pipe_error}

        with self._lock:
            frame = self.latest_raw.copy()
            seq = self._raw_seq
        if seq == self._last_capture_raw_seq:
            return {"ok": False, "code": "no_new_frame",
                    "message": "The camera frame has not changed - move "
                               "slightly or wait a moment."}

        faces = pipe.detect(frame)
        if not faces:
            return {"ok": False, "code": "no_face",
                    "message": "No face detected - look at the camera."}
        if len(faces) > 1:
            return {"ok": False, "code": "multiple_faces",
                    "message": "Multiple faces detected - only one person "
                               "should be in the frame."}

        face = faces[0]
        min_w = int(db.get_setting("min_face_width"))
        if face.width < min_w:
            return {"ok": False, "code": "too_small",
                    "message": f"Face is too small ({face.width}px) - move "
                               f"closer. Need at least {min_w}px."}

        emb, aligned = pipe.embed_and_aligned(frame, face)
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        min_sharp = float(db.get_setting("min_sharpness"))
        if sharpness < min_sharp:
            return {"ok": False, "code": "too_blurry",
                    "message": "Image is too blurry - hold still and keep "
                               "the camera in focus."}

        with self._lock:
            if self._session is None:
                return {"ok": False, "code": "no_session",
                        "message": "Registration session ended."}
            self._session["samples"].append({
                "embedding": emb, "sharpness": round(sharpness, 1),
                "face_width": face.width})
            self._last_capture_raw_seq = seq
            n = len(self._session["samples"])

        self._set_flash(f"Sample {n} captured - keep your face in view",
                        GREEN)
        return {"ok": True, "code": "captured", "sample_count": n,
                "quality": {"sharpness": round(sharpness, 1),
                            "face_width": face.width},
                "message": f"Sample {n} captured."}

    def _draw_register_header(self, frame, n_faces: int) -> None:
        session = self._session_locked()
        if session is None:
            return
        n = len(session["samples"])
        target = int(db.get_setting("registration_samples"))
        mode_txt = "RE-ENROL" if session["mode"] == "collect" else "REGISTER"
        label = session["name"] or session["code"]
        header = (f"{mode_txt}: {label}"
                  + (f" ({session['code']})" if session["code"] else "")
                  + f"  |  samples {n}/{target}")
        self._banner(frame, header, SLATE)
        if n_faces == 0 and n < target:
            self._banner(frame, "No face detected - look at the camera",
                         AMBER, y0=40)
        elif n_faces > 1:
            self._banner(frame, "Multiple faces - only one person allowed",
                         RED, y0=40)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_face(self, frame, face: Face, color, display: str,
                   person: Optional[dict]) -> None:
        h, w = frame.shape[:2]
        x1 = max(0, min(face.x, w - 1))
        y1 = max(0, min(face.y, h - 1))
        x2 = max(0, min(face.x + face.w, w - 1))
        y2 = max(0, min(face.y + face.h, h - 1))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = display or ""
        if text:
            scale = 0.5
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                          scale, 1)
            tx = min(x1, max(0, w - tw - 4))
            ty = y1 - 6
            if ty - th - 4 < 0:      # label would leave the top of the frame
                ty = min(y2 + th + 12, h - 4)
            cv2.rectangle(frame, (tx, ty - th - 5),
                          (tx + tw + 8, ty + 4), color, -1)
            cv2.putText(frame, text, (tx + 4, ty - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1,
                        cv2.LINE_AA)
            if person is not None:   # small code line under the label
                sub = person.get("code") or ""
                (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.4, 1)
                cv2.rectangle(frame, (tx, ty + 5),
                              (tx + sw + 8, ty + 5 + sh + 6), BLACK, -1)
                cv2.putText(frame, sub, (tx + 4, ty + sh + 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1,
                            cv2.LINE_AA)

    def _banner(self, frame, text, color, y0: int = 0) -> None:
        h, w = frame.shape[:2]
        bh = 32
        cv2.rectangle(frame, (0, y0), (w, y0 + bh), color, -1)
        cv2.putText(frame, text, (10, y0 + 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, WHITE, 1, cv2.LINE_AA)

    def _draw_status_bar(self, frame, title: str, faces) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        text = (f"{title}  |  {now}  |  {self.fps:.0f} fps  |  "
                f"{len(faces)} face(s)")
        self._banner(frame, text, BLACK)

    def _set_flash(self, text: str, color) -> None:
        self._flash = {"text": text, "color": color,
                       "until": time.time() + 4.0}

    def _draw_flash(self, frame, y0: int = 36) -> None:
        flash = self._flash
        if flash and time.time() < flash["until"]:
            self._banner(frame, flash["text"], flash["color"], y0=y0)

    def _set_state(self, state: str, error: Optional[str]) -> None:
        with self._lock:
            self.state = state
            self.error = error

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def status(self) -> dict:
        session = self.session()
        return {
            "state": self.state,
            "error": self.error,
            "fps": round(self.fps, 1),
            "resolution": list(self.resolution),
            "models_available": self.cfg.models_available(),
            "registration": None if session is None else {
                "mode": session["mode"],
                "name": session["name"],
                "code": session["code"],
                "sample_count": len(session["samples"]),
                "target": int(db.get_setting("registration_samples")),
            },
        }

    def mjpeg_frames(self):
        """Generator yielding a multipart MJPEG stream to one HTTP client."""
        last_seq = -1
        try:
            while not self._stop_event.is_set():
                with self._cond:
                    self._cond.wait_for(
                        lambda: self._frame_seq != last_seq
                        or self._stop_event.is_set(), timeout=0.4)
                    if self._stop_event.is_set():
                        break
                    seq = self._frame_seq
                    data = self.latest_jpeg
                if data is None:
                    data = self._placeholder_jpeg()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
                last_seq = seq
        except GeneratorExit:
            pass

    def _placeholder_jpeg(self) -> bytes:
        """A JPEG that explains the current state when no live frame exists."""
        lines = []
        if self.state == "error":
            lines = [self.error or "Camera error"] + lines
        elif self.state == "starting":
            lines = ["Starting camera..."]
        else:
            lines = ["Waiting for camera..."]
        frame = np.zeros((300, 560, 3), dtype=np.uint8)
        frame[:, :] = (35, 38, 42)
        y = 140
        for line in lines[:4]:
            if not line:
                continue
            cv2.putText(frame, line[:58], (24, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
            y += 26
        ok, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes()
