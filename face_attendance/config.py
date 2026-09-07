"""Configuration for the attendance application.

Every value has a sensible default and can be overridden with an environment
variable (or a ``.env`` file in the project root).  Runtime *settings* that the
user may want to tweak from the Settings page (thresholds, camera index, ...)
live in the SQLite ``settings`` table instead - see ``SETTINGS_SCHEMA`` below.
"""

import os
from pathlib import Path

try:  # optional dependency - only needed if you want a .env file
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Settings schema (key -> metadata).  Values are stored as text in the DB and
# are converted to the correct Python type on read/write.
# ---------------------------------------------------------------------------
SETTINGS_SCHEMA = {
    # Camera
    "camera_index":            {"type": "int",   "default": 0, "min": 0, "max": 9},
    "camera_width":            {"type": "int",   "default": 640, "min": 320, "max": 1280},
    "camera_height":           {"type": "int",   "default": 480, "min": 240, "max": 720},
    "process_fps":             {"type": "int",   "default": 10, "min": 2, "max": 30},
    # Recognition behaviour
    "recognition_interval_seconds": {"type": "float", "default": 0.7,
                                     "min": 0.15, "max": 5.0},
    "similarity_threshold":    {"type": "float", "default": 0.50, "min": 0.0, "max": 1.0},
    "uncertain_threshold":     {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0},
    "confirm_frames":          {"type": "int",   "default": 2, "min": 1, "max": 10},
    # Attendance rules
    "attendance_mode":         {"type": "str",   "default": "once_per_day",
                                "choices": ["once_per_day", "cooldown"]},
    "cooldown_minutes":        {"type": "int",   "default": 10, "min": 1, "max": 1440},
    # Registration quality rules
    "registration_samples":    {"type": "int",   "default": 5, "min": 3, "max": 10},
    "min_face_width":          {"type": "int",   "default": 90, "min": 50, "max": 300},
    "min_sharpness":           {"type": "float", "default": 30.0, "min": 1.0, "max": 500.0},
    "max_embeddings_per_person": {"type": "int", "default": 24, "min": 1, "max": 100},
    # Activity feed coalescing (avoid flooding the log with identical events)
    "activity_log_cooldown_seconds": {"type": "float", "default": 8.0,
                                      "min": 0.0, "max": 300.0},
}


class Config:
    """Static configuration resolved once at startup (env + .env)."""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.name = "AI Face Attendance"

        # Where runtime data lives (DB created here on first start)
        data_dir = Path(_env("FA_DATA_DIR", str(PROJECT_ROOT / "data")))
        self.data_dir = data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(_env("FA_DB_PATH", str(data_dir / "faceattend.db")))

        # Model files (YuNet face detector + SFace recognizer)
        self.models_dir = Path(_env("FA_MODELS_DIR", str(PROJECT_ROOT / "models")))
        self.detector_model = self.models_dir / _env(
            "FA_DETECTOR_MODEL", "face_detection_yunet_2023mar.onnx")
        self.recognizer_model = self.models_dir / _env(
            "FA_RECOGNIZER_MODEL", "face_recognition_sface_2021dec.onnx")

        # Server
        self.host = _env("FA_HOST", "127.0.0.1")
        self.port = int(_env("FA_PORT", "8000"))
        self.debug = _env("FA_DEBUG", "0") == "1"
        self.secret_key = _env("FA_SECRET_KEY", "dev-only-change-me")

        # Camera backend: auto | dshow | msmf | v4l2 (Windows prefers dshow)
        self.camera_backend = _env("FA_CAMERA_BACKEND", "auto")

        # Start the camera thread as soon as the app starts
        self.autostart_camera = _env("FA_AUTOSTART_CAMERA", "1") == "1"

    def models_available(self) -> bool:
        """Both ONNX files exist on disk (fast check used all over the app)."""
        return self.detector_model.exists() and self.recognizer_model.exists()


def typed_defaults() -> dict:
    """Current defaults for every runtime setting (typed)."""
    out = {}
    for key, meta in SETTINGS_SCHEMA.items():
        out[key] = meta["default"]
    return out
