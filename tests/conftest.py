"""Shared pytest fixtures.

Every test gets a fresh temporary database so tests never touch real data and
run independently of each other.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

MODEL_FILES = [
    MODELS_DIR / "face_detection_yunet_2023mar.onnx",
    MODELS_DIR / "face_recognition_sface_2021dec.onnx",
]
ASSET_FILES = [
    ASSETS_DIR / "lena.jpg",
    ASSETS_DIR / "messi5.jpg",
]


def models_ready() -> bool:
    return all(p.exists() for p in MODEL_FILES)


def assets_ready() -> bool:
    return all(p.exists() for p in ASSET_FILES)


@pytest.fixture
def cfg(tmp_path):
    """Config pointing at a temporary database; camera auto-start disabled."""
    from face_attendance.config import Config
    c = Config()
    c.data_dir = tmp_path
    c.db_path = tmp_path / "test.db"
    c.autostart_camera = False
    return c


@pytest.fixture
def app(cfg):
    """A Flask app bound to the temporary database (camera not started)."""
    from face_attendance.web import create_app
    application = create_app(cfg)
    yield application
    application.extensions["camera"].stop()


@pytest.fixture
def pipeline():
    """The real YuNet+SFace pipeline (skipped when models are missing)."""
    if not models_ready():
        pytest.skip("ONNX models not downloaded - run download_models.py")
    from face_attendance.pipeline import FacePipeline
    return FacePipeline(MODEL_FILES[0], MODEL_FILES[1])


@pytest.fixture
def lena_img():
    if not assets_ready():
        pytest.skip("Test face images missing")
    import cv2
    return cv2.imread(str(ASSETS_DIR / "lena.jpg"))


@pytest.fixture
def messi_img():
    if not assets_ready():
        pytest.skip("Test face images missing")
    import cv2
    return cv2.imread(str(ASSETS_DIR / "messi5.jpg"))
