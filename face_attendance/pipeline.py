"""Face pipeline: detection (YuNet) + face embeddings (SFace).

Both come from OpenCV's DNN module running ONNX models, which keeps the
dependency list tiny (opencv-contrib-python is enough).

Why embeddings?
---------------
The old project trained an LBPH model (Local Binary Pattern histograms) that
compares raw texture histograms of face crops.  It is brittle across lighting,
pose and camera changes and needs full retraining whenever a person is added.

This project instead converts every face into a fixed 128-dimensional
*embedding* vector using SFace.  Two faces belong to the same person when the
cosine similarity of their embeddings is high.  People can be added at any
time without retraining - recognition is just a nearest-neighbour search over
the stored embeddings of registered people.

Models are loaded exactly once (module level) and reused by every request and
by the camera thread.  All model inference is serialized with a lock because
OpenCV DNN nets are not guaranteed thread-safe.
"""

import threading
from typing import Optional

import cv2
import numpy as np

# Scores above this are kept by the YuNet detector (low value = more faces,
# also more false positives).  0.6 is the default used by opencv_zoo samples.
DETECT_SCORE_THRESHOLD = 0.6
EMBEDDING_DIM = 128


class ModelUnavailableError(RuntimeError):
    """Raised when the ONNX model files are missing or cannot be loaded."""


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Normalize an embedding to unit length (float32, flat)."""
    v = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return v
    return v / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two (already normalized) embeddings."""
    return float(np.dot(a.ravel(), b.ravel()))


class Face:
    """One detected face inside a frame."""

    __slots__ = ("x", "y", "w", "h", "score", "landmarks")

    def __init__(self, x, y, w, h, score, landmarks):
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        self.score = float(score)
        self.landmarks = landmarks  # 10 floats: 5 x (x, y)

    @property
    def box(self):
        return (self.x, self.y, self.w, self.h)

    @property
    def width(self):
        return self.w

    def iou(self, other: "Face") -> float:
        """Intersection-over-union with another face box (0..1)."""
        ax1, ay1 = self.x, self.y
        ax2, ay2 = self.x + self.w, self.y + self.h
        bx1, by1 = other.x, other.y
        bx2, by2 = other.x + other.w, other.y + other.h
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union > 0 else 0.0


class FacePipeline:
    """YuNet detector + SFace recognizer, loaded once and reused."""

    def __init__(self, detector_path, recognizer_path):
        if not detector_path or not str(detector_path):
            raise ModelUnavailableError("YuNet detector model path is empty.")
        if not recognizer_path or not str(recognizer_path):
            raise ModelUnavailableError("SFace recognizer model path is empty.")
        detector_path, recognizer_path = str(detector_path), str(recognizer_path)
        self._detector = None
        self._recognizer = None
        self._lock = threading.RLock()
        try:
            self._detector = cv2.FaceDetectorYN.create(
                detector_path, "", (320, 320), DETECT_SCORE_THRESHOLD, 0.3, 5000)
            self._recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
        except cv2.error as exc:  # e.g. file not found / corrupt ONNX
            raise ModelUnavailableError(
                f"Could not load face models: {exc}") from exc

    # -- public API ---------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Face]:
        """Detect every face in a BGR frame."""
        with self._lock:
            h, w = frame.shape[:2]
            self._detector.setInputSize((w, h))
            _, faces = self._detector.detect(frame)
        out = []
        if faces is None:
            return out
        for row in faces:
            x, y, bw, bh = row[:4]
            lmk = row[4:14]
            out.append(Face(x, y, bw, bh, row[14], lmk))
        return out

    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray:
        """Return the L2-normalized 128-d embedding for one face."""
        with self._lock:
            aligned = self._recognizer.alignCrop(frame, self._face_row(face))
            feat = self._recognizer.feature(aligned)
        return l2_normalize(np.asarray(feat, dtype=np.float32))

    def embed_and_aligned(self, frame: np.ndarray, face: Face):
        """Embedding plus the aligned 112x112 crop (used for quality checks)."""
        with self._lock:
            aligned = self._recognizer.alignCrop(frame, self._face_row(face))
            feat = self._recognizer.feature(aligned)
        return l2_normalize(np.asarray(feat, dtype=np.float32)), aligned

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _face_row(face: Face) -> np.ndarray:
        """Row format expected by alignCrop: rect + 5 landmarks (float32)."""
        row = np.zeros((1, 14), dtype=np.float32)
        row[0, :4] = [face.x, face.y, face.w, face.h]
        row[0, 4:14] = face.landmarks
        return row


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_pipeline: Optional[FacePipeline] = None
_pipeline_paths: tuple = ()
_pipeline_lock = threading.Lock()


def get_pipeline(detector_path, recognizer_path) -> FacePipeline:
    """Return the shared pipeline, creating it once for the given model paths.

    If the models were missing on an earlier attempt (and are later downloaded)
    a new attempt is made automatically, so a running app picks them up after
    ``python download_models.py`` without a restart of the model code.
    """
    global _pipeline, _pipeline_paths
    key = (str(detector_path), str(recognizer_path))
    with _pipeline_lock:
        if _pipeline is None or _pipeline_paths != key:
            _pipeline = FacePipeline(*key)
            _pipeline_paths = key
        return _pipeline


def reset_pipeline() -> None:
    """Drop the cached pipeline (used by tests and model re-downloads)."""
    global _pipeline, _pipeline_paths
    with _pipeline_lock:
        _pipeline = None
        _pipeline_paths = ()
