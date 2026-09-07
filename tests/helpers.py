"""Helpers shared by the tests: deterministic image variants, embeddings."""

import numpy as np

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def variants(img, count=5):
    """Deterministic variations of an image (rotation / brightness / scale).

    These approximate small webcam changes (pose wobble, exposure drift).
    """
    h, w = img.shape[:2]
    outs = [img.copy()]
    steps = [
        (-5, 1.00), (5, 1.00),
        (0, 0.88), (0, 1.12),
        (-3, 1.06), (3, 0.94),
    ]
    for deg, bright in steps[:count - 1]:
        m = cv_rotation_matrix(w, h, deg)
        out = cv_warp(img, m, w, h)
        out = cv_scale_brightness(out, bright)
        outs.append(out)
    return outs


def cv_rotation_matrix(w, h, deg):
    import cv2
    import math
    cx, cy = w / 2.0, h / 2.0
    a = math.radians(deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    return cv2.getRotationMatrix2D((cx, cy), deg, 1.0)  # noqa: F841 (kept for clarity)


def cv_warp(img, matrix, w, h):
    import cv2
    return cv2.warpAffine(img, matrix, (w, h),
                          borderMode=cv2.BORDER_REFLECT)


def cv_scale_brightness(img, factor):
    import cv2
    import numpy as np
    lut = np.clip(np.arange(256, dtype=np.float32) * factor, 0, 255)
    return cv2.LUT(img, lut.astype(np.uint8))


def single_face(img, pipe):
    """First detected face in the image, or None."""
    faces = pipe.detect(img)
    return faces[0] if faces else None


def embeddings_of(img, pipe, count=5):
    """Embeddings of the person in *img* using deterministic variants."""
    out = []
    for v in variants(img, count):
        face = single_face(v, pipe)
        if face is not None:
            out.append(pipe.embed(v, face))
    if not out:
        raise AssertionError("No face found in test image - cannot embed.")
    return out


# ---------------------------------------------------------------------------
# Synthetic embeddings (for logic tests that must not depend on the models)
# ---------------------------------------------------------------------------

def random_embedding(dim=128, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)
