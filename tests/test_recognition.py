"""Recognition tests.

Model-dependent tests are skipped automatically when the ONNX models or the
test face images are not present (run ``python download_models.py`` first).

The live-loop tests at the bottom exercise the exact code path the camera
thread uses (recognition pass, consecutive confirmation, attendance marking)
with a fake pipeline so they run fast and deterministically.
"""

from datetime import datetime

import numpy as np
import pytest

from face_attendance import attendance, db, people
from face_attendance.camera import CameraService
from face_attendance.pipeline import Face
from tests import helpers
from tests.helpers import embeddings_of, random_embedding, single_face


def _init(cfg):
    db.init(cfg.db_path)


# ---------------------------------------------------------------------------
# Embedding behaviour with the real SFace model
# ---------------------------------------------------------------------------

def test_same_person_similarity_well_above_threshold(lena_img, pipeline):
    _init_required()
    embs = embeddings_of(lena_img, pipeline, count=5)
    threshold = 0.50
    for i in range(len(embs)):
        for j in range(len(embs)):
            sim = people_mod_cos(embs[i], embs[j])
            assert sim >= threshold, f"same face sim {sim:.3f} too low"
            assert sim >= 0.75


def test_different_people_similarity_below_threshold(lena_img, messi_img,
                                                     pipeline):
    lena = embeddings_of(lena_img, pipeline, count=3)
    messi = embeddings_of(messi_img, pipeline, count=3)
    worst = 1.0
    for a in lena:
        for b in messi:
            worst = min(worst, people_mod_cos(a, b))
    assert worst < 0.35, f"cross-person sim too high: {worst:.3f}"


def _init_required():
    pass  # fixtures already skip when assets/models are missing


def people_mod_cos(a, b):
    from face_attendance.pipeline import cosine_similarity
    return cosine_similarity(a, b)


# ---------------------------------------------------------------------------
# Classification decision (pure logic, no models needed)
# ---------------------------------------------------------------------------

def test_classify_decision_boundaries(cfg):
    _init(cfg)
    classify = people.classify
    good = {"person_id": 1, "code": "A", "name": "Alice",
            "similarity": 0.60}
    weak = {"person_id": 1, "code": "A", "name": "Alice",
            "similarity": 0.42}
    far = {"person_id": 1, "code": "A", "name": "Alice",
           "similarity": 0.10}
    label, person, sim = classify(good, 0.50, 0.35)
    assert label == "recognized" and person == good
    label, person, _ = classify(weak, 0.50, 0.35)
    assert label == "uncertain" and person is not None
    label, person, _ = classify(far, 0.50, 0.35)
    assert label == "unknown" and person is None
    label, person, _ = classify(None, 0.50, 0.35)
    assert label == "unknown" and person is None


def test_classify_registered_vs_unknown_with_gallery(cfg):
    _init(cfg)
    emb_a = random_embedding(seed=1)
    emb_b = random_embedding(seed=2)
    people.create_person("A1", "Alice", [emb_a])
    gallery = people.Gallery()
    best = gallery.best_match(emb_a)          # identical -> sim 1.0
    label, person, sim = people.classify(best, 0.5, 0.35)
    assert label == "recognized"
    assert person["code"] == "A1"
    best_b = gallery.best_match(emb_b)         # unrelated random vector
    label, person, _ = people.classify(best_b, 0.5, 0.35)
    assert label == "unknown"


# ---------------------------------------------------------------------------
# Live recognition loop: confirmation + attendance + dedupe
# ---------------------------------------------------------------------------

class FakeFace(Face):
    def __init__(self, x=10, y=10, w=120, h=120):
        lmk = [x + w * 0.3, y + h * 0.3, x + w * 0.7, y + h * 0.3,
               x + w * 0.5, y + h * 0.45, x + w * 0.3, y + h * 0.75,
               x + w * 0.7, y + h * 0.75]
        super().__init__(x, y, w, h, 0.99, lmk)


class FakePipe:
    """Minimal stand-in for FacePipeline used by the recognition pass."""

    def __init__(self, embedding):
        self.embedding = np.asarray(embedding, dtype=np.float32)

    def embed(self, frame, face):
        return self.embedding

    def detect(self, frame):
        return []


def _camera_with_person(cfg, emb, code="EMP-1", name="Test Person"):
    _init(cfg)
    person = people.create_person(code, name, [emb])
    cam = CameraService(cfg)
    return cam, person


def test_attendance_requires_consecutive_confirmations(cfg):
    emb = random_embedding(seed=7)
    cam, person = _camera_with_person(cfg, emb)
    pipe = FakePipe(emb)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    face = FakeFace()

    # One isolated pass must NOT mark attendance.
    cam._recognize_pass(frame, pipe, [face], t=100.0)
    assert len(attendance.records()) == 0

    # After `confirm_frames` consecutive agreeing passes -> marked once.
    confirm = int(db.get_setting("confirm_frames"))
    for i in range(confirm):
        cam._recognize_pass(frame, pipe, [face], t=101.0 + i)
    records = attendance.records()
    assert len(records) == 1
    assert records[0]["name"] == "Test Person"
    assert records[0]["confidence"] >= 0.99

    # Person keeps standing there -> still exactly one record (duplicate guard).
    for i in range(20):
        cam._recognize_pass(frame, pipe, [face], t=200.0 + i)
    assert len(attendance.records()) == 1


def test_unknown_face_never_marks_attendance(cfg):
    registered = random_embedding(seed=1)
    stranger = random_embedding(seed=2)   # unrelated -> low similarity
    cam, _ = _camera_with_person(cfg, registered)
    pipe = FakePipe(stranger)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    face = FakeFace()
    for i in range(12):
        cam._recognize_pass(frame, pipe, [face], t=300.0 + i)
    assert len(attendance.records()) == 0


def test_identity_switch_resets_confirmation(cfg):
    """Alice stands there twice, then Bob briefly, then Alice again: Bob's
    appearance must not contribute to Alice's confirmation chain."""
    emb_alice = random_embedding(seed=1)
    emb_bob = random_embedding(seed=2)
    _init(cfg)
    alice = people.create_person("AL", "Alice", [emb_alice])
    people.create_person("BO", "Bob", [emb_bob])
    cam = CameraService(cfg)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    cam._recognize_pass(frame, FakePipe(emb_alice), [FakeFace()], t=1.0)
    # Bob interrupts the chain (different person in frame).
    cam._recognize_pass(frame, FakePipe(emb_bob), [FakeFace()], t=2.0)
    cam._recognize_pass(frame, FakePipe(emb_alice), [FakeFace()], t=3.0)
    # confirm_frames=2 consecutive Alice passes are needed after the reset.
    cam._recognize_pass(frame, FakePipe(emb_alice), [FakeFace()], t=4.0)
    assert len(attendance.records()) == 1
    assert attendance.records()[0]["name"] == "Alice"


class MultiFacePipe:
    """Fake pipeline that embeds each face using its x coordinate."""

    def __init__(self, by_x):
        self.by_x = by_x

    def embed(self, frame, face):
        return self.by_x[face.x]

    def detect(self, frame):
        return []


def test_two_people_in_frame_marked_independently(cfg):
    """Both people are confirmed and marked; no cross-talk between faces."""
    emb_a = random_embedding(seed=1)
    emb_b = random_embedding(seed=2)
    _init(cfg)
    people.create_person("A", "Alice", [emb_a])
    people.create_person("B", "Bob", [emb_b])
    cam = CameraService(cfg)
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    pipe = MultiFacePipe({20: emb_a, 200: emb_b})
    confirm = int(db.get_setting("confirm_frames"))
    for i in range(confirm + 2):
        cam._recognize_pass(frame, pipe,
                            [FakeFace(x=20), FakeFace(x=200)], t=50.0 + i)
    records = attendance.records()
    names = sorted(r["name"] for r in records)
    assert names == ["Alice", "Bob"]
    # extra passes never add more rows (both already marked today)
    assert len(records) == 2
