"""End-to-end tests through the HTTP API.

The camera is emulated by feeding real test images into the CameraService so
the whole registration flow (capture -> quality checks -> duplicate
protection -> store) runs against the real models and a real SQLite file.
"""

from datetime import datetime

import cv2
import numpy as np
import pytest

from face_attendance import attendance, db, people
from tests import helpers
from tests.conftest import models_ready, assets_ready


@pytest.fixture
def cam(app):
    """CameraService in a fake 'running' state with the real pipeline."""
    from tests import conftest
    if not models_ready() or not assets_ready():
        pytest.skip("models/test images missing - run download_models.py")
    from face_attendance.pipeline import FacePipeline
    pipe = FacePipeline(conftest.MODEL_FILES[0], conftest.MODEL_FILES[1])
    camera = app.extensions["camera"]
    camera._pipeline = pipe
    camera._load_pipeline = lambda: (pipe, None)
    camera._set_state("running", None)
    return camera


def _feed_image(camera, img):
    """Make *img* the latest camera frame (as if the webcam delivered it)."""
    with camera._lock:
        camera.latest_raw = img.copy()
        camera._raw_seq += 1


# ---------------------------------------------------------------------------
# Pages and basic API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/", "/recognize", "/register", "/people",
                                  "/records", "/settings"])
def test_pages_render(app, path):
    res = app.test_client().get(path)
    assert res.status_code == 200
    assert b"Face" in res.data or b"face" in res.data


def test_dashboard_api_empty(app):
    res = app.test_client().get("/api/dashboard")
    data = res.get_json()
    assert data["registered"] == 0
    assert data["present_today"] == 0


def test_settings_get_and_update(app):
    client = app.test_client()
    data = client.get("/api/settings").get_json()
    assert data["settings"]["similarity_threshold"] == 0.50
    assert data["models_available"] is True

    res = client.post("/api/settings", json={
        "updates": {"similarity_threshold": 0.55,
                    "confirm_frames": 3}})
    assert res.status_code == 200
    assert db.get_setting("similarity_threshold") == 0.55

    # invalid value
    res = client.post("/api/settings", json={"updates": {"camera_index": -3}})
    assert res.status_code == 400
    assert "errors" in res.get_json()

    # uncertain threshold above match threshold
    res = client.post("/api/settings", json={
        "updates": {"similarity_threshold": 0.4,
                    "uncertain_threshold": 0.6}})
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# Registration through the API (real models + fake live camera)
# ---------------------------------------------------------------------------

def _capture_samples(app, cam, img, n):
    client = app.test_client()
    count = 0
    for variant in helpers.variants(img, n):
        _feed_image(cam, variant)
        res = client.post("/api/register/capture")
        data = res.get_json()
        if data.get("ok"):
            count += 1
    return count


def test_full_registration_and_duplicate_protection(app, cam, lena_img):
    client = app.test_client()

    # --- register Lena as a new person --------------------------------
    res = client.post("/api/register/start", json={
        "mode": "new", "name": "Lena Park", "code": "EMP-L1"})
    assert res.status_code == 200, res.get_json()
    session = res.get_json()["session"]
    assert session["name"] == "Lena Park"

    captured = _capture_samples(app, cam, lena_img, 5)
    assert captured >= 3, "expected at least 3 accepted samples"

    res = client.post("/api/register/finish", json={})
    assert res.status_code == 200, res.get_json()
    person = res.get_json()["person"]
    assert person["name"] == "Lena Park"
    assert person["sample_count"] == captured

    # --- same person ID again is blocked -------------------------------
    res = client.post("/api/register/start", json={
        "mode": "new", "name": "Lena Park", "code": "EMP-L1"})
    assert res.status_code == 409
    assert res.get_json()["code"] == "duplicate_code"

    # --- same face under a different ID is blocked ----------------------
    res = client.post("/api/register/start", json={
        "mode": "new", "name": "Lena Again", "code": "EMP-L2"})
    assert res.status_code == 200
    _capture_samples(app, cam, lena_img, 3)
    res = client.post("/api/register/finish", json={})
    assert res.status_code == 409, res.get_json()
    body = res.get_json()
    assert body["code"] == "duplicate_face"
    assert body["match"]["code"] == "EMP-L1"
    assert len(people.list_people()) == 1   # nothing was saved

    # --- an explicit override registers anyway --------------------------
    res = client.post("/api/register/finish", json={"force": True})
    assert res.status_code == 200
    assert len(people.list_people()) == 2

    # --- invalid input is rejected --------------------------------------
    res = client.post("/api/register/start", json={
        "mode": "new", "name": "", "code": "X"})
    assert res.status_code == 400
    res = client.post("/api/register/start", json={
        "mode": "new", "name": "No Code", "code": ""})
    assert res.status_code == 400


def test_poor_quality_capture_rejected(app, cam, lena_img):
    """Blurry or tiny faces must not be accepted as samples."""
    client = app.test_client()
    client.post("/api/register/start", json={
        "mode": "new", "name": "Quality Test", "code": "Q-1"})

    # a heavily blurred version of Lena -> too_blurry
    blurry = cv2.GaussianBlur(lena_img, (25, 25), 0)
    _feed_image(cam, blurry)
    res = client.post("/api/register/capture")
    data = res.get_json()
    assert data["ok"] is False
    assert data["code"] == "too_blurry"

    # no face at all
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    _feed_image(cam, blank)
    res = client.post("/api/register/capture")
    assert res.get_json()["code"] == "no_face"

    # finally a good sample is accepted
    _feed_image(cam, lena_img)
    res = client.post("/api/register/capture")
    assert res.get_json()["ok"] is True

    # cancel cleans the session
    client.post("/api/register/cancel")
    res = client.get("/api/register/session").get_json()
    assert res["session"] is None


def test_re_enrol_existing_person(app, cam, lena_img):
    client = app.test_client()
    from tests import conftest
    pipe = cam._pipeline
    embs = helpers.embeddings_of(lena_img, pipe, count=2)
    person = people.create_person("R-1", "Re Enrol", embs)
    assert person["sample_count"] == 2

    res = client.post("/api/register/start", json={
        "mode": "existing", "code": "R-1"})
    assert res.status_code == 200
    n = _capture_samples(app, cam, lena_img, 3)
    assert n == 3
    res = client.post("/api/register/finish", json={})
    assert res.status_code == 200
    person2 = res.get_json()["person"]
    assert person2["sample_count"] == 5


# ---------------------------------------------------------------------------
# Records, export, deletion through the API
# ---------------------------------------------------------------------------

def _seed_attendance(cfg):
    p1 = people.create_person("API-1", "Alice",
                              [helpers.random_embedding(seed=1)])
    p2 = people.create_person("API-2", "Bob",
                              [helpers.random_embedding(seed=2)])
    attendance.mark_present(p1["id"], p1["code"], p1["name"], 0.91,
                            now=datetime(2026, 3, 2, 9, 5, 0))
    attendance.mark_present(p2["id"], p2["code"], p2["name"], 0.80,
                            now=datetime(2026, 3, 2, 9, 30, 0))
    attendance.mark_present(p1["id"], p1["code"], p1["name"], 0.95,
                            now=datetime(2026, 3, 3, 9, 0, 0))
    return p1, p2


def test_records_api_and_export(app, cfg):
    _seed_attendance(cfg)
    client = app.test_client()

    data = client.get("/api/records").get_json()
    assert data["total"] == 3

    data = client.get("/api/records?q=alice&date=2026-03-02").get_json()
    assert data["total"] == 1
    assert data["records"][0]["name"] == "Alice"

    meta = client.get("/api/records/meta").get_json()
    assert meta["dates"] == ["2026-03-03", "2026-03-02"]
    assert len(meta["people"]) == 2

    res = client.get("/api/records/export?q=alice")
    assert res.status_code == 200
    text = res.get_data(as_text=True)
    assert "Alice,API-1,2026-03-02" in text
    assert "Alice,API-1,2026-03-03" in text
    assert "Bob" not in text


def test_people_api_and_delete_keeps_records(app, cfg):
    p1, _ = _seed_attendance(cfg)
    client = app.test_client()
    people_data = client.get("/api/people").get_json()["people"]
    assert len(people_data) == 2
    by_code = {p["code"]: p for p in people_data}
    assert by_code["API-1"]["sample_count"] == 1   # face samples, not records
    assert by_code["API-1"]["last_seen"] is not None

    res = client.delete(f"/api/people/{p1['id']}")
    assert res.status_code == 200
    assert len(client.get("/api/people").get_json()["people"]) == 1
    # history remains
    data = client.get("/api/records?q=alice").get_json()
    assert data["total"] == 2
