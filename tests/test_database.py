"""Tests for the database, people store and attendance rules."""

from datetime import datetime

import pytest

from face_attendance import attendance, db, people
from tests.helpers import random_embedding


def _init(cfg):
    db.init(cfg.db_path)
    return cfg.db_path


def _make_person(code="EMP-1", name="Test Person", n_samples=3):
    embs = [random_embedding(seed=i) for i in range(n_samples)]
    return people.create_person(code, name, embs)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def test_default_settings_present_and_typed(cfg):
    _init(cfg)
    s = db.all_settings()
    assert s["similarity_threshold"] == 0.50
    assert isinstance(s["similarity_threshold"], float)
    assert s["camera_index"] == 0
    assert isinstance(s["camera_index"], int)
    assert s["attendance_mode"] == "once_per_day"


def test_setting_roundtrip_and_validation(cfg):
    _init(cfg)
    db.set_setting("similarity_threshold", 0.61)
    assert db.get_setting("similarity_threshold") == pytest.approx(0.61)
    with pytest.raises(ValueError):
        db.set_setting("similarity_threshold", 1.5)      # out of range
    with pytest.raises(ValueError):
        db.set_setting("camera_index", -2)
    with pytest.raises(ValueError):
        db.set_setting("attendance_mode", "weekly")       # not a choice


# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------

def test_create_and_list_people(cfg):
    _init(cfg)
    p = _make_person()
    people_list = people.list_people()
    assert len(people_list) == 1
    assert people_list[0]["name"] == "Test Person"
    assert people_list[0]["code"] == "EMP-1"
    assert people_list[0]["sample_count"] == 3


def test_duplicate_code_rejected(cfg):
    _init(cfg)
    _make_person()
    with pytest.raises(ValueError, match="already registered"):
        _make_person(code="EMP-1", name="Other")


def test_code_unique_case_insensitive(cfg):
    _init(cfg)
    _make_person(code="emp-1")
    with pytest.raises(ValueError):
        _make_person(code="EMP-1", name="Other")


def test_invalid_input_rejected(cfg):
    _init(cfg)
    with pytest.raises(ValueError):
        people.create_person("", "No ID", [random_embedding()])
    with pytest.raises(ValueError):
        people.create_person("OK-ID", "", [random_embedding()])
    with pytest.raises(ValueError):
        people.create_person("OK-ID", "Name", [])


def test_add_samples_and_cap(cfg):
    _init(cfg)
    p = _make_person(n_samples=1)
    db.set_setting("max_embeddings_per_person", 3)
    p2 = people.add_embeddings(p["id"], [random_embedding(seed=9)], max_total=3)
    assert p2["sample_count"] == 2
    with pytest.raises(ValueError, match="room for 1 more"):
        people.add_embeddings(p["id"],
                              [random_embedding(seed=10),
                               random_embedding(seed=11)], max_total=3)
    # two separate additions that respect the cap both work
    p3 = people.add_embeddings(p["id"], [random_embedding(seed=12)],
                               max_total=3)
    assert p3["sample_count"] == 3


def test_delete_keeps_history(cfg):
    _init(cfg)
    p = _make_person()
    attendance.mark_present(p["id"], p["code"], p["name"], 0.95,
                            now=datetime(2026, 1, 5, 9, 0, 0))
    assert people.delete_person(p["id"]) is True
    assert people.list_people() == []
    # the attendance row survives with the name snapshot
    rows = attendance.records(date="2026-01-05")
    assert len(rows) == 1
    assert rows[0]["name"] == "Test Person"
    assert people.embedding_count() == 0


# ---------------------------------------------------------------------------
# Attendance rules
# ---------------------------------------------------------------------------

def test_mark_present_once_per_day(cfg):
    _init(cfg)
    p = _make_person()
    r1 = attendance.mark_present(p["id"], p["code"], p["name"], 0.9,
                                 now=datetime(2026, 1, 5, 8, 30, 0))
    assert r1["created"] is True
    r2 = attendance.mark_present(p["id"], p["code"], p["name"], 0.9,
                                 now=datetime(2026, 1, 5, 14, 0, 0))
    assert r2["created"] is False          # already present today
    assert "already marked" in r2["message"]
    r3 = attendance.mark_present(p["id"], p["code"], p["name"], 0.9,
                                 now=datetime(2026, 1, 6, 8, 30, 0))
    assert r3["created"] is True           # different day -> new record
    assert len(attendance.records()) == 2


def test_mark_present_cooldown_mode(cfg):
    _init(cfg)
    db.set_setting("attendance_mode", "cooldown")
    db.set_setting("cooldown_minutes", 30)
    p = _make_person()
    t0 = datetime(2026, 1, 5, 8, 0, 0)
    assert attendance.mark_present(p["id"], p["code"], p["name"], 0.9,
                                   now=t0)["created"] is True
    # within the 30 minute window -> blocked
    assert attendance.mark_present(p["id"], p["code"], p["name"], 0.9,
                                   now=datetime(2026, 1, 5, 8, 20, 0))[
                                       "created"] is False
    # after the window -> allowed even on the same day
    assert attendance.mark_present(p["id"], p["code"], p["name"], 0.9,
                                   now=datetime(2026, 1, 5, 8, 45, 0))[
                                       "created"] is True
    assert len(attendance.records()) == 2


def test_concurrent_mark_no_duplicates(cfg):
    """The conditional INSERT must hold even if called twice at once."""
    _init(cfg)
    p = _make_person()
    now = datetime(2026, 1, 5, 9, 0, 0)
    import threading
    results = []
    barrier = threading.Barrier(2)

    def do():
        barrier.wait()
        results.append(attendance.mark_present(p["id"], p["code"],
                                               p["name"], 0.9, now=now))
    threads = [threading.Thread(target=do) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    created = sum(1 for r in results if r["created"])
    assert created == 1
    assert len(attendance.records()) == 1


# ---------------------------------------------------------------------------
# Records / activity / export
# ---------------------------------------------------------------------------

def test_records_filters_and_count(cfg):
    _init(cfg)
    p1 = _make_person(code="AA-1", name="Alice")
    p2 = _make_person(code="BB-2", name="Bob")
    attendance.mark_present(p1["id"], p1["code"], p1["name"], 0.91,
                            now=datetime(2026, 1, 5, 9, 0, 0))
    attendance.mark_present(p2["id"], p2["code"], p2["name"], 0.82,
                            now=datetime(2026, 1, 5, 10, 0, 0))
    attendance.mark_present(p1["id"], p1["code"], p1["name"], 0.93,
                            now=datetime(2026, 1, 6, 9, 0, 0))

    assert attendance.records_count(search="ali") == 2
    assert attendance.records_count(date="2026-01-06") == 1
    assert attendance.records_count(person_db_id=p1["id"]) == 2
    rows = attendance.records(search="ali", limit=5)
    assert all(r["name"] == "Alice" for r in rows)

    dates = attendance.distinct_dates()
    assert dates == ["2026-01-06", "2026-01-05"]


def test_export_csv(cfg):
    _init(cfg)
    p = _make_person()
    attendance.mark_present(p["id"], p["code"], p["name"], 0.9,
                            now=datetime(2026, 1, 5, 9, 0, 0))
    csv_text = attendance.export_csv(date="2026-01-05")
    assert "Name,Person ID,Date,Time,Confidence,Status" in csv_text
    assert "Test Person,EMP-1,2026-01-05" in csv_text


def test_activity_coalescing(cfg):
    _init(cfg)
    now = datetime(2026, 1, 5, 9, 0, 0)
    assert attendance.log_activity("recognized", 1, "Alice", 0.9, now=now) is True
    # same person+kind within cooldown -> no new row
    assert attendance.log_activity("recognized", 1, "Alice", 0.9,
                                   now=now) is False
    # attendance events are never coalesced
    assert attendance.log_activity("attendance", 1, "Alice", 0.9,
                                   now=now) is True
    act = attendance.activity()
    assert len(act) == 2
    assert act[0]["kind"] == "attendance"


def test_dashboard_stats(cfg):
    _init(cfg)
    stats = attendance.dashboard_stats(now=datetime(2026, 1, 5, 12, 0, 0))
    assert stats["registered"] == 0
    p1 = _make_person(code="A", name="Alice")
    _make_person(code="B", name="Bob")
    attendance.mark_present(p1["id"], p1["code"], p1["name"], 0.9,
                            now=datetime(2026, 1, 5, 9, 0, 0))
    stats = attendance.dashboard_stats(now=datetime(2026, 1, 5, 12, 0, 0))
    assert stats["registered"] == 2
    assert stats["present_today"] == 1
    assert stats["absent_today"] == 1
    assert stats["attendance_rate"] == 50.0
