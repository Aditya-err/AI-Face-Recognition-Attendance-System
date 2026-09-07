"""Attendance records, statistics and recognition activity.

Duplicate prevention
--------------------
The mark operation runs as a *single conditional INSERT* so two threads can
never both insert a record for the same person at the same time:

- ``once_per_day`` (default): at most one record per person per calendar day.
- ``cooldown``: at most one record per person per cooldown window (minutes).

Attendance rows keep a snapshot of the person's name/code, so deleting a
person never destroys history.
"""

import csv
import io
from datetime import datetime, timedelta
from typing import Optional

from . import db

STATUS_PRESENT = "present"
ACTIVITY_KINDS = ("recognized", "uncertain", "unknown", "attendance")


# ---------------------------------------------------------------------------
# Marking attendance
# ---------------------------------------------------------------------------

def mark_present(person_db_id: int, code: str, name: str,
                 confidence: float,
                 now: Optional[datetime] = None) -> dict:
    """Try to record attendance for a person. Returns
    ``{created: bool, record: dict|None, message: str}``.

    If a valid record already exists for the current period nothing is
    inserted and ``created`` is False.
    """
    now = now or datetime.now()
    mode = db.get_setting("attendance_mode")
    if mode == "cooldown":
        minutes = int(db.get_setting("cooldown_minutes"))
        cutoff = (now - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S")
        insert_sql = (
            "INSERT INTO attendance(person_id, person_code, person_name, "
            "date, time, timestamp, confidence, status) "
            "SELECT ?, ?, ?, ?, ?, ?, ?, ? WHERE NOT EXISTS ("
            "  SELECT 1 FROM attendance "
            "  WHERE person_id = ? AND timestamp >= ?)")
        params = (person_db_id, code, name, now.strftime("%Y-%m-%d"),
                  now.strftime("%H:%M:%S"), now.strftime("%Y-%m-%dT%H:%M:%S"),
                  float(confidence), STATUS_PRESENT, person_db_id, cutoff)
        dup_msg = (f"{name} was already marked within the last "
                   f"{minutes} minutes.")
    else:  # once_per_day
        today = now.strftime("%Y-%m-%d")
        insert_sql = (
            "INSERT INTO attendance(person_id, person_code, person_name, "
            "date, time, timestamp, confidence, status) "
            "SELECT ?, ?, ?, ?, ?, ?, ?, ? WHERE NOT EXISTS ("
            "  SELECT 1 FROM attendance "
            "  WHERE person_id = ? AND date = ?)")
        params = (person_db_id, code, name, today, now.strftime("%H:%M:%S"),
                  now.strftime("%Y-%m-%dT%H:%M:%S"), float(confidence),
                  STATUS_PRESENT, person_db_id, today)
        dup_msg = f"{name} is already marked present today."

    inserted = db.execute(insert_sql, params) == 1
    if inserted:
        record_id = db.query_one(
            "SELECT id FROM attendance WHERE person_id = ? "
            "ORDER BY timestamp DESC LIMIT 1", (person_db_id,))["id"]
        record = _get_record(record_id)
        return {"created": True, "record": record,
                "message": f"Attendance marked for {name}."}
    return {"created": False, "record": None, "message": dup_msg}


def _get_record(record_id: int) -> Optional[dict]:
    row = db.query_one("SELECT * FROM attendance WHERE id = ?", (record_id,))
    return _record_to_dict(row) if row else None


def _record_to_dict(row) -> dict:
    return {
        "id": row["id"],
        "person_id": row["person_id"],
        "code": row["person_code"],
        "name": row["person_name"],
        "date": row["date"],
        "time": row["time"],
        "timestamp": row["timestamp"],
        "confidence": round(float(row["confidence"]), 3),
        "status": row["status"],
    }


# ---------------------------------------------------------------------------
# Recognition activity log (dashboard feed)
# ---------------------------------------------------------------------------

def log_activity(kind: str, person_id: Optional[int], person_name: str,
                 confidence: Optional[float],
                 now: Optional[datetime] = None) -> bool:
    """Append an activity event.

    For repeated events (same person + same kind) only one row per cooldown
    window is written, so a face staring at the camera for an hour produces
    a handful of rows, not thousands.  ``attendance`` events are never
    coalesced - they are rare and important.
    """
    kind = kind if kind in ACTIVITY_KINDS else "unknown"
    now = now or datetime.now()
    stamp = now.strftime("%Y-%m-%dT%H:%M:%S")
    if kind != "attendance":
        cooldown = float(db.get_setting("activity_log_cooldown_seconds"))
        cutoff = (now - timedelta(seconds=cooldown)).strftime("%Y-%m-%dT%H:%M:%S")
        exists = db.query_one(
            "SELECT 1 FROM activity_log WHERE kind = ? "
            "AND person_name = ? AND timestamp >= ? LIMIT 1",
            (kind, person_name or "", cutoff))
        if exists:
            return False
    db.execute(
        "INSERT INTO activity_log(timestamp, person_id, person_name, kind, "
        "confidence) VALUES(?, ?, ?, ?, ?)",
        (stamp, person_id, person_name, kind, confidence))
    return True


def activity(limit: int = 25) -> list[dict]:
    rows = db.query(
        "SELECT * FROM activity_log ORDER BY timestamp DESC, id DESC LIMIT ?",
        (limit,))
    return [_activity_to_dict(r) for r in rows]


def _activity_to_dict(row) -> dict:
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "person_id": row["person_id"],
        "person_name": row["person_name"],
        "kind": row["kind"],
        "confidence": (round(float(row["confidence"]), 3)
                       if row["confidence"] is not None else None),
    }


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def _build_filters(search: Optional[str], date: Optional[str],
                   person_db_id: Optional[int],
                   status: Optional[str]):
    clauses, params = [], []
    if search:
        clauses.append("(person_name LIKE ? OR person_code LIKE ?)")
        like = f"%{search}%"
        params += [like, like]
    if date:
        clauses.append("date = ?")
        params.append(date)
    if person_db_id:
        clauses.append("person_id = ?")
        params.append(person_db_id)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def records(search: Optional[str] = None, date: Optional[str] = None,
            person_db_id: Optional[int] = None,
            status: Optional[str] = None, limit: int = 50,
            offset: int = 0) -> list[dict]:
    where, params = _build_filters(search, date, person_db_id, status)
    rows = db.query(
        f"SELECT * FROM attendance {where} "
        "ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
        params + [limit, offset])
    return [_record_to_dict(r) for r in rows]


def records_count(search: Optional[str] = None, date: Optional[str] = None,
                  person_db_id: Optional[int] = None,
                  status: Optional[str] = None) -> int:
    where, params = _build_filters(search, date, person_db_id, status)
    row = db.query_one(f"SELECT COUNT(*) AS n FROM attendance {where}", params)
    return int(row["n"])


def distinct_dates() -> list[str]:
    rows = db.query("SELECT DISTINCT date FROM attendance ORDER BY date DESC")
    return [r["date"] for r in rows]


def export_csv(search: Optional[str] = None, date: Optional[str] = None,
               person_db_id: Optional[int] = None,
               status: Optional[str] = None) -> str:
    where, params = _build_filters(search, date, person_db_id, status)
    rows = db.query(
        f"SELECT * FROM attendance {where} ORDER BY timestamp DESC, id DESC",
        params)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Name", "Person ID", "Date", "Time", "Confidence",
                     "Status"])
    for r in rows:
        writer.writerow([r["person_name"], r["person_code"], r["date"],
                         r["time"], round(float(r["confidence"]), 3),
                         "Present"])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Dashboard statistics
# ---------------------------------------------------------------------------

def dashboard_stats(now: Optional[datetime] = None) -> dict:
    now = now or datetime.now()
    today = now.strftime("%Y-%m-%d")
    registered = int(db.query_one("SELECT COUNT(*) AS n FROM people")["n"])
    present_today = int(db.query_one(
        "SELECT COUNT(*) AS n FROM attendance WHERE date = ?", (today,))["n"])
    absent = max(registered - present_today, 0)
    rate = (present_today / registered * 100.0) if registered else 0.0
    recent = records(date=today, limit=8)
    act = activity(limit=10)
    return {
        "registered": registered,
        "present_today": present_today,
        "absent_today": absent,
        "attendance_rate": round(rate, 1),
        "date": today,
        "recent_attendance": recent,
        "activity": act,
    }


def today_ids() -> set[int]:
    """DB ids of people already marked today (used to annotate the UI)."""
    today = datetime.now().strftime("%Y-%m-%d")
    rows = db.query("SELECT DISTINCT person_id FROM attendance WHERE date = ?",
                    (today,))
    return {r["person_id"] for r in rows}
