"""People store: registered persons and their face embeddings.

Responsibilities
----------------
- CRUD for registered people (name + unique person code such as an employee
  or student ID).
- Storing face embeddings (normalized float32 vectors serialized as BLOBs).
- Loading the embedding "gallery" used for recognition (matrix + metadata).
- Duplicate-registration protection:
    * the person code is unique,
    * a new face that strongly matches an *already registered* person is
      rejected, so the same face cannot be quietly enrolled twice under a
      different name/code.
"""

import sqlite3
import threading
from typing import Optional

import numpy as np

from . import db
from .pipeline import l2_normalize

PERSON_CODE_MAX = 40
NAME_MAX = 100


def _row_to_person(row: sqlite3.Row) -> dict:
    keys = row.keys()
    return {
        "id": row["id"],
        "code": row["person_id"],
        "name": row["name"],
        "created_at": row["created_at"],
        "sample_count": row["sample_count"] if "sample_count" in keys else 0,
        "last_seen": row["last_seen"] if "last_seen" in keys else None,
    }


def list_people() -> list[dict]:
    """All people with sample counts and their most recent attendance."""
    rows = db.query(
        """
        SELECT p.id, p.person_id, p.name, p.created_at,
               COUNT(e.id)                          AS sample_count,
               (SELECT MAX(a.timestamp) FROM attendance a
                 WHERE a.person_id = p.id)          AS last_seen
        FROM people p
        LEFT JOIN face_embeddings e ON e.person_id = p.id
        GROUP BY p.id
        ORDER BY p.created_at DESC
        """
    )
    return [_row_to_person(r) for r in rows]


def get_person(person_db_id: int) -> Optional[dict]:
    row = db.query_one(
        """
        SELECT p.id, p.person_id, p.name, p.created_at,
               COUNT(e.id) AS sample_count,
               (SELECT MAX(a.timestamp) FROM attendance a
                 WHERE a.person_id = p.id) AS last_seen
        FROM people p
        LEFT JOIN face_embeddings e ON e.person_id = p.id
        WHERE p.id = ?
        GROUP BY p.id
        """,
        (person_db_id,),
    )
    return _row_to_person(row) if row else None


def find_by_code(code: str) -> Optional[dict]:
    row = db.query_one(
        "SELECT id, person_id, name, created_at FROM people "
        "WHERE person_id = ? COLLATE NOCASE",
        (code.strip(),),
    )
    if row:
        d = dict(row)
        d.setdefault("sample_count", 0)
        d.setdefault("last_seen", None)
        return d
    return None


def _validate_text(label: str, value: str, max_len: int) -> Optional[str]:
    value = (value or "").strip()
    if not value:
        return f"{label} is required."
    if len(value) > max_len:
        return f"{label} must be {max_len} characters or fewer."
    return None


def create_person(code: str, name: str,
                  embeddings: list[np.ndarray]) -> dict:
    """Register a new person. Raises ValueError with a user-friendly message
    when the code is taken or the input is invalid."""
    code = (code or "").strip()
    name = (name or "").strip()
    err = _validate_text("Name", name, NAME_MAX) or _validate_text(
        "Person ID", code, PERSON_CODE_MAX)
    if err:
        raise ValueError(err)
    if find_by_code(code):
        raise ValueError(f"Person ID '{code}' is already registered.")
    if not embeddings:
        raise ValueError("No usable face samples were captured.")

    now = _now()
    person_id = db.execute_lastrowid(
        "INSERT INTO people(person_id, name, created_at) VALUES(?, ?, ?)",
        (code, name, now))
    _insert_embeddings(person_id, embeddings, now)
    invalidate_gallery()
    person = get_person(person_id)
    return person


def add_embeddings(person_id: int, embeddings: list[np.ndarray],
                   max_total: int) -> dict:
    """Add more face samples to an existing person (re-enrolment).

    Caps the total number of stored samples per person.
    """
    person = get_person(person_id)
    if person is None:
        raise ValueError("Person not found.")
    if not embeddings:
        raise ValueError("No usable face samples were captured.")
    room = max_total - person["sample_count"]
    if room <= 0 or len(embeddings) > room:
        raise ValueError(
            f"'{person['name']}' only has room for {max(room, 0)} more face "
            f"samples (max {max_total}). Delete the person and register them "
            "again if you need a fresh enrolment.")
    now = _now()
    _insert_embeddings(person_id, embeddings, now)
    invalidate_gallery()
    return get_person(person_id)


def _insert_embeddings(person_id: int,
                       embeddings: list[np.ndarray], now: str) -> None:
    rows = [(person_id, l2_normalize(e).tobytes(), now) for e in embeddings]
    db.executemany(
        "INSERT INTO face_embeddings(person_id, embedding, created_at) "
        "VALUES(?, ?, ?)", rows)


def delete_person(person_db_id: int) -> bool:
    """Delete a person and their embeddings (attendance history is kept)."""
    person = get_person(person_db_id)
    if person is None:
        return False
    db.execute("DELETE FROM people WHERE id = ?", (person_db_id,))
    invalidate_gallery()
    return True


def embedding_count() -> int:
    row = db.query_one("SELECT COUNT(*) AS n FROM face_embeddings")
    return int(row["n"])


# ---------------------------------------------------------------------------
# Recognition gallery
# ---------------------------------------------------------------------------

class Gallery:
    """In-memory snapshot of every registered face embedding.

    Built once per recognition pass and reused for all faces in that pass, so
    a pass over N faces does not run N SQL queries.
    """

    def __init__(self):
        self.matrix = np.zeros((0, 128), dtype=np.float32)
        self.person_ids: list[int] = []
        self.codes: list[str] = []
        self.names: list[str] = []
        rows = db.query(
            """
            SELECT e.id AS eid, e.embedding, p.id AS pid, p.person_id,
                   p.name
            FROM face_embeddings e
            JOIN people p ON p.id = e.person_id
            """
        )
        if rows:
            self.matrix = np.vstack([
                np.frombuffer(r["embedding"], dtype=np.float32) for r in rows
            ]).reshape(len(rows), 128)
            self.person_ids = [r["pid"] for r in rows]
            self.codes = [r["person_id"] for r in rows]
            self.names = [r["name"] for r in rows]

    def __len__(self):
        return len(self.person_ids)

    def best_match(self, embedding: np.ndarray) -> Optional[dict]:
        """Highest-similarity registered person for one embedding."""
        if len(self.person_ids) == 0:
            return None
        v = l2_normalize(embedding)
        sims = self.matrix @ v
        idx = int(np.argmax(sims))
        return {
            "person_id": self.person_ids[idx],
            "code": self.codes[idx],
            "name": self.names[idx],
            "similarity": float(sims[idx]),
        }


_gallery_cache: Optional["Gallery"] = None
_gallery_lock = threading.RLock()


def invalidate_gallery() -> None:
    """Invalidate the cached embedding gallery so it is rebuilt on next use."""
    global _gallery_cache
    with _gallery_lock:
        _gallery_cache = None


def get_gallery() -> Gallery:
    """Get the cached gallery, or build it if invalid."""
    global _gallery_cache
    with _gallery_lock:
        if _gallery_cache is None:
            _gallery_cache = Gallery()
        return _gallery_cache


def highest_similarity(embedding: np.ndarray) -> Optional[dict]:
    """Match an embedding against every registered person.

    Used both by live recognition and by the duplicate-face check during
    registration.
    """
    return get_gallery().best_match(embedding)


def classify(best: Optional[dict], similarity_threshold: float,
             uncertain_threshold: float):
    """Turn a best match into one of three decisions.

    Returns ``(label, person_or_None, similarity)`` where label is one of
    ``recognized`` / ``uncertain`` / ``unknown``.  Only ``recognized`` may
    ever lead to an attendance record - this is the single place where that
    rule is decided, so it is easy to test and reason about.
    """
    if best is None:
        return "unknown", None, 0.0
    sim = float(best["similarity"])
    if sim >= similarity_threshold:
        return "recognized", best, sim
    if sim >= uncertain_threshold:
        return "uncertain", best, sim
    return "unknown", None, sim


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
