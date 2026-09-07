"""SQLite database layer.

A fresh connection is opened for every operation and closed afterwards, which
makes this layer safe to call from several threads (the camera thread writes
attendance records while web request threads read them).  WAL journal mode
keeps reads fast and avoids readers blocking the writer.

Schema
------
people            one row per registered person (name + unique person code)
face_embeddings   one row per stored 128-d face embedding (normalized float32)
attendance        one row per attendance event
activity_log      recent recognition activity, shown on the dashboard
settings          runtime settings editable from the Settings page
"""

import sqlite3
import threading
from typing import Any, Optional, Sequence

from . import config as config_module

_DB_PATH: Optional[str] = None
_lock = threading.RLock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   TEXT NOT NULL COLLATE NOCASE UNIQUE,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    embedding   BLOB NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id    INTEGER NOT NULL,
    person_code  TEXT NOT NULL,
    person_name  TEXT NOT NULL,
    date         TEXT NOT NULL,
    time         TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    confidence   REAL NOT NULL,
    status       TEXT NOT NULL DEFAULT 'present'
);

CREATE TABLE IF NOT EXISTS activity_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,
    person_id    INTEGER,
    person_name  TEXT,
    kind         TEXT NOT NULL,
    confidence   REAL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_person ON face_embeddings(person_id);
CREATE INDEX IF NOT EXISTS idx_attendance_person_date ON attendance(person_id, date);
CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
CREATE INDEX IF NOT EXISTS idx_activity_time ON activity_log(timestamp);
"""


def init(db_path: str) -> None:
    """Point the module at a database file and create tables if missing."""
    global _DB_PATH
    _DB_PATH = str(db_path)
    with _connect() as conn:
        conn.executescript(_SCHEMA)
    _ensure_default_settings()


def _connect() -> sqlite3.Connection:
    if _DB_PATH is None:
        raise RuntimeError("Database not initialised - call db.init(path) first.")
    conn = sqlite3.connect(_DB_PATH, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class _Conn:
    """Context manager that always closes the connection."""

    def __init__(self, commit_on_exit: bool):
        self._commit = commit_on_exit
        self.conn = _connect()

    def __enter__(self) -> sqlite3.Connection:
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None and self._commit:
                self.conn.commit()
            elif exc_type is not None:
                self.conn.rollback()
        finally:
            self.conn.close()
        return False


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def query(sql: str, params: Sequence = ()) -> list[sqlite3.Row]:
    with _Conn(False) as conn:
        return conn.execute(sql, params).fetchall()


def query_one(sql: str, params: Sequence = ()) -> Optional[sqlite3.Row]:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: Sequence = ()) -> int:
    """Run a write statement, return the number of affected rows."""
    with _Conn(True) as conn:
        cur = conn.execute(sql, params)
        return cur.rowcount


def execute_lastrowid(sql: str, params: Sequence = ()) -> int:
    with _Conn(True) as conn:
        cur = conn.execute(sql, params)
        return cur.lastrowid


def executemany(sql: str, params: Sequence[Sequence]) -> None:
    """Run a statement for many parameter sets in one transaction."""
    with _Conn(True) as conn:
        conn.executemany(sql, params)


# ---------------------------------------------------------------------------
# Settings (typed, validated against config.SETTINGS_SCHEMA)
# ---------------------------------------------------------------------------

def _ensure_default_settings() -> None:
    defaults = config_module.typed_defaults()
    for key, value in defaults.items():
        execute("INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)",
                (key, str(value)))


def _coerce(key: str, raw: str) -> Any:
    """Convert a stored string into the schema type."""
    meta = config_module.SETTINGS_SCHEMA[key]
    t = meta["type"]
    if t == "int":
        return int(float(raw)) if isinstance(raw, str) and "." in raw else int(raw)
    if t == "float":
        return float(raw)
    if t == "bool":
        return str(raw).lower() in ("1", "true", "yes", "on")
    return str(raw)


def get_setting(key: str) -> Any:
    """Read one setting, falling back to its default."""
    meta = config_module.SETTINGS_SCHEMA.get(key)
    if meta is None:
        raise KeyError(f"Unknown setting: {key}")
    row = query_one("SELECT value FROM settings WHERE key = ?", (key,))
    return _coerce(key, row["value"]) if row else meta["default"]


def all_settings() -> dict:
    out = {}
    for key in config_module.SETTINGS_SCHEMA:
        out[key] = get_setting(key)
    return out


def validate_value(key: str, value: Any) -> Optional[str]:
    """Return an error message if *value* is invalid for *key*, else None."""
    meta = config_module.SETTINGS_SCHEMA.get(key)
    if meta is None:
        return f"Unknown setting '{key}'."
    t = meta["type"]
    try:
        if t == "int":
            value = int(value)
        elif t == "float":
            value = float(value)
    except (TypeError, ValueError):
        return f"'{value}' is not a valid number for '{key}'."
    if t in ("int", "float"):
        if "min" in meta and value < meta["min"]:
            return f"'{key}' must be >= {meta['min']}."
        if "max" in meta and value > meta["max"]:
            return f"'{key}' must be <= {meta['max']}."
    if t == "str" and "choices" in meta and str(value) not in meta["choices"]:
        return f"'{value}' is not a valid choice for '{key}'."
    return None


def set_setting(key: str, value: Any) -> None:
    """Validate and persist a setting."""
    error = validate_value(key, value)
    if error:
        raise ValueError(error)
    execute("INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)))
