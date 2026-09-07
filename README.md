# AI Face Recognition Attendance System

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-green.svg)
![Flask](https://img.shields.io/badge/Flask-Web-lightgrey.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
A complete, local, web-based attendance system:

**Register a person → capture their face → generate face embeddings → store →
live camera recognition → confident match → attendance record → dashboard**

No retraining is ever needed when people are added or removed: recognition is
a nearest-neighbour search over stored *face embeddings*, not a trained
classifier over photos.

| | |
|---|---|
| Face detection | YuNet (OpenCV DNN, ONNX) - handles multiple faces |
| Face recognition | SFace 128-d embeddings (OpenCV DNN, ONNX), cosine similarity |
| Web UI | Flask + vanilla JS (no build step), responsive |
| Database | SQLite (single file, survives restarts) |
| Camera | any webcam OpenCV can open, streamed as MJPEG into the browser |

---

## Recognition pipeline (why it is reliable)

```
webcam frame
   │  (every processed frame, cheap)
   ▼
YuNet face detection ──► 0 faces? nothing to do
   │  (several faces supported, each handled independently)
   ▼
is this face new/moved or is the check interval up?   ← NO ─► reuse cached result
   │ YES
   ▼
SFace embedding (128-d)      ← expensive step, so it is rate-limited,
   │                           never run on every single frame
   ▼
cosine similarity vs every registered embedding (vectorized)
   │
   ├─ sim ≥ 0.50 ───────────► recognized:  <name> 97%
   ├─ 0.35 ≤ sim < 0.50 ────► uncertain:   <name>? 42%   (never marks attendance)
   └─ sim <  0.35 ──────────► unknown
```

Three decisions deliberately *lower false positives*:

1. **Embeddings, not photos.** Two faces are the same person only when their
   128-d embedding vectors point the same way.  A different person almost
   never scores above ~0.3, while the same person (rotated/brightness
   changed) typically scores 0.8-1.0.
2. **Confirmation window.** A face must be confidently recognized in
   `confirm_frames` **consecutive** checks (default 2).  One lucky frame can
   never mark attendance, and if the person in front of the camera changes,
   the counter resets.
3. **Uncertainty band.** A weak match is displayed as *low confidence* and is
   never recorded as attendance.

Thresholds are adjustable in **Settings**.  The defaults (match 0.50,
uncertain 0.35) sit far from both the same-person and different-person score
distributions measured with the included models.

### Duplicate registration protection

- A person ID can only be used once.
- Enrolling the **same face** under a different name/ID is refused with a
  clear message naming the existing person.  An explicit override exists for
  genuine edge cases.
- "Re-enrol existing" adds more samples to an already registered person
  (with a per-person sample cap).

---

## Installation

### 1. Install Python and dependencies

Requires **Python 3.9+** (tested on 3.13).  From the project folder:

```bash
pip install -r requirements.txt
```

> `opencv-contrib-python` is the only vision dependency - it contains the
> standard OpenCV modules **plus** the `cv2.FaceDetectorYN` / `cv2.FaceRecognizerSF`
> models used here.  Do **not** also install `opencv-python` (they conflict).

### 2. Install the face models

```bash
python download_models.py
```

This downloads two small ONNX files from the official OpenCV model zoo into
`models/`:

- `face_detection_yunet_2023mar.onnx` (~0.2 MB) - face detector
- `face_recognition_sface_2021dec.onnx` (~37 MB) - face embedding model

No internet is needed afterwards; the models stay local.

### 3. Configure (optional)

Everything has a working default.  To override, copy `.env.example` to `.env`
and edit, or set real environment variables (`FA_HOST`, `FA_PORT`, ...).

### 4. Start the application

```bash
python run.py
```

Open <http://127.0.0.1:8000> in your browser.  The port is configurable
(`python run.py --port 9000`, or `FA_PORT`).

> The app binds to `127.0.0.1` by default.  Use `--host 0.0.0.0` only if you
> deliberately want to expose it on your local network (there is no login).

### 5. Register a person

1. Open **Register Face**.
2. Enter the person's name and ID (employee/student number).
3. Click **Start registration**, then **Add sample** several times
   (default 5).  Samples are quality-checked: a face must be visible, large
   enough and sharp, and only one person may be in the frame.
4. Click **Save registration**.  The face is now enrolled permanently.

To add more samples later: **People → + Add samples** on that person.

### 6. Start recognition

Open **Live Recognition** and point the webcam at the person.  A green box
with the name marks a confident match and **attendance is recorded
automatically** (after the confirmation window).  A name with "?" is low
confidence and is never recorded.  Red "Unknown" means the face is not
registered.

### 7. View attendance

- **Dashboard** - today's counts, today's attendance, recent activity.
- **Attendance Records** - search by name/ID, filter by date/person/status,
  paginated, with **Export CSV**.

---

## How attendance is stored

SQLite database at `data/faceattend.db` (created automatically; git-ignored).

| Table | Contents |
|---|---|
| `people` | one row per person: unique person ID + name |
| `face_embeddings` | one row per captured sample: normalized 128-d float32 embedding (BLOB). *No face photos are stored.* |
| `attendance` | one row per attendance event: person name/ID snapshot, date, time, timestamp, similarity confidence, status |
| `activity_log` | recent recognition events for the dashboard feed (coalesced to avoid flooding) |
| `settings` | runtime settings editable from the Settings page |

Duplicate rules enforced by a single conditional `INSERT` (safe under
concurrency):

- **once per day** (default): one record per person per calendar day.
- **cooldown window**: one record per person per N minutes.

Attendance rows keep a snapshot of the person's name/ID, so **deleting a
person never erases history**.  Deleting a person removes only their face
data.

## Project layout

```
├── run.py                  # entry point: python run.py
├── download_models.py      # fetches the two ONNX models
├── requirements.txt
├── .env.example
├── face_attendance/
│   ├── config.py           # env / .env configuration + settings schema
│   ├── db.py               # SQLite layer, tables, typed settings
│   ├── pipeline.py         # YuNet detector + SFace embeddings (load once)
│   ├── people.py           # person CRUD, embedding gallery, match decisions
│   ├── attendance.py       # attendance records, dedupe, stats, CSV, activity
│   ├── camera.py           # camera thread, MJPEG, live recognition, sessions
│   ├── web.py              # Flask pages + JSON API
│   ├── templates/          # Jinja pages
│   └── static/             # CSS + JS
├── tests/                  # pytest suite (36 tests)
├── models/                 # ONNX files (downloaded)
└── data/                   # runtime database (created on first run)
```

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `similarity_threshold` | 0.50 | similarity at/above this = recognized |
| `uncertain_threshold` | 0.35 | between this and match threshold = low confidence (never marked) |
| `confirm_frames` | 2 | consecutive agreeing checks before attendance |
| `recognition_interval_seconds` | 0.7 | how often embeddings are computed while a face is present |
| `attendance_mode` | `once_per_day` | `once_per_day` or `cooldown` |
| `cooldown_minutes` | 10 | used in cooldown mode |
| `registration_samples` | 5 | target samples per registration |
| `min_face_width` / `min_sharpness` | 90 px / 30 | registration quality gates |
| `max_embeddings_per_person` | 24 | per-person sample cap |
| `camera_index` / width / height / `process_fps` | 0 / 640×480 / 10 | camera behaviour |

## Running the tests

```bash
pip install pytest
python -m pytest tests -v
```

The suite covers registration, duplicate protection, quality rejection,
recognition decisions, confirmation-window attendance, duplicate-prevention,
filters/export, settings validation and data persistence.  Tests that need
the face models are skipped automatically if `models/` is empty.

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| "Camera: error - Could not open camera" | Another app (Zoom, OBS, another script) holds the webcam; close it and click **Restart camera** in Settings. Wrong device? change `camera_index` in Settings. On some systems set `FA_CAMERA_BACKEND=dshow` or `=msmf` in `.env`. |
| "Face models: missing" | Run `python download_models.py`, then **Restart camera**. |
| Nobody is recognized / everything "Unknown" | People must be registered first (Register Face). In poor light raise camera brightness, get closer, and check `similarity_threshold` in Settings. |
| Wrong person / false positives | Raise `similarity_threshold` and/or `confirm_frames`. |
| Person marked twice | Should not happen: switch `attendance_mode` to `once_per_day`. If you see duplicates, it means an old duplicate row exists in the DB - delete it in Records → (no edit UI) or clear `attendance` table manually. |
| Registration says "blurry" constantly | Hold still / improve lighting; `min_sharpness` can be lowered in Settings. |
| Registration says "too small" | Move closer; `min_face_width` can be lowered in Settings. |
| Browser shows a frozen frame | The MJPEG stream reconnects itself; otherwise click **Reconnect camera**. |
| "Multiple faces detected" | Registration allows exactly one person per sample. |
| Port already in use | `python run.py --port 9000`. |
| Camera looks wrong after Settings change | Camera changes need a restart: Settings → Restart camera. |
| Want to start over | Stop the app, delete `data/faceattend.db`, restart. A fresh database is created automatically. |

## Privacy

Only normalized numeric face embeddings are stored - never face photos.  The
webcam stream never leaves your machine (server binds to 127.0.0.1 by
default).  Runtime data lives in `data/` and is git-ignored.
